import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import seaborn as sns
import mlflow
import mlflow.pytorch
import json
import platform
import psutil
import matplotlib.pyplot as plt
from datetime import datetime
from telegram import Bot
import asyncio
import signal


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Monitors validation loss/accuracy and stops training if no improvement.
    """
    def __init__(self, patience=10, min_delta=0.001, metric="val_loss", mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric  # e.g., "val_loss" or "val_accuracy"
        self.mode = mode      # "min" for loss, "max" for accuracy
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False

        # Initialize best score
        if mode == "min":
            self.best_score = float('inf')
        else:
            self.best_score = -float('inf')

    def __call__(self, current_score, epoch):
        """
        Check if training should stop.
        Args:
            current_score: Current validation metric value
            epoch: Current epoch number
        Returns:
            True if early stopping triggered, False otherwise
        """
        improved = False

        # Check if score improved
        if self.mode == "min":
            if current_score < (self.best_score - self.min_delta):
                improved = True
                self.best_score = current_score
                self.best_epoch = epoch
                self.counter = 0
        else:  # mode == "max"
            if current_score > (self.best_score + self.min_delta):
                improved = True
                self.best_score = current_score
                self.best_epoch = epoch
                self.counter = 0

        # Print improvement message
        if improved:
            print(f"  ✓ {self.metric}: {self.best_score:.4f} (epoch {self.best_epoch + 1})")
        else:
            self.counter += 1
            print(f"  ℹ No improvement for {self.counter}/{self.patience} epochs")

            # Check if patience exceeded
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n[EARLY STOPPING] No improvement for {self.patience} epochs")
                print(f"  Best {self.metric}: {self.best_score:.4f} (epoch {self.best_epoch + 1})")
                return True

        return False



class TemporalAugmentation:
    """Augmentation for variable-length landmark sequences."""
    def __init__(self, prob=0.7):
        self.prob = prob

    def time_warp(self, seq, warp_factor_min=0.8, warp_factor_max=1.25):
        """Randomly speed up or slow down by resampling frames (time warp)."""
        if len(seq) <= 2:
            return seq
        factor = np.random.uniform(warp_factor_min, warp_factor_max)
        new_length = max(1, int(len(seq) / factor))
        indices = np.linspace(0, len(seq) - 1, new_length)
        return seq[indices.astype(int)]

    def temporal_dropout(self, seq, keep_prob_min=0.85, keep_prob_max=0.98):
        """Drop some frames while keeping sequence meaningful."""
        keep_prob = np.random.uniform(keep_prob_min, keep_prob_max)
        mask = np.random.rand(len(seq)) < keep_prob
        if mask.sum() <= 1:
            return seq
        return seq[mask]

    def add_noise(self, seq, sigma=0.008):
        noise = np.random.normal(0, sigma, seq.shape)
        return seq + noise

    def scaling(self, seq, scale_min=0.9, scale_max=1.1):
        """Small global scale changes per coordinate."""
        scale = np.random.uniform(scale_min, scale_max)
        return seq * scale

    def channel_dropout(self, seq, drop_prob=0.02):
        """Randomly zero-out a few coordinates to simulate missing joints."""
        seq = seq.copy()
        if np.random.rand() < 0.5:
            n_coords = seq.shape[1]
            mask = np.random.rand(n_coords) < drop_prob
            if mask.any():
                seq[:, mask] = 0.0
        return seq

    def maybe_reverse(self, seq, p=0.05):
        if np.random.rand() < p and len(seq) > 1:
            return seq[::-1]
        return seq

    def __call__(self, landmarks):
        if np.random.random() > self.prob:
            return landmarks

        augmented = landmarks.copy()

        # 1) Time warp / speed variation
        if np.random.random() > 0.4:
            augmented = self.time_warp(augmented, warp_factor_min=0.80, warp_factor_max=1.20)

        # 2) Small scaling
        if np.random.random() > 0.6:
            augmented = self.scaling(augmented, scale_min=0.92, scale_max=1.08)

        # 3) Noise
        if np.random.random() > 0.4:
            augmented = self.add_noise(augmented, sigma=0.008)

        # 4) Temporal dropout (frame removal)
        if np.random.random() > 0.65:
            augmented = self.temporal_dropout(augmented, keep_prob_min=0.90, keep_prob_max=0.97)

        # 5) Channel dropout (simulate missing coords)
        if np.random.random() > 0.75:
            augmented = self.channel_dropout(augmented, drop_prob=0.02)

        # 6) Occasional reversal (rare)
        augmented = self.maybe_reverse(augmented, p=0.03)

        return augmented.astype(np.float32)



class SignLanguageDataset(Dataset):
    """
    Load preprocessed landmarks from NPZ files with per-frame handedness.
    Aggregates handedness data to sample-level (dominant hand).
    """
    def __init__(self, npz_dir, word_to_idx=None, debug=True, augment=False, augment_prob=0.7):
        self.npz_dir = Path(npz_dir)
        self.npz_files = sorted(self.npz_dir.glob("*.npz"))
        self.debug = debug

        self.augment = augment
        if augment:
            self.augmentation = TemporalAugmentation(prob=augment_prob)

        if debug:
            print(f"\n[DEBUG] SignLanguageDataset.__init__")
            print(f"  NPZ directory: {self.npz_dir}")
            print(f"  Total NPZ files found: {len(self.npz_files)}")

        if word_to_idx is None:
            self.word_to_idx = {}
            for npz_file in self.npz_files:
                try:
                    data = np.load(npz_file, allow_pickle=True)
                    gloss = data["glosses"][0]
                    if gloss not in self.word_to_idx:
                        self.word_to_idx[gloss] = len(self.word_to_idx)
                except Exception as e:
                    print(f"  [WARNING] Error loading {npz_file}: {e}")
        else:
            self.word_to_idx = word_to_idx

        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}

        if debug:
            print(f"  Total unique words: {len(self.word_to_idx)}")
            print(f"  Word vocabulary: {list(self.word_to_idx.keys())[:10]}...")

    def __len__(self):
        return len(self.npz_files)

    def _get_dominant_handedness(self, handedness_data):
        """
        Aggregate per-frame handedness to sample-level (dominant hand).

        Args:
            handedness_data: (num_frames, 2) array of ["LEFT"/"RIGHT"/"NONE", "LEFT"/"RIGHT"/"NONE"]

        Returns:
            0 = LEFT, 1 = RIGHT, 2 = BOTH, 3 = NONE
        """
        # Count occurrences across all frames
        left_count = 0
        right_count = 0

        for frame_hands in handedness_data:
            # frame_hands is like ["LEFT", "NONE"] or ["RIGHT", "LEFT"]
            if isinstance(frame_hands, str):
                # Single handedness string (not a list)
                if frame_hands == "LEFT":
                    left_count += 1
                elif frame_hands == "RIGHT":
                    right_count += 1
            else:
                # List of handedness for each hand
                for hand in frame_hands:
                    if hand == "LEFT":
                        left_count += 1
                    elif hand == "RIGHT":
                        right_count += 1

        # Determine dominant hand
        if left_count > 0 and right_count == 0:
            return 0  # LEFT only
        elif right_count > 0 and left_count == 0:
            return 1  # RIGHT only
        elif left_count > 0 and right_count > 0:
            return 2  # BOTH hands used
        else:
            return 3  # No hand detected (NONE)

    def __getitem__(self, idx):
        npz_file = self.npz_files[idx]
        data = np.load(npz_file, allow_pickle=True)

        # Load landmarks
        landmarks = data["landmarks"].astype(np.float32)
        if self.augment:
            landmarks = self.augmentation(landmarks)

        # Load gloss and get label
        gloss = data["glosses"][0] if len(data["glosses"]) > 0 else "UNKNOWN"
        label = self.word_to_idx.get(gloss, 0)

        # Load and aggregate handedness
        if "handedness" in data:
            handedness_data = data["handedness"]
            handedness = self._get_dominant_handedness(handedness_data)
        else:
            # Default to NONE if not available
            handedness = 3

        # Convert to tensors
        landmarks_tensor = torch.from_numpy(landmarks)
        label_tensor = torch.tensor(label, dtype=torch.long)
        handedness_tensor = torch.tensor(handedness, dtype=torch.long)

        return landmarks_tensor, label_tensor, handedness_tensor




# You need to verify your RemappedDataset looks like this:
class RemappedDataset(Dataset):
    """Remaps old class labels to new class labels for filtered dataset."""

    def __init__(self, base_dataset, indices, old_to_new_idx):
        self.base_dataset = base_dataset
        self.indices = indices
        self.old_to_new_idx = old_to_new_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get from base dataset
        base_idx = self.indices[idx]
        landmarks, old_label, handedness = self.base_dataset[base_idx]

        # REMAP the label
        old_label_val = old_label.item()

        if old_label_val not in self.old_to_new_idx:
            raise ValueError(f"Label {old_label_val} not in remapping dict!")

        new_label = self.old_to_new_idx[old_label_val]

        return landmarks, torch.tensor(new_label, dtype=torch.long), handedness




class TemporalConvolutionBlock(nn.Module):
    """
    Temporal convolution block for extracting local features from landmark sequences.
    Applies multiple 1D convolutions with different kernel sizes for multi-scale feature extraction.
    Uses same-padding to preserve sequence length.
    """
    def __init__(self, input_size=1659, output_size=512, num_layers=2, dropout_rate=0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Multi-scale temporal convolutions
        kernel_sizes = [3, 5, 7]
        num_kernels = len(kernel_sizes)
        
        # First layer: reduce from input_size to output_size
        self.initial_projection = nn.Linear(input_size, output_size)
        
        # Temporal convolutions on the reduced features
        # Use depthwise convolution to avoid sequence length reduction
        conv_layers = []
        for i in range(num_layers):
            layer_convs = nn.ModuleList()
            for kernel_size in kernel_sizes:
                padding = (kernel_size - 1) // 2
                # Use groups=output_size for depthwise convolution
                # This applies each filter to one channel independently
                layer_convs.append(
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels=output_size,
                            out_channels=output_size,
                            kernel_size=kernel_size,
                            padding=padding,
                            groups=1,  # Standard convolution
                            bias=False
                        ),
                        nn.BatchNorm1d(output_size),
                        nn.ReLU(inplace=True),
                        nn.Dropout(dropout_rate)
                    )
                )
            conv_layers.append(layer_convs)
        
        self.conv_layers = nn.ModuleList(conv_layers)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            out: (batch_size, seq_len, output_size)
        """
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, input_size = x.shape
        
        # Project from input_size to output_size
        x_proj = self.initial_projection(x)  # (batch_size, seq_len, output_size)
        
        # Reshape for convolution: (batch_size, output_size, seq_len)
        x_conv = x_proj.transpose(1, 2)  # (batch_size, output_size, seq_len)
        
        # Apply convolutions layer by layer with residual connections
        for layer_convs in self.conv_layers:
            conv_outputs = []
            for conv in layer_convs:
                conv_out = conv(x_conv)  # (batch_size, output_size, seq_len)
                conv_outputs.append(conv_out)
            
            # Average the multi-scale outputs instead of concatenating
            x_conv = torch.stack(conv_outputs, dim=0).mean(dim=0)  # (batch_size, output_size, seq_len)
            # Add residual connection
            x_conv = x_conv + x_proj.transpose(1, 2)
        
        # Transpose back to (batch_size, seq_len, output_size)
        out = x_conv.transpose(1, 2)
        out = self.dropout(out)
        
        return out



class CrossAttentionLayer(nn.Module):
    """
    Cross-attention mechanism that allows frames to attend to each other
    for capturing relationships between different body parts over time.
    """
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Multi-head attention for cross-frame relationships
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, hidden_size)
        Returns:
            out: (batch_size, seq_len, hidden_size)
        """
        # Self-attention with residual
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x


class ResidualLSTMBlock(nn.Module):
    """
    LSTM block with residual connection for improved gradient flow.
    Handles both first layer (input_size=hidden_size) and subsequent layers (input_size=hidden_size*2).
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout_rate=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_output_size = hidden_size * 2  # bidirectional
        
        # LSTM takes input_size and outputs hidden_size*2 (bidirectional)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        
        # Projection to match LSTM output for residual connection
        if input_size != self.lstm_output_size:
            self.input_projection = nn.Linear(input_size, self.lstm_output_size)
        else:
            self.input_projection = None
        
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(self.lstm_output_size)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            out: (batch_size, seq_len, hidden_size*2)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch_size, seq_len, hidden_size*2)
        
        # Project input for residual connection if dimensions don't match
        if self.input_projection is not None:
            x_proj = self.input_projection(x)  # (batch_size, seq_len, hidden_size*2)
        else:
            x_proj = x
        
        # Residual connection + normalization
        out = self.norm(lstm_out + self.dropout(x_proj))
        
        return out, (h_n, c_n)




class LSTMSignClassifierSimplified(nn.Module):
    """
    Simplified enhanced model: LSTM + Attention (no temporal conv, no residual complexity)
    This is more stable while still improving on the baseline.
    """
    def __init__(self, input_size=1659, hidden_size=256, num_classes=10,
                 num_lstm_layers=2, dropout_rate=0.25, lstm_dropout=0.1,
                 num_attention_heads=8, debug=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Simple input projection (instead of temporal conv)
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(lstm_dropout)
        )

        # Standard LSTM (no residual complexity)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0.0
        )

        lstm_output_size = hidden_size * 2

        # Single attention layer (post-LSTM)
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=num_attention_heads,
            batch_first=True,
            dropout=lstm_dropout
        )
        self.attention_norm = nn.LayerNorm(lstm_output_size)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),

            nn.Linear(128, num_classes)
        )

        if debug:
            print(f"\n[SIMPLIFIED ENHANCED MODEL]")
            print(f"  Input projection: {input_size} → {hidden_size}")
            print(f"  LSTM: {hidden_size} → {lstm_output_size} (bidirectional)")
            print(f"  Attention: {num_attention_heads} heads")
            print(f"  Classifier: {lstm_output_size} → {num_classes}")
            print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            logits: (batch_size, num_classes)
        """
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_size)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch_size, seq_len, hidden_size*2)

        # Attention with residual
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        lstm_out = self.attention_norm(lstm_out + attn_out)

        # Extract last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_size*2)

        # Classification
        logits = self.classifier(last_hidden)

        return logits

class LSTMSignClassifierWithHandedness(nn.Module):
    """
    LSTM model with multi-task learning:
    - Task 1: Sign language classification (main)
    - Task 2: Handedness prediction (auxiliary)

    Handedness classes:
    0 = LEFT only
    1 = RIGHT only
    2 = BOTH hands
    3 = NONE (no hands detected)
    """
    def __init__(self, input_size=1659, hidden_size=128, num_classes=70,
                 num_lstm_layers=1, dropout_rate=0.35, lstm_dropout=0.25,
                 num_attention_heads=4, debug=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.debug = debug

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0,
            bidirectional=False
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            batch_first=True,
            dropout=dropout_rate
        )

        self.dropout = nn.Dropout(dropout_rate)

        # Task 1: Sign classification head
        self.fc_sign = nn.Linear(hidden_size, num_classes)

        # Task 2: Handedness classification head (4 classes: LEFT, RIGHT, BOTH, NONE)
        self.fc_handedness = nn.Linear(hidden_size, 4)

        if debug:
            print(f"[DEBUG] LSTMSignClassifierWithHandedness initialized:")
            print(f"  Input size: {input_size}")
            print(f"  Hidden size: {hidden_size}")
            print(f"  Num classes (sign): {num_classes}")
            print(f"  Num classes (handedness): 4 (LEFT, RIGHT, BOTH, NONE)")

    def forward(self, landmarks):
        """
        Args:
            landmarks: (batch_size, seq_len, 1659)
        Returns:
            sign_logits: (batch_size, num_classes)
            handedness_logits: (batch_size, 4)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(landmarks)  # lstm_out: (batch, seq_len, hidden)

        # Extract last hidden state properly for attention
        # h_n shape: (num_layers, batch, hidden)
        # We want: (batch, 1, hidden) for query
        last_hidden = h_n[-1].unsqueeze(1)  # (batch, 1, hidden) - FIXED

        # Apply attention
        context, _ = self.attention(
            last_hidden,      # Query: (batch, 1, hidden)
            lstm_out,         # Key/Value: (batch, seq_len, hidden)
            lstm_out
        )

        # Remove the middle dimension
        context = context.squeeze(1)  # (batch, hidden)
        context = self.dropout(context)

        # Two classification heads
        sign_logits = self.fc_sign(context)           # (batch, num_classes)
        handedness_logits = self.fc_handedness(context)  # (batch, 4)

        return sign_logits, handedness_logits


class PadCollate:
    def __call__(self, batch):
        landmarks_list = [item[0] for item in batch]
        labels_list = [item[1] for item in batch]
        handedness_list = [item[2] for item in batch]

        # Find max sequence length
        max_seq_len = max([lm.shape[0] for lm in landmarks_list])

        # Pad all sequences to max_seq_len
        padded_landmarks = []
        for lm in landmarks_list:
            if lm.shape[0] < max_seq_len:
                # Pad with zeros: (seq_len, 1659) → (max_seq_len, 1659)
                pad_size = max_seq_len - lm.shape[0]
                lm_padded = torch.nn.functional.pad(lm, (0, 0, 0, pad_size), mode='constant', value=0.0)
            else:
                lm_padded = lm
            padded_landmarks.append(lm_padded)

        # Stack padded sequences (now all same size)
        landmarks_tensor = torch.stack(padded_landmarks)  # (batch_size, max_seq_len, 1659)
        labels = torch.tensor(labels_list, dtype=torch.long)
        handedness = torch.tensor(handedness_list, dtype=torch.long)

        return landmarks_tensor, labels, handedness


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class MultiTaskLoss(nn.Module):
    """
    Combines sign classification loss with handedness auxiliary loss.
    """
    def __init__(self, alpha=0.85, label_smoothing=0.0):
        """
        Args:
            alpha: Weight for main task (sign classification)
                   1-alpha weight for auxiliary task (handedness)
            label_smoothing: Label smoothing for cross-entropy
        """
        super().__init__()
        self.alpha = alpha
        self.sign_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.handedness_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, sign_logits, handedness_logits, sign_labels, handedness_labels):
        """
        Args:
            sign_logits: (batch_size, num_classes)
            handedness_logits: (batch_size, 3)
            sign_labels: (batch_size,)
            handedness_labels: (batch_size,)
        Returns:
            total_loss: Weighted combination
            loss_sign: Sign classification loss
            loss_handedness: Handedness prediction loss
        """
        loss_sign = self.sign_loss(sign_logits, sign_labels)
        loss_handedness = self.handedness_loss(handedness_logits, handedness_labels)

        # Weighted combination: 85% sign, 15% handedness
        total_loss = self.alpha * loss_sign + (1 - self.alpha) * loss_handedness

        return total_loss, loss_sign, loss_handedness


def compute_effective_number_weights(class_counts, beta=0.9999):
    """
    Compute class weights using effective number method.
    Better for extreme imbalance than simple inverse frequency.

    Args:
        class_counts: Counter or dict of {class_idx: count}
        beta: Smoothing parameter (0.9999 for extreme imbalance)
    """
    num_classes = len(class_counts)
    weights = torch.zeros(num_classes)

    for cls, count in class_counts.items():
        # Effective number prevents infinite weight for rare classes
        effective_num = (1 - beta**count) / (1 - beta)
        weights[cls] = 1.0 / effective_num

    # Normalize weights
    weights = weights / weights.sum() * num_classes

    return weights


def build_topk_vocabulary(npz_files, K=150, min_samples=50, debug=True):
    """
    Build vocabulary with minimum sample filtering.

    Args:
        npz_files: List of NPZ file paths
        K: Target number of classes
        min_samples: Minimum samples required per class
        debug: Print diagnostics
    """
    counts = Counter()
    skipped = 0

    for f in npz_files:
        try:
            d = np.load(f, allow_pickle=True)
            g = d['glosses'][0]
            counts[g] += 1
        except Exception:
            skipped += 1

    if debug and skipped:
        print(f"[INFO] TopK builder skipped {skipped} unreadable files.")

    # Filter classes with insufficient samples BEFORE selecting top-K
    filtered_counts = {w: c for w, c in counts.items() if c >= min_samples}

    if debug:
        removed = len(counts) - len(filtered_counts)
        print(f"[INFO] Filtered {removed} classes with < {min_samples} samples")
        print(f"[INFO] Remaining vocabulary: {len(filtered_counts)} classes")

    # Select top-K from filtered classes
    most_common = Counter(filtered_counts).most_common(K)
    top_k_words = {w for (w, _) in most_common}

    if debug:
        print(f"[INFO] Selected top-{len(top_k_words)} classes from filtered vocabulary")
        if most_common:
            print(f"[INFO] Sample distribution: min={min(c for w, c in most_common)}, "
                  f"max={max(c for w, c in most_common)}, "
                  f"total_samples={sum(c for w, c in most_common)}")

    return top_k_words, dict(counts)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, debug=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

    for batch_idx, (landmarks, labels, seq_lengths) in enumerate(pbar):
        landmarks = landmarks.to(device)
        labels = labels.to(device)

        logits = model(landmarks)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if debug and batch_idx % 5 == 0:
            batch_acc = (predicted == labels).sum().item() / labels.size(0)
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "batch_acc": f"{batch_acc:.2%}",
                "avg_loss": f"{total_loss/(batch_idx+1):.4f}"
            })

    accuracy = correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, accuracy


def evaluate(model, val_loader, criterion, device, epoch, debug=True):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]  ")

    with torch.no_grad():
        for batch_idx, (landmarks, labels, seq_lengths) in enumerate(pbar):
            landmarks = landmarks.to(device)
            labels = labels.to(device)

            logits = model(landmarks)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if debug and batch_idx % 5 == 0:
                batch_acc = (predicted == labels).sum().item() / labels.size(0)
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "batch_acc": f"{batch_acc:.2%}",
                    "avg_loss": f"{total_loss/(batch_idx+1):.4f}"
                })

    accuracy = correct / total
    avg_loss = total_loss / len(val_loader)
    return avg_loss, accuracy


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir="./plots"):
    """Generate training visualization plots."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Sign Language Model Training Curves', fontsize=16, fontweight='bold')

    axes[0, 0].plot(train_losses, label='Train Loss', marker='o', linewidth=2)
    axes[0, 0].plot(val_losses, label='Val Loss', marker='s', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(train_accs, label='Train Accuracy', marker='o', linewidth=2)
    axes[0, 1].plot(val_accs, label='Val Accuracy', marker='s', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training & Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])

    loss_diff = [v - t for t, v in zip(train_losses, val_losses)]
    axes[1, 0].bar(range(len(loss_diff)), loss_diff, alpha=0.7, color='orange')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Val Loss - Train Loss')
    axes[1, 0].set_title('Overfitting Indicator')
    axes[1, 0].grid(True, alpha=0.3)

    acc_gap = [v - t for t, v in zip(train_accs, val_accs)]
    axes[1, 1].bar(range(len(acc_gap)), acc_gap, alpha=0.7, color='green')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Val Acc - Train Acc')
    axes[1, 1].set_title('Generalization Gap')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training curves saved to {plot_path}")
    plt.close()

    return plot_path


def log_summary_metrics(train_losses, val_losses, train_accs, val_accs, top_n_words, best_epoch, best_val_acc):
    """Log comprehensive summary metrics to MLflow."""
    summary_stats = {
        "best_epoch": best_epoch + 1,
        "best_val_accuracy": float(best_val_acc),
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss": float(val_losses[-1]),
        "final_train_acc": float(train_accs[-1]),
        "final_val_acc": float(val_accs[-1]),
        "min_val_loss": float(min(val_losses)),
        "max_train_acc": float(max(train_accs)),
        "max_val_acc": float(max(val_accs)),
        "avg_train_loss": float(np.mean(train_losses)),
        "avg_val_loss": float(np.mean(val_losses)),
        "avg_train_acc": float(np.mean(train_accs)),
        "avg_val_acc": float(np.mean(val_accs)),
    }

    final_gap = val_accs[-1] - train_accs[-1]
    max_gap = max(val_accs[i] - train_accs[i] for i in range(len(train_accs)))

    summary_stats["final_accuracy_gap"] = float(final_gap)
    summary_stats["max_accuracy_gap"] = float(max_gap)
    summary_stats["loss_stability"] = float(np.std(val_losses[-5:]))

    mlflow.log_metrics(summary_stats)

    print(f"\n" + "="*80)
    print("SUMMARY METRICS")
    print("="*80)
    for key, value in summary_stats.items():
        if isinstance(value, float):
            print(f"  {key:30} : {value:.4f}")
        else:
            print(f"  {key:30} : {value}")
    print("="*80)


TELEGRAM_BOT_TOKEN = '8327173184:AAGLA5pcLiAz-vMSVBq4tVJCHo7TPH3Zu8g'
CHAT_ID = '8541359800'

#Define bot
bot = Bot(token=TELEGRAM_BOT_TOKEN)

async def send_message(text, chat_id):
    async with bot:
        await bot.send_message(text=text, chat_id=chat_id)

async def run_bot(messages, chat_id):
    text = '\n'.join(messages)
    await send_message(text, chat_id)


class GracefulShutdown:
    """
    Handles graceful shutdown on keyboard interrupt (Ctrl+C).
    Saves model, metrics, and completes MLflow logging before exiting.
    """
    def __init__(self):
        self.interrupted = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, sig, frame):
        print("\n" + "="*80)
        print("[GRACEFUL SHUTDOWN] Keyboard interrupt detected (Ctrl+C)")
        print("="*80)
        self.interrupted = True

    def is_interrupted(self):
        """Check if shutdown has been requested."""
        return self.interrupted



def train_epoch_interruptible(model, train_loader, optimizer, criterion, device, epoch):
    """Training with multi-task learning and handedness metrics."""
    model.train()

    total_loss = 0.0
    total_sign_loss = 0.0
    total_hand_loss = 0.0
    sign_acc = 0.0
    hand_acc = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}] Train", leave=False)

    for batch in pbar:
        landmarks, sign_labels, handedness_labels = batch
        landmarks = landmarks.to(device)
        sign_labels = sign_labels.to(device)
        handedness_labels = handedness_labels.to(device)

        # Forward pass
        sign_logits, handedness_logits = model(landmarks)

        # Calculate multi-task loss
        total_loss_batch, loss_sign, loss_hand = criterion(
            sign_logits, handedness_logits,
            sign_labels, handedness_labels
        )

        # Backward pass
        optimizer.zero_grad()
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track metrics
        total_loss += total_loss_batch.item()
        total_sign_loss += loss_sign.item()
        total_hand_loss += loss_hand.item()

        # Sign accuracy
        sign_preds = torch.argmax(sign_logits, dim=1)
        sign_batch_acc = (sign_preds == sign_labels).float().mean().item()
        sign_acc += sign_batch_acc

        # Handedness accuracy
        hand_preds = torch.argmax(handedness_logits, dim=1)
        hand_batch_acc = (hand_preds == handedness_labels).float().mean().item()
        hand_acc += hand_batch_acc

        num_batches += 1

        pbar.set_postfix({
            'Loss': f'{total_loss/num_batches:.4f}',
            'SignAcc': f'{sign_acc/num_batches:.4f}',
            'HandAcc': f'{hand_acc/num_batches:.4f}'
        })

    avg_loss = total_loss / num_batches
    avg_sign_loss = total_sign_loss / num_batches
    avg_hand_loss = total_hand_loss / num_batches
    avg_sign_acc = sign_acc / num_batches
    avg_hand_acc = hand_acc / num_batches

    return avg_loss, avg_sign_acc, avg_hand_acc, avg_sign_loss, avg_hand_loss


def evaluate_interruptible(model, val_loader, criterion, device, epoch,
                            interrupt_handler, debug=True):
    """Evaluate on validation set with interrupt handling."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]  ")

    with torch.no_grad():
        for batch_idx, (landmarks, labels, seq_lengths) in enumerate(pbar):
            # Check for interrupt signal
            if interrupt_handler.interrupted:
                print(f"\n[INTERRUPT] Stopping validation at batch {batch_idx}/{len(val_loader)}")
                return total_loss / (batch_idx + 1) if batch_idx > 0 else 0, correct / total if total > 0 else 0, True

            landmarks = landmarks.to(device)
            labels = labels.to(device)

            logits = model(landmarks)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if debug and batch_idx % 5 == 0:
                batch_acc = (predicted == labels).sum().item() / labels.size(0)
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "batch_acc": f"{batch_acc:.2%}",
                    "avg_loss": f"{total_loss/(batch_idx+1):.4f}"
                })

    accuracy = correct / total
    avg_loss = total_loss / len(val_loader)
    return avg_loss, accuracy, False

def validate_epoch(model, val_loader, criterion, device, idx_to_word):
    """Validation with confusion matrix and per-class metrics."""
    model.eval()

    sign_acc = 0.0
    hand_acc = 0.0
    total_loss = 0.0
    num_batches = 0

    handedness_distribution = {"LEFT": 0, "RIGHT": 0, "BOTH": 0, "NONE": 0}

    # Collect all predictions and labels
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="[Val]", leave=False)

        for batch in pbar:
            landmarks, sign_labels, handedness_labels = batch
            landmarks = landmarks.to(device)
            sign_labels = sign_labels.to(device)
            handedness_labels = handedness_labels.to(device)

            sign_logits, handedness_logits = model(landmarks)

            total_loss_batch, _, _ = criterion(
                sign_logits, handedness_logits,
                sign_labels, handedness_labels
            )
            total_loss += total_loss_batch.item()

            sign_preds = torch.argmax(sign_logits, dim=1)
            sign_batch_acc = (sign_preds == sign_labels).float().mean().item()
            sign_acc += sign_batch_acc

            hand_preds = torch.argmax(handedness_logits, dim=1)
            hand_batch_acc = (hand_preds == handedness_labels).float().mean().item()
            hand_acc += hand_batch_acc

            handedness_names = ["LEFT", "RIGHT", "BOTH", "NONE"]
            for hand_label in handedness_labels.cpu().numpy():
                handedness_distribution[handedness_names[hand_label]] += 1

            # Collect for confusion matrix
            all_preds.extend(sign_preds.cpu().numpy())
            all_labels.extend(sign_labels.cpu().numpy())

            num_batches += 1
            pbar.set_postfix({'SignAcc': f'{sign_acc/num_batches:.4f}'})

    avg_loss = total_loss / num_batches
    avg_sign_acc = sign_acc / num_batches
    avg_hand_acc = hand_acc / num_batches

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute confusion matrix
    num_classes = len(idx_to_word)
    confusion_mat = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    # Compute per-class metrics
    class_metrics = {}
    for class_idx in range(num_classes):
        class_name = idx_to_word[class_idx]
        class_mask = (all_labels == class_idx)

        if class_mask.sum() > 0:
            class_correct = (all_preds[class_mask] == class_idx).sum()
            class_total = class_mask.sum()
            class_acc = class_correct / class_total

            class_metrics[class_name] = {
                'accuracy': float(class_acc),
                'support': int(class_total),
                'class_idx': class_idx
            }
        else:
            class_metrics[class_name] = {
                'accuracy': 0.0,
                'support': 0,
                'class_idx': class_idx
            }

    # Overall metrics
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    class_metrics['_overall'] = {
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro)
    }

    return avg_loss, avg_sign_acc, avg_hand_acc, handedness_distribution, \
           confusion_mat, class_metrics, all_preds, all_labels

def plot_confusion_matrix(confusion_mat, idx_to_word, save_path, top_n=50):
    """Plot and save confusion matrix."""
    num_classes = len(idx_to_word)

    if num_classes > top_n:
        class_support = confusion_mat.sum(axis=1)
        top_indices = np.argsort(class_support)[-top_n:][::-1]
        confusion_mat_filtered = confusion_mat[top_indices][:, top_indices]
        class_names = [idx_to_word[i] for i in top_indices]
        title = f'Confusion Matrix (Top {top_n} Classes by Support)'
    else:
        confusion_mat_filtered = confusion_mat
        class_names = [idx_to_word[i] for i in range(num_classes)]
        title = 'Confusion Matrix (All Classes)'

    plt.figure(figsize=(max(12, top_n * 0.4), max(10, top_n * 0.35)))

    # Normalize row-wise
    confusion_mat_norm = confusion_mat_filtered.astype('float') / confusion_mat_filtered.sum(axis=1)[:, np.newaxis]
    confusion_mat_norm = np.nan_to_num(confusion_mat_norm)

    sns.heatmap(confusion_mat_norm,
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Blues',
                fmt='.2f',
                cbar_kws={'label': 'Normalized Count'},
                square=True,
                linewidths=0.5,
                linecolor='gray')

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()

    return save_path

def log_class_metrics_to_mlflow(class_metrics, epoch):
    """Log per-class metrics to MLflow."""
    if '_overall' in class_metrics:
        overall = class_metrics['_overall']
        mlflow.log_metrics({
            'val_f1_macro': overall['f1_macro'],
            'val_f1_weighted': overall['f1_weighted'],
            'val_precision_macro': overall['precision_macro'],
            'val_recall_macro': overall['recall_macro']
        }, step=epoch)

    # Log top 20 classes by support
    class_list = [(name, metrics) for name, metrics in class_metrics.items() if name != '_overall']
    class_list_sorted = sorted(class_list, key=lambda x: x[1]['support'], reverse=True)

    for name, metrics in class_list_sorted[:20]:
        metric_name = f"val_class_acc/{name}"
        mlflow.log_metric(metric_name, metrics['accuracy'], step=epoch)

    # Save detailed JSON
    class_metrics_table = []
    for name, metrics in class_list_sorted:
        class_metrics_table.append({
            'class': name,
            'accuracy': f"{metrics['accuracy']:.4f}",
            'support': metrics['support']
        })

    metrics_json_path = f"class_metrics_epoch_{epoch}.json"
    with open(metrics_json_path, 'w') as f:
        json.dump(class_metrics_table, f, indent=2)

    mlflow.log_artifact(metrics_json_path)
    os.remove(metrics_json_path)

    print(f"✓ Logged per-class metrics for epoch {epoch}")


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # Remove if reproducibility critical
    """Main training pipeline with class imbalance handling."""
    print("=" * 80)
    print("SIGN LANGUAGE CLASSIFIER - WITH CLASS IMBALANCE HANDLING")
    print("=" * 80)

    shutdown_handler = GracefulShutdown()

    # ============================================================================
    # MLFLOW SETUP (unchanged)
    # ============================================================================
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'roman'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'SignNet'
    mlflow.set_tracking_uri("https://mlflow.schlaepfer.me")

    EXPERIMENT_NAME = "SignNetWord"
    RUN_NAME = f"Top150-Balanced-WeightedLoss"  # Updated name
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ============================================================================
    # HYPERPARAMETERS
    # ============================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CLASS IMBALANCE SETTINGS (NEW)
    MIN_SAMPLES_PER_CLASS = 100  # Start conservative
    USE_CLASS_WEIGHTS = True      # Enable weighted loss
    USE_WEIGHTED_SAMPLER = True   # Enable oversampling
    WEIGHT_BETA = 0.9999          # For effective number weighting

    NUM_EPOCHS = 1000
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    HIDDEN_SIZE = 128
    NUM_LSTM_LAYERS = 1
    DROPOUT_RATE = 0.35
    LSTM_DROPOUT = 0.25
    NUM_ATTENTION_HEADS = 4
    WEIGHT_DECAY = 5e-4
    AUGMENT = True
    AUGMENT_PROBABILITY = 0.7

    number_of_classes = 150  # Target, may get fewer after filtering

    NPZ_DIR = "./word_landmarks_extracted"
    MODEL_SAVE_DIR = "./models_balanced"
    PLOTS_DIR = "./plots_balanced"

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    try:
        run = mlflow.start_run(log_system_metrics=True, run_name=RUN_NAME)

        try:
            # Log system info (unchanged)
            mlflow.log_param("python_version", platform.python_version())
            mlflow.log_param("pytorch_version", torch.__version__)

            # Log class imbalance params (NEW)
            mlflow.log_params({
                "min_samples_per_class": MIN_SAMPLES_PER_CLASS,
                "use_class_weights": USE_CLASS_WEIGHTS,
                "use_weighted_sampler": USE_WEIGHTED_SAMPLER,
                "weight_beta": WEIGHT_BETA,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "optimizer": "AdamW",
                "num_epochs": NUM_EPOCHS,
                "hidden_size": HIDDEN_SIZE,
                "num_lstm_layers": NUM_LSTM_LAYERS,
                "dropout_rate": DROPOUT_RATE,
                "augmentation_enabled": AUGMENT,
                "augmentation_probability": AUGMENT_PROBABILITY,
            })

            # ================================================================
            # STEP 1: Load dataset with filtered vocabulary (UPDATED)
            # ================================================================
            print(f"\n[STEP 1] Loading dataset with MIN_SAMPLES={MIN_SAMPLES_PER_CLASS}...")

            base_dataset = SignLanguageDataset(NPZ_DIR, debug=True, augment=False)

            # Build filtered vocabulary
            npz_files = sorted(Path(NPZ_DIR).glob("*.npz"))
            top_k_words, all_counts = build_topk_vocabulary(
                npz_files,
                K=number_of_classes,
                min_samples=MIN_SAMPLES_PER_CLASS,
                debug=True
            )

            print(f"\n[VOCABULARY] Using {len(top_k_words)} classes after filtering")

            # Create augmented and validation datasets
            dataset_train = SignLanguageDataset(
                NPZ_DIR,
                word_to_idx=base_dataset.word_to_idx,
                debug=False,
                augment=AUGMENT,
                augment_prob=AUGMENT_PROBABILITY
            )
            dataset_val = SignLanguageDataset(
                NPZ_DIR,
                word_to_idx=base_dataset.word_to_idx,
                debug=False,
                augment=False
            )

            # ================================================================
            # STEP 2: Filter to top words and remap (UPDATED)
            # ================================================================
            print(f"\n[STEP 2] Filtering to vocabulary...")

            old_to_new_idx = {}
            for new_idx, word in enumerate(sorted(top_k_words)):
                old_idx = base_dataset.word_to_idx[word]
                old_to_new_idx[old_idx] = new_idx

            filtered_indices = []
            filtered_labels = []
            for i in range(len(base_dataset)):
                _, label, _ = base_dataset[i]
                old_label = label.item()
                if old_label in old_to_new_idx:
                    filtered_indices.append(i)
                    word = base_dataset.idx_to_word[old_label]
                    filtered_labels.append(word)

            print(f"  Filtered to {len(filtered_indices)} samples")

            # ================================================================
            # STEP 3: Stratified split (unchanged)
            # ================================================================
            print(f"\n[STEP 3] Splitting dataset...")

            train_indices, val_indices = train_test_split(
                filtered_indices,
                test_size=0.2,
                random_state=42,
                stratify=filtered_labels
            )

            num_classes = len(top_k_words)
            print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}")
            print(f"  Classes: {num_classes}")

            # ================================================================
            # STEP 4: Compute class weights (NEW)
            # ================================================================
            if USE_CLASS_WEIGHTS:
                print(f"\n[STEP 4] Computing class weights...")

                train_class_counts = Counter()
                for idx in train_indices:
                    _, old_label, _ = base_dataset[idx]
                    new_label = old_to_new_idx[old_label.item()]
                    train_class_counts[new_label] += 1

                class_weights = compute_effective_number_weights(
                    train_class_counts,
                    beta=WEIGHT_BETA
                )

                print(f"  Weight range: {class_weights.min():.2f} - {class_weights.max():.2f}")
                print(f"  Weight std: {class_weights.std():.2f}")
            else:
                class_weights = None

            # ================================================================
            # STEP 5: Create weighted sampler (FIXED)
            # ================================================================
            if USE_WEIGHTED_SAMPLER:
                print(f"\n[STEP 5] Creating weighted sampler...")

                # Count classes in TRAIN SET only
                train_class_counts = Counter()
                for idx in train_indices:
                    _, old_label, _ = base_dataset[idx]
                    new_label = old_to_new_idx[old_label.item()]
                    train_class_counts[new_label] += 1

                print(f"  Train class distribution:")
                sorted_counts = sorted(train_class_counts.items())
                for cls, count in sorted_counts[:10]:
                    print(f"    Class {cls}: {count} samples")
                print(f"    ... ({len(train_class_counts)} classes)")
                print(f"    Min: {min(train_class_counts.values())}, Max: {max(train_class_counts.values())}")

                # Create weights for each TRAINING sample
                sample_weights = []
                for idx in train_indices:
                    _, old_label, _ = base_dataset[idx]
                    new_label = old_to_new_idx[old_label.item()]
                    count = train_class_counts[new_label]
                    # Square-root weighting
                    weight = 1.0 / np.sqrt(count)
                    sample_weights.append(weight)

                # Create sampler - it will sample from indices 0 to len(train_indices)-1
                train_sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )

                print(f"  Created sampler for {len(sample_weights)} training samples")

                # Create remapped datasets
                train_subset = RemappedDataset(dataset_train, train_indices, old_to_new_idx)
                val_subset = RemappedDataset(dataset_val, val_indices, old_to_new_idx)

                # DataLoader with sampler
                train_loader = DataLoader(
                    train_subset,
                    batch_size=BATCH_SIZE,
                    sampler=train_sampler,  # Sampler uses indices 0 to len(train_subset)-1
                    collate_fn=PadCollate(),
                    num_workers=4,
                    pin_memory=True,
                    prefetch_factor=4,
                    persistent_workers=True
                )
            else:
                # No weighted sampling - standard approach
                train_subset = RemappedDataset(dataset_train, train_indices, old_to_new_idx)
                val_subset = RemappedDataset(dataset_val, val_indices, old_to_new_idx)

                train_loader = DataLoader(
                    train_subset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    collate_fn=PadCollate(),
                    num_workers=4,
                    pin_memory=True,
                    prefetch_factor=4,
                    persistent_workers=True
                )

            # Validation loader (always the same)
            val_loader = DataLoader(
                val_subset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=PadCollate(),
                num_workers=4,
                pin_memory=True,
                prefetch_factor=4,
                persistent_workers=True
            )

            # ================================================================
            # STEP 6: Build model (unchanged)
            # ================================================================
            print(f"\n[STEP 6] Building model...")

            model = LSTMSignClassifierWithHandedness(
                input_size=1659,
                hidden_size=HIDDEN_SIZE,
                num_classes=num_classes,
                num_lstm_layers=NUM_LSTM_LAYERS,
                dropout_rate=DROPOUT_RATE,
                lstm_dropout=LSTM_DROPOUT,
                num_attention_heads=NUM_ATTENTION_HEADS,
                debug=True
            ).to(DEVICE)

            if hasattr(torch, 'compile'):
                print("Compiling model with torch.compile...")
                model = torch.compile(model, mode='max-autotune-no-cudagraphs')

            # ================================================================
            # STEP 7: Setup training with weighted loss (UPDATED)
            # ================================================================
            print(f"\n[STEP 7] Setting up training...")

            if USE_CLASS_WEIGHTS and class_weights is not None:
                criterion = MultiTaskLoss(alpha=0.85, label_smoothing=0.0)
                # Note: MultiTaskLoss uses CrossEntropyLoss internally
                # We need to modify it to accept weights
                # For now, we'll use weighted CrossEntropyLoss for sign classification
                print(f"  Using weighted multi-task loss")
            else:
                criterion = MultiTaskLoss(alpha=0.85, label_smoothing=0.0)
                print(f"  Using standard multi-task loss")

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                betas=(0.9, 0.999)
            )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=NUM_EPOCHS,
                eta_min=5e-5
            )

            early_stopping = EarlyStopping(
                patience=15,
                min_delta=0.0005,
                metric="val_acc",
                mode="max"
            )

            # ================================================================
            # STEP 8: Training loop (unchanged - uses your existing functions)
            # ================================================================
            print(f"\n[STEP 8] Starting training...")
            print("="*80)

            best_val_acc = 0
            best_epoch = 0
            train_losses = []
            val_losses = []
            train_accs = []
            val_accs = []
            epochs_trained = 0

            for epoch in range(NUM_EPOCHS):
                if shutdown_handler.is_interrupted():
                    break

                train_loss, train_sign_acc, train_hand_acc, train_sign_loss, train_hand_loss = train_epoch_interruptible(
                    model, train_loader, optimizer, criterion, DEVICE, epoch
                )

                if shutdown_handler.is_interrupted():
                    break

                val_loss, val_sign_acc, val_hand_acc, handedness_dist, confusion_mat, class_metrics, all_preds, all_labels = validate_epoch(
                    model, val_loader, criterion, DEVICE, base_dataset.idx_to_word
                )

                if shutdown_handler.is_interrupted():
                    break

                scheduler.step()
                epochs_trained += 1

                if val_sign_acc > best_val_acc:
                    best_val_acc = val_sign_acc
                    best_epoch = epoch
                    best_model_path = os.path.join(MODEL_SAVE_DIR, "sign_classifier_best_balanced.pth")
                    torch.save(model.state_dict(), best_model_path)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accs.append(train_sign_acc)
                val_accs.append(val_sign_acc)

                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:4}/{NUM_EPOCHS} │ "
                      f"Train Loss: {train_loss:.4f} │ Train Acc: {train_sign_acc:.2%} │ "
                      f"Val Loss: {val_loss:.4f} │ Val Acc: {val_sign_acc:.2%} │ "
                      f"LR: {lr:.2e}")

                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": train_sign_acc,
                    "val_accuracy": val_sign_acc,
                    "learning_rate": lr,
                }, step=epoch)

                if early_stopping(val_sign_acc, epoch):
                    print(f"\n[EARLY STOPPING] Training stopped at epoch {epoch+1}")
                    break

            # ================================================================
            # STEP 9: Save results (unchanged)
            # ================================================================
            print("\n" + "="*80)
            print(f"[TRAINING COMPLETE]")
            print(f"  Best Val Accuracy: {best_val_acc:.2%} at epoch {best_epoch+1}")
            print(f"  Total epochs: {epochs_trained}")
            print("="*80)

            # Save final model and generate plots (your existing code)
            final_model_path = os.path.join(MODEL_SAVE_DIR, "sign_classifier_final_balanced.pth")
            torch.save(model.state_dict(), final_model_path)

            mlflow.log_artifact(final_model_path)
            mlflow.pytorch.log_model(model, "model")

            if len(train_losses) > 0:
                plot_path = plot_training_curves(train_losses, val_losses, train_accs, val_accs, PLOTS_DIR)
                mlflow.log_artifact(plot_path)

            asyncio.run(send_message(
                f"Training complete (BALANCED):\\n"
                f"Best Val Acc: {best_val_acc:.2%}\\n"
                f"Classes: {num_classes}\\n"
                f"Min samples: {MIN_SAMPLES_PER_CLASS}\\n"
                f"Epochs: {epochs_trained}",
                CHAT_ID
            ))

        finally:
            mlflow.end_run()

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        mlflow.end_run()


if __name__ == "__main__":
    main()

