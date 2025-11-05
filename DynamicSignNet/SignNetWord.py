import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import os
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
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

    def __call__(self, landmarks):
        if np.random.random() > self.prob:
            return landmarks

        augmented = landmarks.copy()

        # Speed variation
        if np.random.random() > 0.5:
            speed = np.random.uniform(0.80, 1.20)
            new_length = max(1, int(len(augmented) / speed))
            indices = np.linspace(0, len(augmented) - 1, new_length)
            augmented = augmented[indices.astype(int)]

        # Noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.008, augmented.shape)
            augmented = augmented + noise

        # Frame dropout
        if np.random.random() > 0.6:
            keep_prob = np.random.uniform(0.90, 0.95)
            mask = np.random.rand(len(augmented)) < keep_prob
            if mask.sum() > 1:
                augmented = augmented[mask]

        return augmented.astype(np.float32)


class SignLanguageDataset(torch.utils.data.Dataset):
    """Sign language dataset with advanced augmentation for landmark sequences."""

    def __init__(self, npz_dir, debug=False, augment=False, augment_prob=0.5):
        self.npz_dir = npz_dir
        self.debug = debug
        self.augment = augment
        self.augment_prob = augment_prob  # Probability of applying augmentation

        # Load data
        self.data = {}
        self.word_to_idx = {}
        self.idx_to_word = {}

        npz_files = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])

        for idx, npz_file in enumerate(npz_files):
            word = npz_file.replace('.npz', '')
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

            npz_path = os.path.join(npz_dir, npz_file)
            npz_data = np.load(npz_path)
            self.data[idx] = npz_data['data']  # Shape: (num_samples, seq_len, 1659)

        self.num_classes = len(self.word_to_idx)
        if self.debug:
            print(f"Loaded {self.num_classes} classes from {npz_dir}")

    def __len__(self):
        total = 0
        for word_data in self.data.values():
            total += len(word_data)
        return total

    def __getitem__(self, idx):
        # Find which class and sample within class
        current_idx = 0
        for class_idx in sorted(self.data.keys()):
            class_data = self.data[class_idx]
            if current_idx + len(class_data) > idx:
                sample_idx = idx - current_idx
                landmarks = class_data[sample_idx].astype(np.float32)

                # Apply augmentation
                if self.augment and np.random.rand() < self.augment_prob:
                    landmarks = self._augment(landmarks)

                landmarks_tensor = torch.from_numpy(landmarks)
                label = torch.tensor(class_idx, dtype=torch.long)

                return landmarks_tensor, label

            current_idx += len(class_data)

    def _augment(self, landmarks):
        """
        Apply random augmentations to landmark sequence.
        landmarks: (seq_len, 1659) array
        """
        seq_len = landmarks.shape[0]

        # 1. Temporal Jitter - Shift frames slightly in time
        if np.random.rand() < 0.4:
            landmarks = self._temporal_jitter(landmarks)

        # 2. Gaussian Noise - Add small random noise to coordinates
        if np.random.rand() < 0.4:
            landmarks = self._gaussian_noise(landmarks)

        # 3. Scaling - Scale all coordinates uniformly
        if np.random.rand() < 0.3:
            landmarks = self._random_scaling(landmarks)

        # 4. Temporal Dropout - Drop some frames randomly
        if np.random.rand() < 0.3:
            landmarks = self._temporal_dropout(landmarks)

        # 5. Smooth Noise - More gradual noise changes over time
        if np.random.rand() < 0.3:
            landmarks = self._smooth_noise(landmarks)

        # 6. Speed Variation - Stretch or compress sequence
        if np.random.rand() < 0.2:
            landmarks = self._speed_variation(landmarks)

        return landmarks

    def _temporal_jitter(self, landmarks):
        """
        Randomly shift landmark frames in time.
        Simulates timing variations in signing.
        """
        seq_len = landmarks.shape[0]
        max_shift = min(3, seq_len // 10)  # Shift by 1-3 frames max

        if max_shift > 0:
            # Create a jittered time axis
            shift = np.random.randint(-max_shift, max_shift + 1)

            if shift > 0:
                landmarks = np.vstack([
                    landmarks[shift:],
                    np.repeat(landmarks[-1:], shift, axis=0)  # Pad with last frame
                ])
            elif shift < 0:
                landmarks = np.vstack([
                    np.repeat(landmarks[0:1], -shift, axis=0),  # Pad with first frame
                    landmarks[:shift]
                ])

        return landmarks

    def _gaussian_noise(self, landmarks, noise_scale=0.02):
        """
        Add Gaussian noise to landmark coordinates.
        noise_scale: standard deviation relative to coordinate values
        """
        noise = np.random.normal(0, noise_scale, landmarks.shape)
        return landmarks + noise

    def _random_scaling(self, landmarks, scale_range=(0.92, 1.08)):
        """
        Randomly scale all coordinates uniformly.
        Simulates different distances from camera.
        """
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return landmarks * scale

    def _temporal_dropout(self, landmarks, drop_prob=0.1):
        """
        Randomly drop some frames from the sequence.
        drop_prob: probability of dropping each frame
        """
        seq_len = landmarks.shape[0]

        # Keep at least 70% of frames
        min_keep = max(3, int(seq_len * 0.7))

        keep_mask = np.random.rand(seq_len) > drop_prob

        # Ensure minimum frames
        if keep_mask.sum() < min_keep:
            keep_indices = np.random.choice(seq_len, min_keep, replace=False)
            keep_mask[:] = False
            keep_mask[keep_indices] = True

        landmarks = landmarks[keep_mask]

        # Linear interpolate to restore sequence length (optional, remove for variable length)
        # This keeps the sequence length constant
        if len(landmarks) < seq_len:
            indices = np.linspace(0, len(landmarks) - 1, seq_len)
            landmarks = np.interp(indices, np.arange(len(landmarks)), landmarks.T).T

        return landmarks

    def _smooth_noise(self, landmarks, noise_scale=0.02):
        """
        Add smooth (correlated) noise that changes gradually over time.
        More realistic than pure Gaussian noise.
        """
        seq_len, num_features = landmarks.shape

        # Generate smooth noise using interpolation
        num_keypoints = max(2, seq_len // 5)
        keypoint_indices = np.linspace(0, seq_len - 1, num_keypoints)

        smooth_noise = np.random.normal(0, noise_scale, (num_keypoints, num_features))

        # Interpolate to sequence length
        time_axis = np.arange(seq_len)
        noise = np.zeros_like(landmarks)

        for feature_idx in range(num_features):
            noise[:, feature_idx] = np.interp(
                time_axis,
                keypoint_indices,
                smooth_noise[:, feature_idx]
            )

        return landmarks + noise

    def _speed_variation(self, landmarks):
        """
        Stretch or compress the sequence in time.
        Simulates different signing speeds.
        """
        seq_len = landmarks.shape[0]
        speed_factor = np.random.uniform(0.85, 1.15)

        new_len = int(seq_len * speed_factor)
        new_len = max(10, new_len)  # Keep minimum length

        # Interpolate to new sequence length
        indices = np.linspace(0, seq_len - 1, new_len)
        old_indices = np.arange(seq_len)

        new_landmarks = np.zeros((new_len, landmarks.shape[1]))
        for feat_idx in range(landmarks.shape[1]):
            new_landmarks[:, feat_idx] = np.interp(
                indices,
                old_indices,
                landmarks[:, feat_idx]
            )

        # Pad or crop back to original length if needed
        if new_len < seq_len:
            padding = np.repeat(new_landmarks[-1:], seq_len - new_len, axis=0)
            new_landmarks = np.vstack([new_landmarks, padding])
        elif new_len > seq_len:
            new_landmarks = new_landmarks[:seq_len]

        return new_landmarks



class RemappedDataset(torch.utils.data.Dataset):
    """
    Remaps labels from original dataset to new indices for top N words.
    Returns (landmarks, new_label, seq_length) tuples.
    """
    def __init__(self, original_dataset, indices, old_to_new_idx):
        self.original_dataset = original_dataset
        self.indices = indices
        self.old_to_new_idx = old_to_new_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get original index
        original_idx = self.indices[idx]

        # Get data from original dataset
        landmarks, old_label = self.original_dataset[original_idx]

        # Remap label
        old_label_item = old_label.item() if hasattr(old_label, 'item') else old_label
        new_label = self.old_to_new_idx[old_label_item]

        # Compute sequence length (number of frames)
        seq_length = landmarks.shape[0]

        # Return as tuple: (landmarks, label, seq_length)
        return landmarks, torch.tensor(new_label, dtype=torch.long), seq_length



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


class PadCollate:
    """Optimized padding collate function."""
    def __init__(self, debug=False):
        self.debug = debug

    def __call__(self, batch):
        """Pad sequences efficiently."""
        landmarks_list = []
        labels_list = []
        seq_lengths_list = []

        for landmarks, label, seq_length in batch:
            landmarks_list.append(landmarks)
            labels_list.append(label)
            seq_lengths_list.append(seq_length)

        seq_lengths_tensor = torch.tensor(seq_lengths_list, dtype=torch.long)
        max_len = seq_lengths_tensor.max().item()

        padded_landmarks = []
        for landmarks in landmarks_list:
            current_len = landmarks.shape[0]
            if current_len < max_len:
                padding = torch.zeros(max_len - current_len, landmarks.shape[1], dtype=landmarks.dtype)
                landmarks = torch.cat([landmarks, padding], dim=0)
            padded_landmarks.append(landmarks)

        landmarks_tensor = torch.stack(padded_landmarks)
        labels_tensor = torch.stack(labels_list)

        if self.debug:
            print(f"Batch shapes: landmarks={landmarks_tensor.shape}, labels={labels_tensor.shape}")

        return landmarks_tensor, labels_tensor, seq_lengths_tensor





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



def train_epoch_interruptible(model, train_loader, criterion, optimizer, device, epoch,
                               interrupt_handler, debug=True):
    """Train for one epoch with interrupt handling."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

    for batch_idx, (landmarks, labels, seq_lengths) in enumerate(pbar):
        # Check for interrupt signal
        if interrupt_handler.interrupted:
            print(f"\n[INTERRUPT] Stopping training at batch {batch_idx}/{len(train_loader)}")
            return total_loss / (batch_idx + 1) if batch_idx > 0 else 0, correct / total if total > 0 else 0, True

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
    return avg_loss, accuracy, False


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


def main():
    """Main training pipeline with graceful shutdown support."""
    print("=" * 80)
    print("SIGN LANGUAGE CLASSIFIER - ENHANCED WITH GRACEFUL SHUTDOWN")
    print("=" * 80)

    # Initialize graceful shutdown handler
    shutdown_handler = GracefulShutdown()
    print("\n[INFO] Press Ctrl+C at any time to gracefully shut down training")

    # ============================================================================
    # MLFLOW SETUP
    # ============================================================================
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'roman'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'SignNet'
    mlflow.set_tracking_uri("https://mlflow.schlaepfer.me")

    EXPERIMENT_NAME = "SignNetWord"
    RUN_NAME = f"Top 50 Words with augment"
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ============================================================================
    # HYPERPARAMETERS
    # ============================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_EPOCHS = 1000
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    HIDDEN_SIZE = 128
    NUM_LSTM_LAYERS = 1
    DROPOUT_RATE = 0.35
    LSTM_DROPOUT = 0.25
    NUM_WORKERS = 8
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2
    NUM_ATTENTION_HEADS = 4

    AUGMENT = True
    AUGMENT_PROBABILITY = 0.6

    EARLY_STOPPING_PATIENCE = 25
    EARLY_STOPPING_MIN_DELTA = 0.0005
    EARLY_STOPPING_METRIC = "val_acc"
    EARLY_STOPPING_MODE = "max"

    NPZ_DIR = "./word_landmarks_extracted"
    MODEL_SAVE_DIR = "./models_enhanced"
    PLOTS_DIR = "./plots_enhanced"
    print(f"\n[CONFIG] Device: {DEVICE}")
    print(f"[CONFIG] Batch size: {BATCH_SIZE}")
    print(f"[CONFIG] Learning rate: {LEARNING_RATE}")
    print(f"[CONFIG] Max epochs: {NUM_EPOCHS}")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    try:
        # ====================================================================
        # START MLFLOW RUN (explicit, not context manager)
        # ====================================================================
        run = mlflow.start_run(log_system_metrics=True, run_name=RUN_NAME)
        print(f"\n✓ MLflow run started: {run.info.run_id}")

        try:
            # ================================================================
            # Log System Information
            # ================================================================
            mlflow.log_param("python_version", platform.python_version())
            mlflow.log_param("pytorch_version", torch.__version__)
            mlflow.log_param("os", platform.system())
            mlflow.log_param("cpu_count", os.cpu_count())
            mlflow.log_param("total_ram_gb", round(psutil.virtual_memory().total / (1024**3), 2))

            if torch.cuda.is_available():
                mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
                mlflow.log_param("cuda_version", torch.version.cuda)
                mlflow.log_param("gpu_memory_gb", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))

            mlflow.log_params({
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "optimizer": "AdamW",
                "num_epochs": NUM_EPOCHS,
                "hidden_size": HIDDEN_SIZE,
                "num_lstm_layers": NUM_LSTM_LAYERS,
                "dropout_rate": DROPOUT_RATE,
                "num_attention_heads": NUM_ATTENTION_HEADS,
                "input_dim": 1659,
                "scheduler": "CosineAnnealingLR",
                "loss_function": "CrossEntropyLoss",
                "architecture": "Simplified_LSTM_Attention",
                "device": str(DEVICE),
                "graceful_shutdown": "enabled",
            })

            # ================================================================
            # STEP 1-5: Load and prepare data
            # ================================================================
            print(f"\n[STEP 1] Loading dataset...")
            dataset = SignLanguageDataset(NPZ_DIR, debug=True, augment=AUGMENT, augment_prob=AUGMENT_PROBABILITY)

            print(f"\n[STEP 2] Analyzing word frequencies...")
            word_counts = Counter()
            for i in range(len(dataset)):
                _, label = dataset[i]
                word = dataset.idx_to_word[label.item()]
                word_counts[word] += 1

            number_of_classes = 50
            top_n_words = [word for word, _ in word_counts.most_common(number_of_classes)]
            print(f"  Top n words: {top_n_words}")
            for idx, (word, count) in enumerate(word_counts.most_common(number_of_classes)):
                print(f"    {idx+1:2}. {word:20} : {count:4} samples")

            print(f"\n[STEP 3] Filtering to top {len(top_n_words)} words...")
            old_to_new_idx = {}
            for new_idx, word in enumerate(top_n_words):
                old_idx = dataset.word_to_idx[word]
                old_to_new_idx[old_idx] = new_idx

            filtered_indices = []
            for i in range(len(dataset)):
                _, label = dataset[i]
                old_label = label.item()
                if old_label in old_to_new_idx:
                    filtered_indices.append(i)

            print(f"\n[STEP 4] Splitting with stratified random split...")
            filtered_labels = []
            for idx in filtered_indices:
                _, label = dataset[idx]
                word = dataset.idx_to_word[label.item()]
                filtered_labels.append(word)

            train_indices, val_indices = train_test_split(
                filtered_indices,
                test_size=0.2,
                random_state=42,
                stratify=filtered_labels
            )

            train_subset = RemappedDataset(dataset, train_indices, old_to_new_idx)
            val_subset = RemappedDataset(dataset, val_indices, old_to_new_idx)
            num_classes = len(top_n_words)

            print(f"\n[STEP 5] Creating data loaders...")
            train_loader = DataLoader(
                train_subset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                collate_fn=PadCollate(debug=False),
                num_workers=NUM_WORKERS,      # ← ADD
                pin_memory=PIN_MEMORY,         # ← ADD
                prefetch_factor=PREFETCH_FACTOR,  # ← ADD
                persistent_workers=True        # ← ADD
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=PadCollate(debug=False),
                num_workers=NUM_WORKERS,       # ← ADD
                pin_memory=PIN_MEMORY,         # ← ADD
                prefetch_factor=PREFETCH_FACTOR,  # ← ADD
                persistent_workers=True        # ← ADD
            )

            print(f"  Train samples: {len(train_indices)}")
            print(f"  Val samples: {len(val_indices)}")

            # ================================================================
            # STEP 6: Build model
            # ================================================================
            print(f"\n[STEP 6] Building model...")
            model = LSTMSignClassifierSimplified(
                input_size=1659,
                hidden_size=HIDDEN_SIZE,
                num_classes=num_classes,
                num_lstm_layers=NUM_LSTM_LAYERS,
                dropout_rate=DROPOUT_RATE,
                lstm_dropout=LSTM_DROPOUT,
                num_attention_heads=NUM_ATTENTION_HEADS,
                debug=True
            ).to(DEVICE)

            # ================================================================
            # STEP 7: Setup training
            # ================================================================
            print(f"\n[STEP 7] Setting up training...")
            criterion = nn.CrossEntropyLoss()

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=8e-4,
                betas=(0.9, 0.999)
            )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=NUM_EPOCHS,
                eta_min=5e-5
            )

            early_stopping = EarlyStopping(
                patience=EARLY_STOPPING_PATIENCE,
                min_delta=EARLY_STOPPING_MIN_DELTA,
                metric=EARLY_STOPPING_METRIC,
                mode=EARLY_STOPPING_MODE
            )

            # ================================================================
            # STEP 8: Training loop
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
                    print(f"\n[INTERRUPTED] Stopping at epoch {epoch+1}")
                    break

                train_loss, train_acc, interrupted = train_epoch_interruptible(
                    model, train_loader, criterion, optimizer, DEVICE, epoch,
                    shutdown_handler, debug=True
                )

                if interrupted:
                    print(f"[INTERRUPTED] Training stopped during epoch {epoch+1}")
                    break

                val_loss, val_acc, interrupted = evaluate_interruptible(
                    model, val_loader, criterion, DEVICE, epoch,
                    shutdown_handler, debug=True
                )

                if interrupted:
                    print(f"[INTERRUPTED] Validation stopped during epoch {epoch+1}")
                    break

                scheduler.step()
                epochs_trained += 1

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    best_model_path = os.path.join(MODEL_SAVE_DIR, "sign_classifier_best.pth")
                    torch.save(model.state_dict(), best_model_path)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)

                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:4}/{NUM_EPOCHS} │ "
                      f"Train Loss: {train_loss:.4f} │ Train Acc: {train_acc:.2%} │ "
                      f"Val Loss: {val_loss:.4f} │ Val Acc: {val_acc:.2%} │ "
                      f"LR: {lr:.2e}")

                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "learning_rate": lr,
                }, step=epoch)

                if early_stopping(val_acc, epoch):
                    print(f"\n[EARLY STOPPING] Training stopped at epoch {epoch+1}")
                    break

            # ================================================================
            # STEP 9: Save results
            # ================================================================
            print("\n" + "="*80)
            print(f"[TRAINING COMPLETE / INTERRUPTED]")
            print(f"  Total epochs trained: {epochs_trained}")
            print(f"  Best Val Accuracy: {best_val_acc:.2%} at epoch {best_epoch+1}")

            if len(train_losses) > 0:
                print(f"  Final Train Loss: {train_losses[-1]:.4f}")
                print(f"  Final Val Loss: {val_losses[-1]:.4f}")
                print(f"  Final Train Acc: {train_accs[-1]:.2%}")
                print(f"  Final Val Acc: {val_accs[-1]:.2%}")

            # Generate plots
            if len(train_losses) > 0:
                print(f"\n[PLOTTING] Generating training curves...")
                plot_path = plot_training_curves(train_losses, val_losses, train_accs, val_accs, PLOTS_DIR)
                mlflow.log_artifact(plot_path)

            final_model_path = os.path.join(MODEL_SAVE_DIR, "sign_classifier_final.pth")
            torch.save(model.state_dict(), final_model_path)
            print(f"  ✓ Final model saved: {final_model_path}")

            mlflow.log_artifact(final_model_path)
            mlflow.pytorch.log_model(model, "model")

            mlflow.set_tags({
                "model_type": "LSTM_Simplified",
                "task": "sign_language_word_classification",
                "status": "completed" if not shutdown_handler.is_interrupted() else "interrupted",
            })

            print("="*80)

            # Send notification
            asyncio.run(send_message(f"Training summary: Best val acc \n\n{best_val_acc:.2%}", CHAT_ID))

        finally:
            # ====================================================================
            # END MLFLOW RUN (explicit)
            # ====================================================================
            mlflow.end_run()
            print(f"\n✓ MLflow run ended")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        mlflow.end_run()


if __name__ == "__main__":
    main()

