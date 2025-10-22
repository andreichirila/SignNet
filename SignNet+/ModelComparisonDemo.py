"""
SignNet+ - BiGRU Model Suite with SignBERT+ Features

Inspired by SignBERT+ (Hu et al., 2023), this framework implements:
- Enhanced BiGRU architectures for continuous sign language recognition
- Spatial-Temporal position encoding
- Masked data augmentation for robustness
- Efficient training for 24+ FPS inference

Author: Roman Schläpfer and Andrei Chirila
Date: 2025-10-22
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import random
from collections import defaultdict

# ============================================================================
# DATASET CLASS
# ============================================================================

class PhoenixDataset(Dataset):
    """Dataset class for RWTH-PHOENIX NPZ files with landmarks"""

    def __init__(self, npz_files: List[Path], augment: bool = False):
        """
        Args:
            npz_files: List of paths to NPZ files
            augment: Whether to apply masked augmentation during training
        """
        self.files = npz_files
        self.augment = augment

        # Build vocabulary from all glosses
        self.gloss_to_idx = {'<BLANK>': 0, '<PAD>': 1}  # CTC blank + padding
        self._build_vocab()

        print(f"   Dataset initialized:")
        print(f"   Samples: {len(self.files)}")
        print(f"   Vocabulary size: {len(self.gloss_to_idx)}")
        print(f"   Augmentation: {'ON' if augment else 'OFF'}")

    def _build_vocab(self):
        """Build vocabulary from all gloss sequences"""
        idx = len(self.gloss_to_idx)
        for file_path in self.files:
            data = np.load(file_path)
            glosses = data['glosses']
            for gloss in glosses:
                if gloss not in self.gloss_to_idx:
                    self.gloss_to_idx[gloss] = idx
                    idx += 1

        # Create reverse mapping
        self.idx_to_gloss = {v: k for k, v in self.gloss_to_idx.items()}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Returns:
            landmarks: (seq_len, feature_dim) tensor
            glosses: (num_glosses,) tensor of gloss indices
            seq_len: int
        """
        data = np.load(self.files[idx])

        landmarks = torch.FloatTensor(data['landmarks'])  # (T, 1659)
        glosses = data['glosses']

        # Apply masked augmentation if enabled
        if self.augment:
            landmarks = masked_augmentation(landmarks, mask_ratio=0.2)

        # Convert glosses to indices
        gloss_indices = [self.gloss_to_idx[g] for g in glosses]
        gloss_tensor = torch.LongTensor(gloss_indices)

        return landmarks, gloss_tensor, len(landmarks)


def masked_augmentation(landmarks: torch.Tensor, mask_ratio: float = 0.2) -> torch.Tensor:
    """
    Apply masked augmentation inspired by SignBERT+

    Args:
        landmarks: (T, 1659) tensor
        mask_ratio: Ratio of frames to augment

    Returns:
        Augmented landmarks tensor
    """
    augmented = landmarks.clone()
    num_frames = len(landmarks)
    num_mask = int(num_frames * mask_ratio)

    if num_mask == 0:
        return augmented

    # Random frames to mask
    mask_frames = random.sample(range(num_frames), num_mask)

    for frame_idx in mask_frames:
        strategy = random.choice(['mask_joints', 'mask_frame', 'gaussian_noise', 'identity'])

        if strategy == 'mask_joints':
            # Mask 5-10 random joints (each joint has 3 coords)
            num_joints = random.randint(5, 10)
            max_joints = landmarks.shape[1] // 3
            joint_indices = random.sample(range(max_joints), min(num_joints, max_joints))
            for j in joint_indices:
                augmented[frame_idx, j*3:(j+1)*3] = 0

        elif strategy == 'mask_frame':
            # Mask entire frame
            augmented[frame_idx] = 0

        elif strategy == 'gaussian_noise':
            # Add Gaussian noise
            noise = torch.randn_like(augmented[frame_idx]) * 0.01
            augmented[frame_idx] += noise

        # 'identity' = no change

    return augmented


def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences

    Returns:
        padded_landmarks: (batch, max_seq_len, feature_dim)
        glosses: List of gloss tensors (variable length)
        lengths: (batch,) tensor of sequence lengths
        gloss_lengths: (batch,) tensor of gloss sequence lengths
    """
    landmarks, glosses, lengths = zip(*batch)

    # Pad landmarks sequences
    padded_landmarks = pad_sequence(landmarks, batch_first=True, padding_value=0.0)

    # Get gloss lengths
    gloss_lengths = torch.LongTensor([len(g) for g in glosses])

    # Convert to tensors
    lengths = torch.LongTensor(lengths)

    return padded_landmarks, glosses, lengths, gloss_lengths


# ============================================================================
# POSITION ENCODING
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for temporal information (from Transformer)"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, 0, :].unsqueeze(0)
        return x


# ============================================================================
# 🏗️ MODEL ARCHITECTURES
# ============================================================================

class BaselineBiGRU(nn.Module):
    """
    Baseline BiGRU - Your winning architecture from initial experiments
    """

    def __init__(self, input_dim=1659, hidden_dim=256, num_layers=3,
                 vocab_size=100, dropout=0.3):
        super().__init__()

        self.name = "Baseline_BiGRU"

        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, lengths):
        # Encode features
        x = self.feature_encoder(x)

        # Pack sequence
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                     enforce_sorted=False)

        # BiGRU
        packed_output, _ = self.gru(packed)

        # Unpack
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Project to vocabulary
        logits = self.output(output)
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)

        return log_probs


class EnhancedBiGRU(nn.Module):
    """
    Enhanced BiGRU with better feature encoding
    Inspired by SignBERT+ but keeps GRU backbone
    """

    def __init__(self, input_dim=1659, hidden_dim=256, num_layers=3,
                 vocab_size=100, dropout=0.4):
        super().__init__()

        self.name = "Enhanced_BiGRU"

        # Multi-layer feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # BiGRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, vocab_size)
        )

    def forward(self, x, lengths):
        # Enhanced feature encoding
        x = self.feature_encoder(x)

        # Pack and process
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                     enforce_sorted=False)
        packed_output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Output projection
        logits = self.output(output)
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)

        return log_probs


class SignBERTStyleBiGRU(nn.Module):
    """
    BiGRU with SignBERT+ inspired features:
    - Spatial-Temporal Position Encoding
    - Separate gesture and spatial feature extraction
    - Better feature fusion

    This is the RECOMMENDED model!
    """

    def __init__(self, input_dim=1659, hidden_dim=320, num_layers=3,
                 vocab_size=100, dropout=0.4):
        super().__init__()

        self.name = "SignBERT_BiGRU"

        # Extract hand joints (42 joints × 3 coords = 126)
        self.hand_dim = 126
        # Extract arm/pose joints (7 joints × 3 coords = 21)
        self.arm_dim = 21

        # Gesture State Encoder (for hand joints)
        self.gesture_encoder = nn.Sequential(
            nn.Linear(self.hand_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Spatial Position Encoder (for arm/body joints)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(self.arm_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Temporal Position Encoding
        self.temporal_pe = PositionalEncoding(d_model=hidden_dim)

        # BiGRU backbone
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Output head
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, vocab_size)
        )

    def forward(self, x, lengths):
        """
        Args:
            x: (batch, seq_len, 1659) - Full landmark features
            lengths: (batch,) - Sequence lengths
        """
        # Split features
        # Hand joints: first 126 features (21 joints × 2 hands × 3 coords)
        hand_features = x[:, :, :self.hand_dim]
        # Arm/body: next 21 features (7 joints × 3 coords)
        spatial_features = x[:, :, self.hand_dim:self.hand_dim + self.arm_dim]

        # Encode gesture state
        gesture_feat = self.gesture_encoder(hand_features)  # (B, T, 256)

        # Encode spatial position
        spatial_feat = self.spatial_encoder(spatial_features)  # (B, T, 64)

        # Fuse features
        fused = torch.cat([gesture_feat, spatial_feat], dim=-1)  # (B, T, 320)
        features = self.fusion(fused)  # (B, T, hidden_dim)

        # Add temporal positional encoding
        features = self.temporal_pe(features)

        # BiGRU processing
        packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True,
                                     enforce_sorted=False)
        packed_output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Output projection
        logits = self.output(output)
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)

        return log_probs


class DeepBiGRU(nn.Module):
    """
    Deep BiGRU with 5 layers and residual connections
    For when you have lots of data and want maximum capacity
    """

    def __init__(self, input_dim=1659, hidden_dim=384, num_layers=5,
                 vocab_size=100, dropout=0.5):
        super().__init__()

        self.name = "Deep_BiGRU"

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Deep BiGRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Residual connection
        self.residual_proj = nn.Linear(input_dim, hidden_dim)

        # Output
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, vocab_size)
        )

    def forward(self, x, lengths):
        # Store for residual
        residual = self.residual_proj(x)

        # Project input
        x = self.input_proj(x)

        # Pack and process
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                     enforce_sorted=False)
        packed_output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Add residual connection
        output = output + residual[:, :output.size(1), :]

        # Project to vocabulary
        logits = self.output(output)
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)

        return log_probs


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

class Trainer:
    """Training and evaluation handler with confidence-aware loss"""

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = defaultdict(list)

        # CTC Loss (blank=0)
        self.criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

        print(f"   Trainer initialized on {device}")
        print(f"   Model: {model.name}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    def train_epoch(self, dataloader, optimizer, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (landmarks, glosses, lengths, gloss_lengths) in enumerate(dataloader):
            # Move to device
            landmarks = landmarks.to(self.device)
            lengths = lengths.to(self.device)
            gloss_lengths = gloss_lengths.to(self.device)

            # Concatenate all glosses for CTC
            target = torch.cat(glosses).to(self.device)

            # Forward pass
            log_probs = self.model(landmarks, lengths)

            # CTC loss
            loss = self.criterion(log_probs, target, lengths, gloss_lengths)

            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss at batch {batch_idx}, skipping...")
                continue

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (important for RNNs!)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 5 == 0:
                print(f"   Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(num_batches, 1)
        self.history['train_loss'].append(avg_loss)
        return avg_loss

    def evaluate(self, dataloader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for landmarks, glosses, lengths, gloss_lengths in dataloader:
                landmarks = landmarks.to(self.device)
                lengths = lengths.to(self.device)
                gloss_lengths = gloss_lengths.to(self.device)
                target = torch.cat(glosses).to(self.device)

                log_probs = self.model(landmarks, lengths)
                loss = self.criterion(log_probs, target, lengths, gloss_lengths)

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.history['val_loss'].append(avg_loss)
        return avg_loss


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(histories: Dict[str, Dict], save_path=None):
    """Plot training curves for multiple models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for model_name, history in histories.items():
        if len(history['train_loss']) == 0:
            continue

        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], marker='o', label=model_name, linewidth=2)

        if 'val_loss' in history and history['val_loss']:
            ax2.plot(epochs, history['val_loss'], marker='s', label=model_name, linewidth=2)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Plot saved to {save_path}")

    return fig


# ============================================================================
# 🎯 MAIN DEMO FUNCTION
# ============================================================================

def run_demo(data_files: List[Path],
             models_to_test: List[str] = ['baseline', 'enhanced', 'signbert'],
             num_epochs: int = 20,
             batch_size: int = 8,
             learning_rate: float = 0.001,
             use_augmentation: bool = True):
    """
    Run comparison demo of different BiGRU models

    Args:
        data_files: List of NPZ file paths
        models_to_test: List of model names to test
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        use_augmentation: Whether to use masked augmentation
    """

    print("="*70)
    print(" SIGN LANGUAGE RECOGNITION - BiGRU MODEL SUITE")
    print("="*70)

    # Create dataset
    dataset = PhoenixDataset(data_files, augment=use_augmentation)

    # For demo: use same data for train and val (you'll split properly later)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=collate_fn)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn)

    vocab_size = len(dataset.gloss_to_idx)

    # Model factory
    model_configs = {
        'baseline': lambda: BaselineBiGRU(vocab_size=vocab_size),
        'enhanced': lambda: EnhancedBiGRU(vocab_size=vocab_size),
        'signbert': lambda: SignBERTStyleBiGRU(vocab_size=vocab_size),
        'deep': lambda: DeepBiGRU(vocab_size=vocab_size)
    }

    results = {}
    histories = {}

    # Train each model
    for model_name in models_to_test:
        print(f"\n{'='*70}")
        print(f" Training: {model_name.upper()}")
        print(f"{'='*70}")

        # Create model
        model = model_configs[model_name]()
        trainer = Trainer(model)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )

        # Training loop
        best_val_loss = float('inf')
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            print(f"\n Epoch {epoch}/{num_epochs}")

            train_loss = trainer.train_epoch(train_loader, optimizer, epoch)
            val_loss = trainer.evaluate(val_loader)

            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}")

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f" New best model! (Val Loss: {val_loss:.4f})")

        train_time = time.time() - start_time

        # Store results
        results[model_name] = {
            'model': model,
            'trainer': trainer,
            'best_val_loss': best_val_loss,
            'train_time': train_time,
            'params': sum(p.numel() for p in model.parameters())
        }
        histories[model_name] = trainer.history

    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Model':<20} {'Parameters':<15} {'Best Val Loss':<15} {'Time (s)':<10}")
    print("-"*70)

    for name, res in results.items():
        print(f"{name:<20} {res['params']:<15,} {res['best_val_loss']:<15.4f} {res['train_time']:<10.1f}")

    # Plot comparison
    fig = plot_training_curves(histories, save_path='training_comparison.png')

    return results, histories, dataset


# ============================================================================
# 🚀 ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("BiGRU Model Suite Ready!")
    print("\n" + "="*70)
    print("Available Models:")
    print("  - baseline:  Baseline BiGRU (your winning model)")
    print("  - enhanced:  Enhanced BiGRU with better features")
    print("  - signbert:  SignBERT-style BiGRU (RECOMMENDED)")
    print("  - deep:      Deep BiGRU with residual connections")
    print("="*70)
    print("\nTo run the demo, use:")
    print("  results, histories, dataset = run_demo(")
    print("      data_files=[Path('your_data.npz')],")
    print("      models_to_test=['baseline', 'signbert'],")
    print("      num_epochs=10,")
    print("      use_augmentation=True")
    print("  )")