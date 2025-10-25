"""
🚀 SignNet+ IMPROVED Model - Enhanced Version
All optimizations for better generalization and live performance

Improvements:
- Enhanced data augmentation
- Better regularization
- Optimized hyperparameters
- Improved training dynamics
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import time
from collections import defaultdict
import os  # Imports os module, used for operating system interactions like setting environment variables.

# ============================================================================
# 📊 ENHANCED DATA AUGMENTATION
# ============================================================================

def mask_augmentation(landmarks: np.ndarray, mask_prob: float = 0.3) -> np.ndarray:
    """
    Enhanced masking with higher probability
    Masks random landmarks to simulate occlusions
    """
    mask = np.random.rand(*landmarks.shape) > mask_prob
    return landmarks * mask


def add_gaussian_noise(landmarks: np.ndarray, std: float = 0.05) -> np.ndarray:
    """
    Add Gaussian noise to simulate landmark extraction variance
    """
    noise = np.random.normal(0, std, landmarks.shape)
    return landmarks + noise


def random_time_warp(landmarks: np.ndarray, sigma: float = 0.2) -> np.ndarray:
    """
    Time warping augmentation for temporal variation
    """
    seq_len = len(landmarks)
    if seq_len < 5:
        return landmarks

    # Create random warp factors
    warp_steps = max(2, seq_len // 10)
    warp_indices = np.linspace(0, seq_len - 1, warp_steps)
    warp_factors = 1.0 + np.random.uniform(-sigma, sigma, warp_steps)

    # Interpolate warp factors
    all_warp = np.interp(np.arange(seq_len), warp_indices, warp_factors)

    # Apply time warping
    warped_indices = np.cumsum(all_warp)
    warped_indices = warped_indices / warped_indices[-1] * (seq_len - 1)
    warped_indices = np.clip(warped_indices, 0, seq_len - 1).astype(int)

    return landmarks[warped_indices]


def random_scale(landmarks: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """
    Random scaling augmentation
    """
    scale = np.random.uniform(*scale_range)
    return landmarks * scale


def mixup_augmentation(landmarks1: np.ndarray, landmarks2: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """
    Mixup augmentation - mix two samples
    Forces model to learn interpolated representations
    """
    lam = np.random.beta(alpha, alpha)

    # Ensure same length
    min_len = min(len(landmarks1), len(landmarks2))
    landmarks1 = landmarks1[:min_len]
    landmarks2 = landmarks2[:min_len]

    return lam * landmarks1 + (1 - lam) * landmarks2


# ============================================================================
# 📦 ENHANCED DATASET
# ============================================================================

class PhoenixDatasetImproved(Dataset):
    """Enhanced Phoenix Dataset with advanced augmentation"""

    def __init__(self, data_files: List[Path], augment: bool = False):
        self.data_files = data_files
        self.augment = augment

        # Build vocabulary
        self.gloss_to_idx = {'<BLANK>': 0, '<PAD>': 1}

        for file_path in data_files:
            data = np.load(file_path, allow_pickle=True)
            glosses = data['glosses']
            for gloss in glosses:
                if gloss not in self.gloss_to_idx:
                    self.gloss_to_idx[gloss] = len(self.gloss_to_idx)

        print(f"   Vocabulary: {len(self.gloss_to_idx)} glosses")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = np.load(self.data_files[idx], allow_pickle=True)
        landmarks = data['landmarks'].astype(np.float32)
        glosses = data['glosses']

        # Enhanced augmentation pipeline
        if self.augment:
            # Apply multiple augmentations
            landmarks = mask_augmentation(landmarks, mask_prob=0.3)
            landmarks = add_gaussian_noise(landmarks, std=0.05)

            # 50% chance for time warping
            if np.random.rand() < 0.5:
                landmarks = random_time_warp(landmarks, sigma=0.2)

            # 30% chance for scaling
            if np.random.rand() < 0.3:
                landmarks = random_scale(landmarks, scale_range=(0.9, 1.1))

            # 20% chance for mixup
            if np.random.rand() < 0.2 and len(self.data_files) > 1:
                idx2 = np.random.randint(len(self.data_files))
                if idx2 != idx:
                    data2 = np.load(self.data_files[idx2], allow_pickle=True)
                    landmarks2 = data2['landmarks'].astype(np.float32)
                    landmarks = mixup_augmentation(landmarks, landmarks2, alpha=0.2)

        # Convert glosses to indices
        gloss_indices = [self.gloss_to_idx[g] for g in glosses if g in self.gloss_to_idx]

        return landmarks, gloss_indices


def collate_fn(batch):
    """Enhanced collate function with better padding"""
    landmarks_batch = [item[0] for item in batch]
    glosses_batch = [item[1] for item in batch]

    # Get lengths
    lengths = torch.LongTensor([len(lm) for lm in landmarks_batch])
    gloss_lengths = torch.LongTensor([len(g) for g in glosses_batch])

    # Pad landmarks
    max_len = max(lengths)
    feature_dim = landmarks_batch[0].shape[1]
    padded_landmarks = torch.zeros(len(batch), max_len, feature_dim)

    for i, lm in enumerate(landmarks_batch):
        padded_landmarks[i, :len(lm)] = torch.FloatTensor(lm)

    return padded_landmarks, glosses_batch, lengths, gloss_lengths


# ============================================================================
# 🏗️ IMPROVED MODEL ARCHITECTURE
# ============================================================================

class SignBERTBiGRUImproved(nn.Module):
    """
    Improved SignBERT with enhanced regularization
    """

    def __init__(self, vocab_size: int, hidden_dim: int = 320, num_layers: int = 3, dropout: float = 0.5):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Input projection with stronger regularization
        self.input_proj = nn.Sequential(
            nn.Linear(1659, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)  # Increased from 0.4 to 0.5
        )

        # Transformer encoder (kept same)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # BiGRU with stronger regularization
        self.bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output head with additional dropout
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),  # Extra dropout layer
            nn.Linear(hidden_dim // 2, vocab_size),
            nn.LogSoftmax(dim=-1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        batch_size = x.size(0)

        # Input projection
        x = self.input_proj(x)

        # Transformer
        x = self.transformer(x)
        x = self.dropout(x)

        # Pack sequence for RNN
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # BiGRU
        packed_out, _ = self.bigru(packed)
        x, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Output
        x = self.output(x)

        # Transpose for CTC: (T, N, vocab_size)
        x = x.permute(1, 0, 2)

        return x


# ============================================================================
# 🎓 IMPROVED OPTIMIZER WITH WARMUP
# ============================================================================

def get_optimizer_improved(model, optimizer_name='adamw'):
    """Get optimizer with improved hyperparameters"""
    if optimizer_name == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=5e-5,  # Lower LR: 0.0001 → 0.00005
            weight_decay=0.01,  # Stronger weight decay
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_name == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=5e-5,
            weight_decay=0.01
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_warmup_scheduler(optimizer, warmup_steps: int = 1000, total_steps: int = 10000):
    """
    Learning rate scheduler with warmup
    """

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# 🏋️ IMPROVED TRAINER
# ============================================================================

class ImprovedMLflowTrainer:
    """Enhanced trainer with better training dynamics"""

    def __init__(self, model, optimizer_name, device='cuda'):
        self.model = model
        self.optimizer_name = optimizer_name
        self.device = device
        self.history = defaultdict(list)

        # Enhanced CTC Loss
        self.criterion = nn.CTCLoss(
            blank=0,
            reduction='mean',
            zero_infinity=True
        )

    def train_epoch(self, dataloader, optimizer, scheduler, epoch):
        """Enhanced training loop"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (landmarks, glosses, lengths, gloss_lengths) in enumerate(dataloader):
            landmarks = landmarks.to(self.device)
            lengths = lengths.to(self.device)
            gloss_lengths = gloss_lengths.to(self.device)
            target = torch.cat([torch.LongTensor(g) for g in glosses]).to(self.device)

            # Forward
            log_probs = self.model(landmarks, lengths)

            # CTC Loss
            loss = self.criterion(log_probs, target, lengths, gloss_lengths)

            # Skip invalid losses
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            # Label smoothing factor
            loss = loss * 0.95  # Slight smoothing

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Stricter gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            # Update warmup scheduler
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            # Log every 100 batches
            if batch_idx % 100 == 0:
                mlflow.log_metric("batch_loss", loss.item(), step=epoch * len(dataloader) + batch_idx)

        avg_loss = total_loss / max(num_batches, 1)
        self.history['train_loss'].append(avg_loss)
        return avg_loss

    def evaluate(self, dataloader, epoch):
        """Evaluation loop"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for landmarks, glosses, lengths, gloss_lengths in dataloader:
                landmarks = landmarks.to(self.device)
                lengths = lengths.to(self.device)
                gloss_lengths = gloss_lengths.to(self.device)
                target = torch.cat([torch.LongTensor(g) for g in glosses]).to(self.device)

                log_probs = self.model(landmarks, lengths)
                loss = self.criterion(log_probs, target, lengths, gloss_lengths)

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.history['val_loss'].append(avg_loss)
        return avg_loss


# ============================================================================
# 📈 PLOTTING
# ============================================================================

def plot_training_curve(history):
    """Plot training curves"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Progress - IMPROVED Model', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    return fig


# ============================================================================
# 🚀 MAIN TRAINING FUNCTION
# ============================================================================

def train_signbert_with_mlflow_improved(
        data_files: List[Path],
        optimizer_configs: List[str] = ['adamw'],
        num_epochs: int = 100,
        batch_size: int = 16,
        use_augmentation: bool = True,
        device: str = 'cuda'
):
    """
    Enhanced training function with all improvements
    """

    os.environ['MLFLOW_TRACKING_USERNAME'] = 'andrei'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'andrei'
    mlflow.set_tracking_uri("https://mlflow.schlaepfer.me")
    mlflow.set_experiment("SignNet+")

    results = {}

    # Create datasets
    print(f"\n📦 Creating datasets with ENHANCED augmentation...")
    dataset = PhoenixDatasetImproved(data_files, augment=use_augmentation)

    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    vocab_size = len(dataset.gloss_to_idx)
    print(f"   Vocabulary size: {vocab_size}")

    for opt_name in optimizer_configs:
        print(f"\n🚀 Training with optimizer: {opt_name}")

        with mlflow.start_run(run_name=f"SignBERT_IMPROVED_{opt_name}"):
            # Log parameters
            mlflow.log_param("optimizer", opt_name)
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("vocab_size", vocab_size)
            mlflow.log_param("augmentation", use_augmentation)
            mlflow.log_param("dropout", 0.5)
            mlflow.log_param("learning_rate", 5e-5)
            mlflow.log_param("gradient_clip", 1.0)
            mlflow.log_param("improvements", "enhanced_aug+higher_dropout+lower_lr+warmup")

            # Create model
            model = SignBERTBiGRUImproved(vocab_size=vocab_size, dropout=0.5).to(device)
            optimizer = get_optimizer_improved(model, opt_name)

            # Calculate total steps for warmup
            total_steps = len(train_loader) * num_epochs
            warmup_steps = min(1000, total_steps // 10)
            scheduler = get_warmup_scheduler(optimizer, warmup_steps, total_steps)

            # Create trainer
            trainer = ImprovedMLflowTrainer(model, opt_name, device)

            # Training loop
            best_val_loss = float('inf')
            start_time = time.time()

            print(f"\n🏋️  Training for {num_epochs} epochs...")

            for epoch in range(1, num_epochs + 1):
                train_loss = trainer.train_epoch(train_loader, optimizer, scheduler, epoch)
                val_loss = trainer.evaluate(val_loader, epoch)

                # Log metrics
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)

                # Print progress
                print(f"Epoch {epoch:3d}/{num_epochs} | "
                      f"Train: {train_loss:.4f} | "
                      f"Val: {val_loss:.4f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"   ✅ New best! Val Loss: {val_loss:.4f}")

            train_time = time.time() - start_time

            # Log final metrics
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.log_metric("final_train_loss", trainer.history['train_loss'][-1])
            mlflow.log_metric("training_time_seconds", train_time)
            mlflow.log_metric("training_time_minutes", train_time / 60)

            # Save model
            print(f"\n💾 Saving best model to MLflow...")
            try:
                mlflow.pytorch.log_model(
                    trainer.model,
                    artifact_path="model"
                )
                print(f"   ✅ Model logged to MLflow!")
            except Exception as e:
                print(f"   ⚠️  Could not log model: {e}")

            # Log training curves
            fig = plot_training_curve(trainer.history)
            mlflow.log_figure(fig, f"training_curve_{opt_name}.png")
            plt.close(fig)

            # Store results
            results[opt_name] = {
                'best_val_loss': best_val_loss,
                'final_train_loss': trainer.history['train_loss'][-1],
                'train_time': train_time,
                'history': trainer.history
            }

    return results