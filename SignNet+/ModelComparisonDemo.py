"""
Sign Language Recognition SignNet+ [Model Comparison Demo]

This script allows you to quickly experiment with different architectures
for continuous sign language recognition using the RWTH-PHOENIX dataset.

Author: Roman Schläpfer und Andrei Chirila
Date: 2025-10-22
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import time
from collections import defaultdict


# ============================================================================
# DATASET CLASS
# ============================================================================

class PhoenixDataset(Dataset):
    """Dataset class for RWTH-PHOENIX NPZ files"""

    def __init__(self, npz_files: List[Path]):
        """
        Args:
            npz_files: List of paths to NPZ files
        """
        self.files = npz_files

        # Build vocabulary from all glosses
        self.gloss_to_idx = {'<BLANK>': 0, '<PAD>': 1}  # CTC blank + padding
        self._build_vocab()

        print(f"📚 Dataset initialized:")
        print(f"   Samples: {len(self.files)}")
        print(f"   Vocabulary size: {len(self.gloss_to_idx)}")

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

        # Convert glosses to indices
        gloss_indices = [self.gloss_to_idx[g] for g in glosses]
        gloss_tensor = torch.LongTensor(gloss_indices)

        return landmarks, gloss_tensor, len(landmarks)


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
# MODEL ARCHITECTURES
# ============================================================================

class BaselineModel(nn.Module):
    """Simple LSTM baseline"""

    def __init__(self, input_dim=1659, hidden_dim=256, num_layers=2,
                 vocab_size=100, dropout=0.3):
        super().__init__()

        self.name = "Baseline_LSTM"

        # Feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Output layer for CTC
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, lengths):
        """
        Args:
            x: (batch, seq_len, input_dim)
            lengths: (batch,)
        Returns:
            log_probs: (seq_len, batch, vocab_size) for CTC
        """
        # Project features
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)

        # Pack padded sequence
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)

        # LSTM forward
        packed_output, _ = self.lstm(packed)

        # Unpack
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Project to vocabulary
        logits = self.output(output)  # (batch, seq_len, vocab_size)

        # Transpose for CTC: (seq_len, batch, vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)

        return log_probs


class BiLSTMModel(nn.Module):
    """Bidirectional LSTM - usually better for sequence labeling"""

    def __init__(self, input_dim=1659, hidden_dim=256, num_layers=3,
                 vocab_size=100, dropout=0.3):
        super().__init__()

        self.name = "BiLSTM"

        # Feature extraction
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,  # Divide by 2 because bidirectional
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Output layer
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, lengths):
        # Encode features
        x = self.feature_encoder(x)

        # Pack sequence
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)

        # BiLSTM
        packed_output, _ = self.bilstm(packed)

        # Unpack
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Project to vocabulary
        logits = self.output(output)
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)

        return log_probs


class DeepLSTMModel(nn.Module):
    """Deeper LSTM with residual connections"""

    def __init__(self, input_dim=1659, hidden_dim=512, num_layers=4,
                 vocab_size=100, dropout=0.4):
        super().__init__()

        self.name = "Deep_BiLSTM_Residual"

        # Feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Deep BiLSTM
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Residual connection
        self.residual_proj = nn.Linear(input_dim, hidden_dim)

        # Output layer
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
        packed_output, _ = self.bilstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Add residual connection
        output = output + residual[:, :output.size(1), :]

        # Project to vocabulary
        logits = self.output(output)
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)

        return log_probs


class GRUModel(nn.Module):
    """GRU-based model (often faster than LSTM)"""

    def __init__(self, input_dim=1659, hidden_dim=256, num_layers=3,
                 vocab_size=100, dropout=0.3):
        super().__init__()

        self.name = "BiGRU"

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
        x = self.feature_encoder(x)

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        packed_output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        logits = self.output(output)
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)

        return log_probs


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

class Trainer:
    """Training and evaluation handler"""

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = defaultdict(list)

        # CTC Loss (blank=0)
        self.criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

        print(f"🚀 Trainer initialized on {device}")
        print(f"   Model: {model.name}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    def train_epoch(self, dataloader, optimizer, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for batch_idx, (landmarks, glosses, lengths, gloss_lengths) in enumerate(dataloader):
            # Move to device
            landmarks = landmarks.to(self.device)
            lengths = lengths.to(self.device)
            gloss_lengths = gloss_lengths.to(self.device)

            # Concatenate all glosses for CTC
            target = torch.cat(glosses).to(self.device)

            # Forward pass
            log_probs = self.model(landmarks, lengths)

            # CTC loss expects: (T, N, C), targets, input_lengths, target_lengths
            loss = self.criterion(log_probs, target, lengths, gloss_lengths)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (important for RNNs!)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 5 == 0:
                print(f"   Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        self.history['train_loss'].append(avg_loss)
        return avg_loss

    def evaluate(self, dataloader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for landmarks, glosses, lengths, gloss_lengths in dataloader:
                landmarks = landmarks.to(self.device)
                lengths = lengths.to(self.device)
                gloss_lengths = gloss_lengths.to(self.device)
                target = torch.cat(glosses).to(self.device)

                log_probs = self.model(landmarks, lengths)
                loss = self.criterion(log_probs, target, lengths, gloss_lengths)
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        self.history['val_loss'].append(avg_loss)
        return avg_loss

    def decode_ctc(self, log_probs, idx_to_gloss):
        """Simple CTC decoding (greedy)"""
        # Get most likely class at each timestep
        _, predictions = torch.max(log_probs, dim=2)  # (T, N)
        predictions = predictions.transpose(0, 1)  # (N, T)

        decoded = []
        for pred_seq in predictions:
            # Remove blanks and consecutive duplicates
            prev = -1
            decoded_seq = []
            for p in pred_seq:
                p = p.item()
                if p != 0 and p != prev:  # Not blank and not duplicate
                    decoded_seq.append(idx_to_gloss.get(p, '<UNK>'))
                prev = p
            decoded.append(decoded_seq)

        return decoded


# ============================================================================
# 🎨 VISUALIZATION
# ============================================================================

def plot_training_curves(histories: Dict[str, Dict], save_path=None):
    """Plot training curves for multiple models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for model_name, history in histories.items():
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], marker='o', label=model_name)
        if 'val_loss' in history and history['val_loss']:
            ax2.plot(epochs, history['val_loss'], marker='s', label=model_name)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Plot saved to {save_path}")

    return fig


# ============================================================================
# MAIN DEMO FUNCTION
# ============================================================================

def run_demo(data_files: List[Path],
             models_to_test: List[str] = ['baseline', 'bilstm', 'gru'],
             num_epochs: int = 10,
             batch_size: int = 4,
             learning_rate: float = 0.001):
    """
    Run comparison demo of different models

    Args:
        data_files: List of NPZ file paths
        models_to_test: List of model names to test
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """

    print("=" * 70)
    print("🎯 SIGN LANGUAGE RECOGNITION - MODEL COMPARISON DEMO")
    print("=" * 70)

    # Create dataset
    dataset = PhoenixDataset(data_files)

    # For demo: use same data for train and val (you'll split properly later)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn)

    vocab_size = len(dataset.gloss_to_idx)

    # Model factory
    model_configs = {
        'baseline': lambda: BaselineModel(vocab_size=vocab_size),
        'bilstm': lambda: BiLSTMModel(vocab_size=vocab_size),
        'deep': lambda: DeepLSTMModel(vocab_size=vocab_size),
        'gru': lambda: GRUModel(vocab_size=vocab_size)
    }

    results = {}
    histories = {}

    # Train each model
    for model_name in models_to_test:
        print(f"\n{'=' * 70}")
        print(f" Training: {model_name.upper()}")
        print(f"{'=' * 70}")

        # Create model
        model = model_configs[model_name]()
        trainer = Trainer(model)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )

        # Training loop
        best_val_loss = float('inf')
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            print(f"\n📅 Epoch {epoch}/{num_epochs}")

            train_loss = trainer.train_epoch(train_loader, optimizer, epoch)
            val_loss = trainer.evaluate(val_loader)

            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}")

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"   ✅ New best model! (Val Loss: {val_loss:.4f})")

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
    print("\n" + "=" * 70)
    print("📊 RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Parameters':<15} {'Best Val Loss':<15} {'Time (s)':<10}")
    print("-" * 70)

    for name, res in results.items():
        print(f"{name:<20} {res['params']:<15,} {res['best_val_loss']:<15.4f} {res['train_time']:<10.1f}")

    # Plot comparison
    fig = plot_training_curves(histories, save_path='/home/claude/training_comparison.png')

    return results, histories, dataset


# ============================================================================
# 🚀 ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("🎬 Demo Script Ready!")
    print("\nTo run the demo, use:")
    print("  results, histories, dataset = run_demo(")
    print("      data_files=[Path('your_data.npz')],")
    print("      models_to_test=['baseline', 'bilstm', 'gru'],")
    print("      num_epochs=10")
    print("  )")