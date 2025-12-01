"""
Multi-Stream Training Script for Sign Language Recognition

Usage:
    python TrainMultiStream.py --top-k 200 --epochs 100 --fusion attention

Author: Andrei Chirila, Roman Schläpfer
Date: 2025-12-01
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR

from MultiStreamModel import MultiStreamSignLanguageTransformer, MultiStreamModelConfig
from MultiStreamDataset import MultiStreamDataset, DatasetConfig, collate_fn


# ============================================================================
# 📋 CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""

    # Data
    data_dir: str = "data/preprocessed"
    vocab_file: str = "data_analysis_comprehensive/top200_glosses.csv"
    top_k: int = 200

    # Multi-Stream
    fusion_type: str = 'attention'

    # Model
    num_landmarks: int = 543
    landmark_dim: int = 2
    num_bones: int = 70
    stream_hidden_dim: int = 256
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1

    # Training
    batch_size: int = 8
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5

    # Early stopping
    patience: int = 15

    # Checkpoints
    checkpoint_dir: str = "checkpoints"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True


# ============================================================================
# 📖 VOCABULARY
# ============================================================================

class Vocabulary:
    """Vocabulary for gloss encoding."""

    def __init__(self):
        self.gloss2idx = {'<PAD>': 0, '<BLANK>': 1, '<UNK>': 2}
        self.idx2gloss = {0: '<PAD>', 1: '<BLANK>', 2: '<UNK>'}

    @classmethod
    def from_csv(cls, path: str) -> 'Vocabulary':
        vocab = cls()
        with open(path, 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',')
                if parts:
                    gloss = parts[0].strip()
                    if gloss and gloss not in vocab.gloss2idx:
                        idx = len(vocab.gloss2idx)
                        vocab.gloss2idx[gloss] = idx
                        vocab.idx2gloss[idx] = gloss
        return vocab

    @property
    def vocab_size(self) -> int:
        return len(self.gloss2idx)


# ============================================================================
# 📊 METRICS
# ============================================================================

def compute_wer(predictions: List[List[int]], targets: List[List[int]]) -> float:
    """Compute Word Error Rate."""
    total_errors = 0
    total_length = 0

    for pred, target in zip(predictions, targets):
        pred = [p for p in pred if p > 2]
        target = [t for t in target if t > 2]

        d = np.zeros((len(pred) + 1, len(target) + 1), dtype=np.int32)
        d[:, 0] = np.arange(len(pred) + 1)
        d[0, :] = np.arange(len(target) + 1)

        for i in range(1, len(pred) + 1):
            for j in range(1, len(target) + 1):
                cost = 0 if pred[i - 1] == target[j - 1] else 1
                d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)

        total_errors += d[len(pred), len(target)]
        total_length += max(len(target), 1)

    return total_errors / total_length if total_length > 0 else 1.0


def ctc_decode(log_probs: torch.Tensor, lengths: torch.Tensor, blank_idx: int = 1) -> List[List[int]]:
    """Greedy CTC decoding."""
    predictions = []

    for i in range(log_probs.size(0)):
        seq_len = lengths[i].item()
        seq = log_probs[i, :seq_len].argmax(dim=-1).tolist()

        decoded = []
        prev = -1
        for token in seq:
            if token != blank_idx and token != prev:
                decoded.append(token)
            prev = token

        predictions.append(decoded)

    return predictions


# ============================================================================
# 🏋️ TRAINER
# ============================================================================

class Trainer:
    """Multi-stream model trainer."""

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, config: TrainingConfig, vocab: Vocabulary):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.vocab = vocab
        self.device = torch.device(config.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        self.scheduler = OneCycleLR(
            self.optimizer, max_lr=config.learning_rate,
            epochs=config.epochs, steps_per_epoch=len(train_loader),
            pct_start=config.warmup_epochs / config.epochs)

        self.ctc_loss = nn.CTCLoss(blank=1, zero_infinity=True)
        self.scaler = GradScaler() if config.use_amp else None

        self.best_wer = float('inf')
        self.patience_counter = 0

        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def train_epoch(self) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0

        for batch in self.train_loader:
            streams = {
                'joint': batch['joint'].to(self.device),
                'bone': batch['bone'].to(self.device),
                'joint_motion': batch['joint_motion'].to(self.device),
                'bone_motion': batch['bone_motion'].to(self.device),
            }
            labels = batch['labels'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            label_lengths = batch['label_lengths'].to(self.device)

            self.optimizer.zero_grad()

            if self.config.use_amp:
                with autocast():
                    log_probs, out_lens = self.model(streams, lengths)
                    loss = self.ctc_loss(log_probs.transpose(0, 1), labels, out_lens, label_lengths)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                log_probs, out_lens = self.model(streams, lengths)
                loss = self.ctc_loss(log_probs.transpose(0, 1), labels, out_lens, label_lengths)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            self.scheduler.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        all_preds, all_targets = [], []

        for batch in self.val_loader:
            streams = {
                'joint': batch['joint'].to(self.device),
                'bone': batch['bone'].to(self.device),
                'joint_motion': batch['joint_motion'].to(self.device),
                'bone_motion': batch['bone_motion'].to(self.device),
            }
            labels = batch['labels'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            label_lengths = batch['label_lengths'].to(self.device)

            log_probs, out_lens = self.model(streams, lengths)
            loss = self.ctc_loss(log_probs.transpose(0, 1), labels, out_lens, label_lengths)
            total_loss += loss.item()

            preds = ctc_decode(log_probs, out_lens)
            all_preds.extend(preds)

            for i in range(labels.size(0)):
                all_targets.append(labels[i, :label_lengths[i]].tolist())

        return total_loss / len(self.val_loader), compute_wer(all_preds, all_targets)

    def save_checkpoint(self, epoch: int, wer: float, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'wer': wer,
        }

        torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, 'latest_model.pt'))

        if is_best:
            torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, 'best_model.pt'))
            print(f"   💾 New best model! WER: {wer:.2%}")

    def train(self):
        """Full training loop."""
        print("\n" + "=" * 60)
        print("🚀 Multi-Stream Training")
        print("=" * 60)
        print(f"   Fusion: {self.config.fusion_type}")
        print(f"   Device: {self.device}")
        print("=" * 60 + "\n")

        for epoch in range(1, self.config.epochs + 1):
            start = time.time()

            train_loss = self.train_epoch()
            val_loss, val_wer = self.validate()

            is_best = val_wer < self.best_wer - 0.001
            if is_best:
                self.best_wer = val_wer
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, val_wer, is_best)

            elapsed = time.time() - start
            print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"WER: {val_wer:.2%} | Best: {self.best_wer:.2%} | {elapsed:.1f}s")

            if self.patience_counter >= self.config.patience:
                print(f"\n⚠️ Early stopping at epoch {epoch}")
                break

        print(f"\n🏁 Done! Best WER: {self.best_wer:.2%}")


# ============================================================================
# 🔧 DATA LOADING
# ============================================================================

def load_data(data_dir: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load train/val/test data."""
    train_list, val_list, test_list = [], [], []

    for split, lst in [('train', train_list), ('dev', val_list), ('test', test_list)]:
        split_dir = Path(data_dir) / split
        if not split_dir.exists():
            continue

        for npz_file in sorted(split_dir.glob("*.npz")):
            ann_file = npz_file.with_suffix('.json')
            if ann_file.exists():
                with open(ann_file) as f:
                    ann = json.load(f)
                glosses = ann.get('glosses', ann.get('annotation', []))
            else:
                glosses = []

            lst.append({'features': str(npz_file), 'glosses': glosses})

    print(f"📊 Data: {len(train_list)} train, {len(val_list)} val, {len(test_list)} test")
    return train_list, val_list, test_list


# ============================================================================
# 🚀 MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-k', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fusion', type=str, default='attention',
                        choices=['concat', 'attention', 'gated', 'weighted'])
    parser.add_argument('--data-dir', type=str, default='data/preprocessed')
    parser.add_argument('--vocab-file', type=str, default='data_analysis_comprehensive/top200_glosses.csv')
    args = parser.parse_args()

    config = TrainingConfig(
        top_k=args.top_k, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, fusion_type=args.fusion,
        data_dir=args.data_dir, vocab_file=args.vocab_file)

    # Vocab
    print(f"\n📖 Loading vocabulary: {config.vocab_file}")
    vocab = Vocabulary.from_csv(config.vocab_file)
    print(f"   Size: {vocab.vocab_size}")

    # Data
    print(f"\n📁 Loading data: {config.data_dir}")
    train_list, val_list, _ = load_data(config.data_dir)

    ds_config = DatasetConfig()
    train_ds = MultiStreamDataset(train_list, vocab, ds_config, is_train=True)
    val_ds = MultiStreamDataset(val_list, vocab, ds_config, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, collate_fn=collate_fn)

    # Model
    model_config = MultiStreamModelConfig(
        num_landmarks=config.num_landmarks, landmark_dim=config.landmark_dim,
        num_bones=train_ds.num_bones, stream_hidden_dim=config.stream_hidden_dim,
        d_model=config.d_model, n_heads=config.n_heads, n_layers=config.n_layers,
        d_ff=config.d_ff, dropout=config.dropout, num_classes=vocab.vocab_size,
        fusion_type=config.fusion_type, device=config.device)

    model = MultiStreamSignLanguageTransformer(model_config)

    # Train
    trainer = Trainer(model, train_loader, val_loader, config, vocab)
    trainer.train()


if __name__ == "__main__":
    main()