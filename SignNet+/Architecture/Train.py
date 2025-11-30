# Train.py
"""
SignNet Training Script with MLflow Integration
"""

import os
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

# MLflow
import mlflow
import mlflow.pytorch

# Metrics & Visualization
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
sys.path.append(str(Path(__file__).parent))
from Config import Config, get_config
from Dataset import create_dataloaders, Vocabulary
from Model import SignLanguageTransformer

warnings.filterwarnings('ignore')


# ============================================
# FOCAL LOSS FOR CLASS IMBALANCE
# ============================================

class FocalCTCLoss(nn.Module):
    """
    CTC Loss with Focal Loss weighting for class imbalance.
    """

    def __init__(
            self,
            blank: int = 1,
            alpha: float = 0.25,
            gamma: float = 2.0,
            reduction: str = 'mean'
    ):
        super().__init__()
        self.blank = blank
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=True)

    def forward(
            self,
            log_probs: torch.Tensor,
            targets: torch.Tensor,
            input_lengths: torch.Tensor,
            target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            log_probs: [T, B, C] log probabilities (CTC format)
            targets: [B, S] target sequences
            input_lengths: [B] input lengths
            target_lengths: [B] target lengths
        """
        # Flatten targets for CTC
        targets_flat = []
        for i in range(targets.size(0)):
            targets_flat.extend(targets[i, :target_lengths[i]].tolist())
        targets_flat = torch.tensor(targets_flat, dtype=torch.long, device=targets.device)

        # Standard CTC loss
        loss = self.ctc_loss(log_probs, targets_flat, input_lengths, target_lengths)

        # Apply focal weighting
        pt = torch.exp(-loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============================================
# METRICS CALCULATION
# ============================================

def levenshtein_distance(s1: List[int], s2: List[int]) -> int:
    """Calculate Levenshtein (edit) distance between two sequences."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_wer(predictions: List[List[int]], targets: List[List[int]]) -> float:
    """Calculate Word Error Rate (WER)."""
    total_distance = 0
    total_length = 0

    for pred, target in zip(predictions, targets):
        total_distance += levenshtein_distance(pred, target)
        total_length += len(target)

    return total_distance / max(total_length, 1)


def decode_ctc_greedy(log_probs: torch.Tensor, blank_idx: int = 1) -> List[List[int]]:
    """
    Greedy CTC decoding.

    Args:
        log_probs: [B, T, C] log probabilities
        blank_idx: blank token index

    Returns:
        List of decoded sequences
    """
    predictions = log_probs.argmax(dim=-1)  # [B, T]

    decoded = []
    for seq in predictions:
        decoded_seq = []
        prev_token = -1

        for token in seq.tolist():
            if token != prev_token and token != blank_idx:
                decoded_seq.append(token)
            prev_token = token

        # Remove PAD tokens (idx=0)
        decoded_seq = [t for t in decoded_seq if t != 0]
        decoded.append(decoded_seq)

    return decoded


# ============================================
# CONFUSION MATRIX
# ============================================

def create_confusion_matrix(
        all_predictions: List[int],
        all_targets: List[int],
        vocabulary: Vocabulary,
        top_k: int = 20,
        save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create confusion matrix for top-k most frequent classes.

    Args:
        all_predictions: Flat list of all predicted token indices
        all_targets: Flat list of all target token indices
        vocabulary: Vocabulary object
        top_k: Number of top classes to show
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    # Get unique classes (excluding special tokens 0, 1, 2)
    valid_classes = sorted(set(all_targets) - {0, 1, 2})[:top_k]

    if len(valid_classes) == 0:
        print("⚠️  No valid classes for confusion matrix")
        return None

    # Filter to only valid classes
    filtered_preds = []
    filtered_targets = []

    for p, t in zip(all_predictions, all_targets):
        if t in valid_classes:
            filtered_preds.append(p if p in valid_classes else -1)
            filtered_targets.append(t)

    if len(filtered_targets) == 0:
        print("⚠️  No samples for confusion matrix")
        return None

    # Create confusion matrix
    cm = confusion_matrix(filtered_targets, filtered_preds, labels=valid_classes)

    # Normalize
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Get class names
    class_names = []
    for idx in valid_classes:
        name = vocabulary.idx2gloss.get(idx, f"idx_{idx}")
        class_names.append(name[:10])  # Truncate long names

    # Plot
    sns.heatmap(
        cm_normalized,
        annot=False,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Confusion Matrix (Top-{len(valid_classes)} Classes)', fontsize=14)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# ============================================
# TRAINING FUNCTIONS
# ============================================

def train_epoch(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        scaler: GradScaler,
        device: str,
        epoch: int,
        config: Config
) -> Dict[str, float]:
    """Train for one epoch."""

    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Train]", leave=False)

    for batch in pbar:
        # Move to device
        landmarks = batch['landmarks'].to(device)
        glosses = batch['glosses'].to(device)
        landmarks_lengths = batch['landmarks_lengths'].to(device)
        glosses_lengths = batch['glosses_lengths'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast(enabled=config.training.use_mixed_precision):
            log_probs, output_lengths = model(landmarks, landmarks_lengths)

            # Prepare for CTC: [B, T, C] -> [T, B, C]
            log_probs_ctc = log_probs.transpose(0, 1)

            # Calculate loss
            loss = criterion(
                log_probs_ctc,
                glosses,
                output_lengths,
                glosses_lengths
            )

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Update scheduler (per-step for warmup)
        if scheduler is not None:
            scheduler.step()

        # Track metrics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

    return {
        'loss': total_loss / max(num_batches, 1),
        'lr': optimizer.param_groups[0]['lr']
    }


def validate(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        vocabulary: Vocabulary,
        device: str,
        epoch: int,
        config: Config,
        collect_predictions: bool = False
) -> Dict:
    """Validate model."""

    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_predictions = []
    all_targets = []

    # For WER calculation
    all_pred_sequences = []
    all_target_sequences = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Val]", leave=False)

    with torch.no_grad():
        for batch in pbar:
            # Move to device
            landmarks = batch['landmarks'].to(device)
            glosses = batch['glosses'].to(device)
            landmarks_lengths = batch['landmarks_lengths'].to(device)
            glosses_lengths = batch['glosses_lengths'].to(device)

            # Forward pass
            with autocast(enabled=config.training.use_mixed_precision):
                log_probs, output_lengths = model(landmarks, landmarks_lengths)

                # Prepare for CTC
                log_probs_ctc = log_probs.transpose(0, 1)

                # Calculate loss
                loss = criterion(
                    log_probs_ctc,
                    glosses,
                    output_lengths,
                    glosses_lengths
                )

            total_loss += loss.item()
            num_batches += 1

            # Decode predictions
            decoded_preds = decode_ctc_greedy(log_probs, blank_idx=vocabulary.blank_idx)

            # Get target sequences
            for i in range(glosses.size(0)):
                target_seq = glosses[i, :glosses_lengths[i]].tolist()
                # Remove special tokens
                target_seq = [t for t in target_seq if t not in [0, 1, 2]]

                all_pred_sequences.append(decoded_preds[i])
                all_target_sequences.append(target_seq)

                # For confusion matrix
                if collect_predictions:
                    all_predictions.extend(decoded_preds[i])
                    all_targets.extend(target_seq)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Calculate WER
    wer = calculate_wer(all_pred_sequences, all_target_sequences)

    results = {
        'loss': total_loss / max(num_batches, 1),
        'wer': wer,
    }

    if collect_predictions:
        results['predictions'] = all_predictions
        results['targets'] = all_targets

    return results


# ============================================
# MLFLOW SETUP
# ============================================

def setup_mlflow(config: Config) -> str:
    """Setup MLflow tracking."""

    print("\n📊 Setting up MLflow...")

    try:
        # Set tracking URI
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)

        # Set authentication
        os.environ['MLFLOW_TRACKING_USERNAME'] = config.mlflow.username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = config.mlflow.password

        # Set/create experiment
        mlflow.set_experiment(config.mlflow.experiment_name)

        print(f"   ✅ MLflow tracking: {config.mlflow.tracking_uri}")
        print(f"   ✅ Experiment: {config.mlflow.experiment_name}")

        return "mlflow"

    except Exception as e:
        print(f"   ⚠️  MLflow setup failed: {e}")
        print(f"   📁 Falling back to local logging")
        return "local"


def log_to_mlflow(
        metrics: Dict[str, float],
        step: int,
        prefix: str = ""
):
    """Log metrics to MLflow."""
    for key, value in metrics.items():
        mlflow.log_metric(f"{prefix}{key}", value, step=step)


# ============================================
# MAIN TRAINING LOOP
# ============================================

def train(config: Config):
    """Main training function."""

    print("\n" + "=" * 80)
    print("SIGNNET TRAINING")
    print("=" * 80)

    # Setup device
    device = config.model.device
    print(f"\n🖥️  Device: {device}")

    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # Create dataloaders
    print("\n📚 Loading data...")
    train_loader, dev_loader, test_loader, vocabulary = create_dataloaders(config)

    # Update config with vocabulary size
    config.model.num_classes = vocabulary.vocab_size

    print(f"\n   Vocabulary size: {vocabulary.vocab_size}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Dev batches: {len(dev_loader)}")

    # Create model
    print("\n🧠 Creating model...")
    model = SignLanguageTransformer(config.model)
    model = model.to(device)

    # Loss function
    criterion = FocalCTCLoss(
        blank=vocabulary.blank_idx,
        alpha=config.training.focal_alpha,
        gamma=config.training.focal_gamma
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

    # Learning rate scheduler (Warmup + Cosine)
    total_steps = len(train_loader) * config.training.num_epochs
    warmup_steps = int(total_steps * config.training.warmup_ratio)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=config.training.min_lr
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    print(f"\n📈 Training schedule:")
    print(f"   Total steps: {total_steps}")
    print(f"   Warmup steps: {warmup_steps}")
    print(f"   Initial LR: {config.training.learning_rate}")

    # Mixed precision scaler
    scaler = GradScaler(enabled=config.training.use_mixed_precision)

    # Setup MLflow
    tracking_mode = setup_mlflow(config)

    # Checkpoint directory
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training state
    best_wer = float('inf')
    best_epoch = 0
    patience_counter = 0

    # Start MLflow run
    run_name = f"signnet_top{config.data.use_top_k_classes}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) if tracking_mode == "mlflow" else nullcontext():

        # Log parameters
        if tracking_mode == "mlflow":
            mlflow.log_params({
                "top_k_classes": config.data.use_top_k_classes,
                "vocab_size": vocabulary.vocab_size,
                "batch_size": config.training.batch_size,
                "learning_rate": config.training.learning_rate,
                "num_epochs": config.training.num_epochs,
                "d_model": config.model.d_model,
                "n_layers": config.model.n_layers,
                "n_heads": config.model.n_heads,
                "focal_alpha": config.training.focal_alpha,
                "focal_gamma": config.training.focal_gamma,
                "use_augmentation": config.data.use_augmentation,
            })

        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)

        for epoch in range(config.training.num_epochs):
            epoch_start = time.time()

            # Training
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, scheduler,
                scaler, device, epoch, config
            )

            # Validation
            collect_cm = (epoch + 1) % 5 == 0 or epoch == 0  # Every 5 epochs
            val_metrics = validate(
                model, dev_loader, criterion, vocabulary,
                device, epoch, config, collect_predictions=collect_cm
            )

            epoch_time = time.time() - epoch_start

            # Print epoch summary
            print(f"\n📊 Epoch {epoch + 1}/{config.training.num_epochs} ({epoch_time:.1f}s)")
            print(f"   Train Loss: {train_metrics['loss']:.4f}")
            print(f"   Val Loss:   {val_metrics['loss']:.4f}")
            print(f"   Val WER:    {val_metrics['wer']:.4f} ({val_metrics['wer'] * 100:.1f}%)")
            print(f"   LR:         {train_metrics['lr']:.2e}")

            # Log to MLflow
            if tracking_mode == "mlflow":
                log_to_mlflow(train_metrics, epoch, prefix="train_")
                log_to_mlflow({
                    'loss': val_metrics['loss'],
                    'wer': val_metrics['wer']
                }, epoch, prefix="val_")

                # Log confusion matrix
                if collect_cm and 'predictions' in val_metrics:
                    cm_path = checkpoint_dir / f"confusion_matrix_epoch{epoch + 1}.png"
                    fig = create_confusion_matrix(
                        val_metrics['predictions'],
                        val_metrics['targets'],
                        vocabulary,
                        top_k=20,
                        save_path=str(cm_path)
                    )
                    if fig:
                        mlflow.log_artifact(str(cm_path))
                        plt.close(fig)

            # Check for improvement
            if val_metrics['wer'] < best_wer:
                best_wer = val_metrics['wer']
                best_epoch = epoch + 1
                patience_counter = 0

                # Save best model
                best_path = checkpoint_dir / "best_model.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'wer': best_wer,
                    'top_k': config.data.use_top_k_classes,
                    'vocab_size': vocabulary.vocab_size,
                }, best_path)

                print(f"   ✅ New best model saved! (WER: {best_wer:.4f})")

                if tracking_mode == "mlflow":
                    mlflow.log_artifact(str(best_path))
            else:
                patience_counter += 1
                print(f"   ⏳ No improvement ({patience_counter}/{config.training.patience})")

            # Early stopping
            if patience_counter >= config.training.patience:
                print(f"\n🛑 Early stopping at epoch {epoch + 1}")
                break

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                ckpt_path = checkpoint_dir / f"checkpoint_epoch{epoch + 1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'wer': val_metrics['wer'],
                }, ckpt_path)

        # Final evaluation on test set
        print("\n" + "=" * 80)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 80)

        # Load best model
        best_checkpoint = torch.load(checkpoint_dir / "best_model.pt", weights_only=False)
        model.load_state_dict(best_checkpoint['model_state_dict'])

        test_metrics = validate(
            model, test_loader, criterion, vocabulary,
            device, config.training.num_epochs, config, collect_predictions=True
        )

        print(f"\n📊 Test Results:")
        print(f"   Test Loss: {test_metrics['loss']:.4f}")
        print(f"   Test WER:  {test_metrics['wer']:.4f} ({test_metrics['wer'] * 100:.1f}%)")

        # Final confusion matrix
        if 'predictions' in test_metrics:
            cm_path = checkpoint_dir / "confusion_matrix_test.png"
            fig = create_confusion_matrix(
                test_metrics['predictions'],
                test_metrics['targets'],
                vocabulary,
                top_k=30,
                save_path=str(cm_path)
            )
            if fig:
                if tracking_mode == "mlflow":
                    mlflow.log_artifact(str(cm_path))
                plt.close(fig)

        # Log final metrics
        if tracking_mode == "mlflow":
            mlflow.log_metrics({
                "test_loss": test_metrics['loss'],
                "test_wer": test_metrics['wer'],
                "best_val_wer": best_wer,
                "best_epoch": best_epoch,
            })

            # Log model
            mlflow.pytorch.log_model(model, "model")

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"   Best Epoch: {best_epoch}")
        print(f"   Best Val WER: {best_wer:.4f} ({best_wer * 100:.1f}%)")
        print(f"   Test WER: {test_metrics['wer']:.4f} ({test_metrics['wer'] * 100:.1f}%)")
        print(f"   Checkpoint: {checkpoint_dir / 'best_model.pt'}")
        if tracking_mode == "mlflow":
            print(f"   MLflow Run: {mlflow.active_run().info.run_id}")
        print("=" * 80)


# Context manager fallback
from contextlib import nullcontext


# ============================================
# ENTRY POINT
# ============================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Train SignNet')
    parser.add_argument('--top-k', type=int, default=50, help='Top-K classes (50, 200, or None)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--no-augment', action='store_true', help='Disable augmentation')

    args = parser.parse_args()

    # Get config
    config = get_config(
        top_k=args.top_k,
        use_augmentation=not args.no_augment
    )

    # Override with CLI args
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr

    # Train
    train(config)


if __name__ == '__main__':
    main()