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


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Monitors validation loss/accuracy and stops training if no improvement.
    """
    def __init__(self, patience=10, min_delta=0.001, metric="loss", mode="min"):
        """
        Args:
            patience: Number of epochs with no improvement after which training stops
            min_delta: Minimum change to qualify as an improvement
            metric: "loss" or "accuracy" - which metric to monitor
            mode: "min" for loss, "max" for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False

        if mode == "min":
            self.best_score = float('inf')
        else:
            self.best_score = -float('inf')

    def __call__(self, current_score, epoch):
        """
        Check if training should stop.

        Args:
            current_score: Current validation loss or accuracy
            epoch: Current epoch number

        Returns:
            True if training should stop, False otherwise
        """
        improved = False

        if self.mode == "min":
            # For loss: lower is better
            if current_score < (self.best_score - self.min_delta):
                improved = True
                self.best_score = current_score
                self.best_epoch = epoch
                self.counter = 0
        else:
            # For accuracy: higher is better
            if current_score > (self.best_score + self.min_delta):
                improved = True
                self.best_score = current_score
                self.best_epoch = epoch
                self.counter = 0

        if not improved:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n[EARLY STOPPING] No improvement for {self.patience} epochs")
                print(f"  Best {self.metric}: {self.best_score:.4f} at epoch {self.best_epoch + 1}")
                return True
        else:
            if improved:
                print(f"  ✓ Best {self.metric} improved to {self.best_score:.4f}")

        return False


class SignLanguageDataset(Dataset):
    """
    Load preprocessed landmarks from NPZ files.
    Handles variable-length sequences without padding.
    """
    def __init__(self, npz_dir, word_to_idx=None, debug=True):
        self.npz_dir = Path(npz_dir)
        self.npz_files = sorted(self.npz_dir.glob("*.npz"))
        self.debug = debug

        if debug:
            print(f"\n[DEBUG] SignLanguageDataset.__init__")
            print(f"  NPZ directory: {self.npz_dir}")
            print(f"  Total NPZ files found: {len(self.npz_files)}")

        if word_to_idx is None:
            # Build vocabulary from all unique glosses
            self.word_to_idx = {}
            for npz_file in self.npz_files:
                try:
                    data = np.load(npz_file, allow_pickle=True)
                    gloss = data["glosses"][0]  # Single label per file
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

    def __getitem__(self, idx):
        npz_file = self.npz_files[idx]
        data = np.load(npz_file, allow_pickle=True)

        landmarks = data["landmarks"].astype(np.float32)  # Shape: (seq_len, 1659)
        gloss = data["glosses"][0]
        label = self.word_to_idx[gloss]

        # Return as tensors
        landmarks_tensor = torch.from_numpy(landmarks)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return landmarks_tensor, label_tensor


class RemappedDataset(torch.utils.data.Dataset):
    """
    Wraps a dataset and remaps labels from original indices to new indices.
    Used to convert top-N word indices to 0-(N-1) range.
    """
    def __init__(self, dataset, indices, remapping):
        self.dataset = dataset
        self.indices = indices
        self.remapping = remapping

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_dataset_idx = self.indices[idx]
        landmarks, old_label = self.dataset[actual_dataset_idx]
        old_label_value = old_label.item()
        new_label = torch.tensor(self.remapping[old_label_value], dtype=torch.long)
        return landmarks, new_label


class LSTMSignClassifier(nn.Module):
    """
    Simple LSTM-based sign language classifier.
    Handles variable-length sequences.
    """
    def __init__(self, input_size=1659, hidden_size=256, num_classes=5, num_layers=2, debug=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.debug = debug

        if debug:
            print(f"\n[DEBUG] LSTMSignClassifier.__init__")
            print(f"  Input size: {input_size}")
            print(f"  Hidden size: {hidden_size}")
            print(f"  Num layers: {num_layers}")
            print(f"  Num classes: {num_classes}")

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )

        # After LSTM: hidden_size * 2 (bidirectional)
        lstm_output_size = hidden_size * 2

        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        if debug:
            print(f"  LSTM output size: {lstm_output_size}")
            print(f"  Total parameters: {sum(p.numel() for p in self.parameters())}")

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            logits: (batch_size, num_classes)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Extract last layer's both directions and concatenate
        forward_hidden = h_n[-2, :, :]
        backward_hidden = h_n[-1, :, :]
        last_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)

        # Classification
        logits = self.classifier(last_hidden)
        return logits


class PadCollate:
    """Custom collate function to pad variable-length sequences."""
    def __init__(self, debug=False):
        self.debug = debug
        self.call_count = 0

    def __call__(self, batch):
        self.call_count += 1
        landmarks_list, labels_list = zip(*batch)
        labels = torch.stack(labels_list)

        # Get sequence lengths before padding
        seq_lengths = torch.tensor([len(lm) for lm in landmarks_list])

        # Pad sequences to max length in batch
        landmarks_padded = nn.utils.rnn.pad_sequence(
            landmarks_list, 
            batch_first=True, 
            padding_value=0.0
        )

        if self.debug and self.call_count == 1:
            print(f"\n[DEBUG] PadCollate first batch")
            print(f"  Batch size: {len(batch)}")
            print(f"  Sequence lengths: min={seq_lengths.min()}, max={seq_lengths.max()}, mean={seq_lengths.float().mean():.1f}")
            print(f"  Padded shape: {landmarks_padded.shape}")
            print(f"  Labels (should be 0-{len(set(labels.tolist()))-1}): {labels.tolist()}")

        return landmarks_padded, labels, seq_lengths


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

        # Forward pass
        logits = model(landmarks)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Debug output every N batches
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

    # Loss curve
    axes[0, 0].plot(train_losses, label='Train Loss', marker='o', linewidth=2)
    axes[0, 0].plot(val_losses, label='Val Loss', marker='s', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[0, 1].plot(train_accs, label='Train Accuracy', marker='o', linewidth=2)
    axes[0, 1].plot(val_accs, label='Val Accuracy', marker='s', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training & Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])

    # Loss difference (overfitting indicator)
    loss_diff = [v - t for t, v in zip(train_losses, val_losses)]
    axes[1, 0].bar(range(len(loss_diff)), loss_diff, alpha=0.7, color='orange')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Val Loss - Train Loss')
    axes[1, 0].set_title('Overfitting Indicator (Negative = Underfitting)')
    axes[1, 0].grid(True, alpha=0.3)

    # Accuracy gap
    acc_gap = [v - t for t, v in zip(train_accs, val_accs)]
    axes[1, 1].bar(range(len(acc_gap)), acc_gap, alpha=0.7, color='green')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Val Acc - Train Acc')
    axes[1, 1].set_title('Generalization Gap (Negative = Overfitting)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training curves saved to {plot_path}")
    plt.close()

    return plot_path


def log_summary_metrics(train_losses, val_losses, train_accs, val_accs, top_n_words, best_epoch, best_val_acc):
    """Log comprehensive summary metrics to MLflow."""

    # Overall statistics
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

    # Generalization metrics
    final_gap = val_accs[-1] - train_accs[-1]
    max_gap = max(val_accs[i] - train_accs[i] for i in range(len(train_accs)))

    summary_stats["final_accuracy_gap"] = float(final_gap)
    summary_stats["max_accuracy_gap"] = float(max_gap)
    summary_stats["loss_stability"] = float(np.std(val_losses[-5:]))  # Last 5 epochs std

    # Log all metrics
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


def main():
    """Main training pipeline with MLflow tracking and early stopping."""
    print("=" * 80)
    print("SIGN LANGUAGE TRANSLATION - TOP N WORDS CLASSIFIER (WITH EARLY STOPPING)")
    print("=" * 80)

    # ============================================================================
    # MLFLOW SETUP
    # ============================================================================
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'roman'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'SignNet'
    mlflow.set_tracking_uri("https://mlflow.schlaepfer.me")

    EXPERIMENT_NAME = "SignNetWord"
    RUN_NAME = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ============================================================================
    # HYPERPARAMETERS
    # ============================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 150
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    NPZ_DIR = "./word_landmarks_extracted"
    MODEL_SAVE_DIR = "./models"
    PLOTS_DIR = "./plots"

    # Early stopping configuration
    EARLY_STOPPING_PATIENCE = 15  # Stop if no improvement for 15 epochs
    EARLY_STOPPING_MIN_DELTA = 0.001  # Minimum improvement threshold
    EARLY_STOPPING_METRIC = "loss"  # Monitor "loss" or "accuracy"
    EARLY_STOPPING_MODE = "min"  # "min" for loss, "max" for accuracy

    print(f"\n[CONFIG] Device: {DEVICE}")
    print(f"[CONFIG] Batch size: {BATCH_SIZE}")
    print(f"[CONFIG] Learning rate: {LEARNING_RATE}")
    print(f"[CONFIG] Max epochs: {NUM_EPOCHS}")
    print(f"[CONFIG] Hidden size: {HIDDEN_SIZE}")
    print(f"[CONFIG] Num LSTM layers: {NUM_LAYERS}")
    print(f"[CONFIG] Early Stopping Patience: {EARLY_STOPPING_PATIENCE} epochs")
    print(f"[CONFIG] Early Stopping Metric: {EARLY_STOPPING_METRIC}")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    with mlflow.start_run(log_system_metrics=True, run_name=RUN_NAME) as run:
        # ========================================================================
        # Log System Information
        # ========================================================================
        mlflow.log_param("python_version", platform.python_version())
        mlflow.log_param("pytorch_version", torch.__version__)
        mlflow.log_param("os", platform.system())
        mlflow.log_param("cpu_count", os.cpu_count())
        mlflow.log_param("total_ram_gb", round(psutil.virtual_memory().total / (1024**3), 2))

        if torch.cuda.is_available():
            mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
            mlflow.log_param("cuda_version", torch.version.cuda)
            mlflow.log_param("gpu_memory_gb", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))

        # ========================================================================
        # Log Hyperparameters
        # ========================================================================
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "input_dim": 1659,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
            "loss_function": "CrossEntropyLoss",
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_metric": EARLY_STOPPING_METRIC,
            "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
            "device": str(DEVICE),
        })

        # ========================================================================
        # STEP 1: Load dataset
        # ========================================================================
        print(f"\n[STEP 1] Loading dataset...")
        dataset = SignLanguageDataset(NPZ_DIR, debug=True)

        # ========================================================================
        # STEP 2: Analyze word frequencies
        # ========================================================================
        print(f"\n[STEP 2] Analyzing word frequencies...")
        word_counts = Counter()
        for i in range(len(dataset)):
            _, label = dataset[i]
            word = dataset.idx_to_word[label.item()]
            word_counts[word] += 1

        top_n_words = [word for word, _ in word_counts.most_common(10)]
        print(f"  Top 10 words: {top_n_words}")
        for idx, (word, count) in enumerate(word_counts.most_common(10)):
            print(f"    {idx+1:2}. {word:20} : {count:4} samples")

        # ========================================================================
        # STEP 3: Filter to top N words
        # ========================================================================
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

        print(f"  Filtered dataset size: {len(filtered_indices)} samples")

        # ========================================================================
        # STEP 4: Stratified split
        # ========================================================================
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

        print(f"  Train samples: {len(train_indices)}")
        print(f"  Val samples: {len(val_indices)}")

        train_subset = RemappedDataset(dataset, train_indices, old_to_new_idx)
        val_subset = RemappedDataset(dataset, val_indices, old_to_new_idx)

        # Check class distribution
        num_classes = len(top_n_words)
        train_label_counts = [0] * num_classes
        for idx in range(len(train_subset)):
            _, label = train_subset[idx]
            train_label_counts[label.item()] += 1

        print(f"\n  Training set distribution:")
        for new_idx, word in enumerate(top_n_words):
            count = train_label_counts[new_idx]
            percentage = 100 * count / len(train_subset) if len(train_subset) > 0 else 0
            print(f"    Label {new_idx:2}: {word:20} : {count:4} samples ({percentage:5.1f}%)")
            mlflow.log_param(f"train_class_{new_idx}_count", count)

        # ========================================================================
        # STEP 5: Create data loaders
        # ========================================================================
        print(f"\n[STEP 5] Creating data loaders...")
        train_loader = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=PadCollate(debug=True)
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=PadCollate(debug=False)
        )

        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")

        mlflow.log_params({
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
            "total_train_samples": len(train_subset),
            "total_val_samples": len(val_subset),
            "num_classes": num_classes,
        })

        # ========================================================================
        # STEP 6: Build model
        # ========================================================================
        print(f"\n[STEP 6] Building model...")
        model = LSTMSignClassifier(
            input_size=1659,
            hidden_size=HIDDEN_SIZE,
            num_classes=num_classes,
            num_layers=NUM_LAYERS,
            debug=True
        ).to(DEVICE)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_params({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        })

        # ========================================================================
        # STEP 7: Setup training
        # ========================================================================
        print(f"\n[STEP 7] Setting up training...")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Initialize early stopping
        if EARLY_STOPPING_METRIC == "loss":
            early_stopping = EarlyStopping(
                patience=EARLY_STOPPING_PATIENCE,
                min_delta=EARLY_STOPPING_MIN_DELTA,
                metric="validation loss",
                mode="min"
            )
        else:
            early_stopping = EarlyStopping(
                patience=EARLY_STOPPING_PATIENCE,
                min_delta=EARLY_STOPPING_MIN_DELTA,
                metric="validation accuracy",
                mode="max"
            )

        # ========================================================================
        # STEP 8: Training loop
        # ========================================================================
        print(f"\n[STEP 8] Starting training with early stopping...")
        print("=" * 80)

        best_val_acc = 0
        best_epoch = 0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, DEVICE, epoch, debug=True
            )
            val_loss, val_acc = evaluate(
                model, val_loader, criterion, DEVICE, epoch, debug=True
            )

            scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_model_path = os.path.join(MODEL_SAVE_DIR, "sign_classifier_best.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"  💾 Saved best model")

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3}/{NUM_EPOCHS} │ "
                  f"Train Loss: {train_loss:.4f} │ Train Acc: {train_acc:.2%} │ "
                  f"Val Loss: {val_loss:.4f} │ Val Acc: {val_acc:.2%} │ "
                  f"LR: {lr:.2e}")

            # Log metrics every epoch
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "learning_rate": lr,
            }, step=epoch)

            # Check early stopping
            if EARLY_STOPPING_METRIC == "loss":
                should_stop = early_stopping(val_loss, epoch)
            else:
                should_stop = early_stopping(val_acc, epoch)

            if should_stop:
                print(f"\n[STOPPED] Training stopped at epoch {epoch+1}")
                break

        # ========================================================================
        # STEP 9: Save results and generate plots
        # ========================================================================
        print("\n" + "=" * 80)
        print(f"[TRAINING COMPLETE]")
        print(f"  Total epochs trained: {len(train_losses)}")
        print(f"  Best Val Accuracy: {best_val_acc:.2%} at epoch {best_epoch+1}")
        print(f"  Final Train Loss: {train_losses[-1]:.4f}")
        print(f"  Final Val Loss: {val_losses[-1]:.4f}")
        print(f"  Final Train Acc: {train_accs[-1]:.2%}")
        print(f"  Final Val Acc: {val_accs[-1]:.2%}")
        print(f"\n  Classes ({num_classes}): {top_n_words}")

        # Generate and save plots
        print(f"\n[PLOTTING] Generating training curves...")
        plot_path = plot_training_curves(train_losses, val_losses, train_accs, val_accs, PLOTS_DIR)

        # Log summary metrics
        log_summary_metrics(train_losses, val_losses, train_accs, val_accs, top_n_words, best_epoch, best_val_acc)

        # Save models and metrics
        final_model_path = os.path.join(MODEL_SAVE_DIR, "sign_classifier_final.pth")
        torch.save(model.state_dict(), final_model_path)

        metrics_path = os.path.join(MODEL_SAVE_DIR, "training_metrics.npz")
        np.savez(
            metrics_path,
            train_losses=np.array(train_losses),
            val_losses=np.array(val_losses),
            train_accs=np.array(train_accs),
            val_accs=np.array(val_accs)
        )

        class_info = {
            "classes": top_n_words,
            "num_classes": num_classes,
            "best_val_acc": float(best_val_acc),
            "best_epoch": int(best_epoch),
            "total_epochs_trained": len(train_losses),
            "label_remapping": {str(k): v for k, v in old_to_new_idx.items()}
        }

        class_info_path = os.path.join(MODEL_SAVE_DIR, "class_info.json")
        with open(class_info_path, 'w') as f:
            json.dump(class_info, f, indent=2)

        # ========================================================================
        # Log artifacts to MLflow
        # ========================================================================
        print(f"\n[MLFLOW] Logging artifacts...")
        mlflow.log_artifact(best_model_path)
        mlflow.log_artifact(final_model_path)
        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(class_info_path)
        mlflow.log_artifact(plot_path)

        # Log the model itself
        mlflow.pytorch.log_model(model, "model")

        # Set tags
        mlflow.set_tags({
            "model_type": "LSTM",
            "task": "sign_language_word_classification",
            "num_classes": num_classes,
            "dataset_type": "MediaPipe_Landmarks",
            "split_strategy": "Stratified",
            "early_stopping": "enabled",
            "status": "completed",
            "classes": ",".join(top_n_words),
        })

        mlflow.log_metric("epochs_trained", len(train_losses))
        mlflow.log_metric("early_stopping_patience_used", early_stopping.counter)

        print(f"\n  Best model: {best_model_path}")
        print(f"  Final model: {final_model_path}")
        print(f"  Metrics: {metrics_path}")
        print(f"  Class info: {class_info_path}")
        print(f"  Plots: {plot_path}")
        print(f"\n  MLflow Run ID: {run.info.run_id}")
        print(f"  MLflow Experiment: {EXPERIMENT_NAME}")

        print("=" * 80)


if __name__ == "__main__":
    main()
