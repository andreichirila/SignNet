# Train.py
"""
SignNet Training Script with MLflow Integration and Sample Count Analysis
"""

import os
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
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
from Dataset import create_dataloaders, Vocabulary, SignLanguageDataset
from Model import SignLanguageTransformer

warnings.filterwarnings('ignore')


# ============================================
# FOCAL LOSS FOR CLASS IMBALANCE
# ============================================

# In Train.py - Ersetze die FocalCTCLoss Klasse

class FocalCTCLoss(nn.Module):
    """
    CTC Loss with Focal Loss weighting and optional Label Smoothing.
    """

    def __init__(
            self,
            blank: int = 1,
            alpha: float = 0.25,
            gamma: float = 2.0,
            label_smoothing: float = 0.0,  # NEU!
            reduction: str = 'mean'
    ):
        super().__init__()
        self.blank = blank
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing  # NEU!
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

        # Apply label smoothing (als zusätzliche Regularisierung)
        if self.label_smoothing > 0:
            # Berechne KL-Divergenz zur Uniform-Verteilung
            num_classes = log_probs.size(-1)
            uniform = torch.full_like(log_probs, 1.0 / num_classes)
            kl_loss = F.kl_div(log_probs, uniform, reduction='batchmean')
            focal_loss = (1 - self.label_smoothing) * focal_loss + self.label_smoothing * kl_loss

        if self.reduction == 'mean':
            return focal_loss.mean() if focal_loss.dim() > 0 else focal_loss
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
# SAMPLE COUNT ANALYSIS
# ============================================

class SampleCountAnalyzer:
    """Analysiert die Prediction-Performance nach Sample Count."""

    def __init__(self, output_dir: str = "./results/sample_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Bin-Definitionen
        self.coarse_bins = {
            'low (1-10)': (1, 10),
            'mid (11-50)': (11, 50),
            'high (51-200)': (51, 200),
            'very_high (200+)': (201, float('inf'))
        }

        self.fine_bins = {
            '1-5': (1, 5),
            '6-10': (6, 10),
            '11-20': (11, 20),
            '21-50': (21, 50),
            '51-100': (51, 100),
            '101-200': (101, 200),
            '200+': (201, float('inf'))
        }

    def _get_bin(self, count: int, bins_dict: dict) -> str:
        """Ordnet einen Count einer Bin-Gruppe zu."""
        for bin_name, (low, high) in bins_dict.items():
            if low <= count <= high:
                return bin_name
        return 'unknown'

    def count_training_samples(self, train_dataset: SignLanguageDataset, vocabulary: Vocabulary) -> Dict[str, int]:
        """
        Zählt die Trainingssamples pro Gloss-Klasse.

        Args:
            train_dataset: Training Dataset
            vocabulary: Vocabulary object

        Returns:
            Dict {gloss_name: count}
        """
        gloss_counter = Counter()

        for sample in train_dataset.samples:
            for gloss in sample['glosses']:
                if gloss in vocabulary.gloss2idx:
                    gloss_counter[gloss] += 1

        return dict(gloss_counter)

    def collect_predictions(
            self,
            model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            vocabulary: Vocabulary,
            device: str,
            use_mixed_precision: bool = True
    ) -> Tuple[List[List[int]], List[List[int]], List[str]]:
        """
        Sammelt alle Predictions und Targets vom DataLoader.

        Returns:
            pred_sequences: Liste von Predicted Sequenzen (als Indizes)
            target_sequences: Liste von Target Sequenzen (als Indizes)
            file_names: Liste der Dateinamen
        """
        model.eval()

        all_pred_sequences = []
        all_target_sequences = []
        all_files = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collecting predictions", leave=False):
                landmarks = batch['landmarks'].to(device)
                glosses = batch['glosses']
                landmarks_lengths = batch['landmarks_lengths'].to(device)
                glosses_lengths = batch['glosses_lengths']
                files = batch['files']

                with autocast(enabled=use_mixed_precision):
                    log_probs, _ = model(landmarks, landmarks_lengths)

                # Decode predictions
                decoded_preds = decode_ctc_greedy(log_probs, blank_idx=vocabulary.blank_idx)

                # Get target sequences
                for i in range(glosses.size(0)):
                    target_seq = glosses[i, :glosses_lengths[i]].tolist()
                    # Remove special tokens (PAD=0, BLANK=1, UNK=2)
                    target_seq = [t for t in target_seq if t not in [0, 1, 2]]
                    pred_seq = [p for p in decoded_preds[i] if p not in [0, 1, 2]]

                    all_pred_sequences.append(pred_seq)
                    all_target_sequences.append(target_seq)
                    all_files.append(files[i])

        return all_pred_sequences, all_target_sequences, all_files

    def analyze(
            self,
            pred_sequences: List[List[int]],
            target_sequences: List[List[int]],
            class_counts: Dict[str, int],
            vocabulary: Vocabulary
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Hauptanalyse-Funktion.

        Args:
            pred_sequences: Liste von Predicted Sequenzen
            target_sequences: Liste von Target Sequenzen
            class_counts: Dict {gloss_name: training_sample_count}
            vocabulary: Vocabulary object

        Returns:
            class_df: DataFrame mit Per-Class Metriken
            group_df: DataFrame mit Per-Group Metriken
        """
        # Flatten sequences und berechne per-token Metriken
        class_metrics = defaultdict(lambda: {'correct': 0, 'total': 0, 'predicted': 0})

        for preds, targets in zip(pred_sequences, target_sequences):
            # Count targets
            for t in targets:
                gloss_name = vocabulary.idx2gloss.get(t, '<UNK>')
                class_metrics[gloss_name]['total'] += 1

            # Count correct predictions (simple token matching)
            pred_counter = Counter(preds)
            target_counter = Counter(targets)

            for token, count in target_counter.items():
                gloss_name = vocabulary.idx2gloss.get(token, '<UNK>')
                pred_count = pred_counter.get(token, 0)
                class_metrics[gloss_name]['correct'] += min(count, pred_count)

            # Count all predictions
            for p in preds:
                gloss_name = vocabulary.idx2gloss.get(p, '<UNK>')
                class_metrics[gloss_name]['predicted'] += 1

        # Erstelle DataFrame
        rows = []
        for gloss_name, metrics in class_metrics.items():
            if gloss_name in ['<PAD>', '<BLANK>', '<UNK>']:
                continue

            sample_count = class_counts.get(gloss_name, 0)
            total = metrics['total']
            correct = metrics['correct']
            predicted = metrics['predicted']

            # Recall (wie oft wurde es korrekt erkannt wenn es vorkam)
            recall = correct / total if total > 0 else 0
            # Precision (wie oft war die Prediction korrekt)
            precision = correct / predicted if predicted > 0 else 0
            # F1
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            rows.append({
                'gloss': gloss_name,
                'sample_count': sample_count,
                'total_in_test': total,
                'correct': correct,
                'predicted': predicted,
                'recall': recall,
                'precision': precision,
                'f1': f1,
                'coarse_bin': self._get_bin(sample_count, self.coarse_bins),
                'fine_bin': self._get_bin(sample_count, self.fine_bins)
            })

        class_df = pd.DataFrame(rows)

        # Gruppierte Statistiken
        group_rows = []
        for bin_name in ['1-5', '6-10', '11-20', '21-50', '51-100', '101-200', '200+']:
            group_data = class_df[class_df['fine_bin'] == bin_name]
            if len(group_data) == 0:
                continue

            n_classes = len(group_data)
            total_samples = group_data['total_in_test'].sum()
            total_correct = group_data['correct'].sum()
            overall_recall = total_correct / total_samples if total_samples > 0 else 0
            mean_recall = group_data['recall'].mean()
            std_recall = group_data['recall'].std() if len(group_data) > 1 else 0
            mean_f1 = group_data['f1'].mean()

            group_rows.append({
                'bin': bin_name,
                'n_classes': n_classes,
                'n_test_samples': int(total_samples),
                'n_correct': int(total_correct),
                'overall_recall': overall_recall,
                'mean_recall': mean_recall,
                'std_recall': std_recall,
                'mean_f1': mean_f1
            })

        group_df = pd.DataFrame(group_rows)

        return class_df, group_df

    def print_report(self, class_df: pd.DataFrame, group_df: pd.DataFrame):
        """Druckt einen formatierten Report."""

        print("\n" + "=" * 80)
        print("SAMPLE COUNT VS. PREDICTION PERFORMANCE ANALYSIS")
        print("=" * 80)

        # Gruppierte Statistiken
        print("\n" + "-" * 80)
        print("PERFORMANCE BY SAMPLE COUNT GROUP")
        print("-" * 80)
        print(f"\n{'Bin':<12} {'Classes':>8} {'Test Samples':>12} {'Correct':>10} {'Recall':>10} {'Mean F1':>10}")
        print("-" * 70)

        for _, row in group_df.iterrows():
            print(f"{row['bin']:<12} {row['n_classes']:>8} {row['n_test_samples']:>12} "
                  f"{row['n_correct']:>10} {row['overall_recall']:>9.1%} {row['mean_f1']:>9.1%}")

        # Korrelationsanalyse
        print("\n" + "-" * 80)
        print("CORRELATION ANALYSIS")
        print("-" * 80)

        # Filtere Klassen mit genug Test-Samples
        valid_classes = class_df[class_df['total_in_test'] >= 5]

        if len(valid_classes) > 5:
            pearson_corr = valid_classes['sample_count'].corr(valid_classes['recall'])
            spearman_corr = valid_classes['sample_count'].corr(valid_classes['recall'], method='spearman')

            print(f"\nPearson Correlation (Sample Count vs Recall):  {pearson_corr:.4f}")
            print(f"Spearman Correlation (Sample Count vs Recall): {spearman_corr:.4f}")

            if pearson_corr > 0.3:
                print("→ Positive Korrelation: Mehr Training Samples = Bessere Erkennung")
            elif pearson_corr < -0.1:
                print("→ Negative/Keine Korrelation: Unerwartetes Ergebnis")
            else:
                print("→ Schwache Korrelation")

        # Worst/Best Classes
        print("\n" + "-" * 80)
        print("WORST PERFORMING CLASSES (Bottom 10)")
        print("-" * 80)

        worst = class_df[class_df['total_in_test'] >= 3].nsmallest(10, 'recall')
        print(f"\n{'Gloss':<20} {'Train Samples':>14} {'Test':>6} {'Recall':>8}")
        print("-" * 55)
        for _, row in worst.iterrows():
            print(f"{row['gloss']:<20} {row['sample_count']:>14} {row['total_in_test']:>6} {row['recall']:>7.1%}")

        print("\n" + "-" * 80)
        print("BEST PERFORMING CLASSES (Top 10)")
        print("-" * 80)

        best = class_df[class_df['total_in_test'] >= 3].nlargest(10, 'recall')
        print(f"\n{'Gloss':<20} {'Train Samples':>14} {'Test':>6} {'Recall':>8}")
        print("-" * 55)
        for _, row in best.iterrows():
            print(f"{row['gloss']:<20} {row['sample_count']:>14} {row['total_in_test']:>6} {row['recall']:>7.1%}")

        # Empfehlungen
        print("\n" + "-" * 80)
        print("EMPFEHLUNGEN")
        print("-" * 80)

        for _, row in group_df.iterrows():
            bin_name = row['bin']
            recall = row['overall_recall']

            if recall < 0.15:
                status = "❌ ZU NIEDRIG - nicht praktikabel"
            elif recall < 0.30:
                status = "⚠️  GRENZWERTIG - unzuverlässig"
            elif recall < 0.50:
                status = "🔶 AKZEPTABEL - verbesserungswürdig"
            else:
                status = "✅ GUT"

            print(f"   {bin_name:<12} Samples: {recall:>6.1%} Recall → {status}")

        print("\n" + "=" * 80)

    def create_plots(self, class_df: pd.DataFrame, group_df: pd.DataFrame) -> List[plt.Figure]:
        """Erstellt Visualisierungen."""

        figures = []

        # Plot 1: Scatter Plot - Sample Count vs Recall
        fig1, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1.1 Scatter mit Regression
        ax1 = axes[0, 0]
        valid_df = class_df[class_df['total_in_test'] >= 3]

        ax1.scatter(valid_df['sample_count'], valid_df['recall'],
                    alpha=0.6, edgecolors='black', linewidth=0.5, s=50)

        # Trendlinie
        if len(valid_df) > 5:
            z = np.polyfit(valid_df['sample_count'], valid_df['recall'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid_df['sample_count'].min(), valid_df['sample_count'].max(), 100)
            corr = valid_df['sample_count'].corr(valid_df['recall'])
            ax1.plot(x_line, p(x_line), "r--", alpha=0.8, label=f'Trend (r={corr:.2f})')
            ax1.legend()

        ax1.set_xlabel('Training Sample Count', fontsize=12)
        ax1.set_ylabel('Recall', fontsize=12)
        ax1.set_title('Sample Count vs. Recognition Recall', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)

        # 1.2 Log-Scale Scatter
        ax2 = axes[0, 1]
        ax2.scatter(valid_df['sample_count'], valid_df['recall'],
                    alpha=0.6, edgecolors='black', linewidth=0.5, s=50)
        ax2.set_xscale('log')
        ax2.set_xlabel('Training Sample Count (Log Scale)', fontsize=12)
        ax2.set_ylabel('Recall', fontsize=12)
        ax2.set_title('Sample Count vs. Recall (Log Scale)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 1.05)

        # Threshold-Linien
        for threshold, color in [(10, 'red'), (50, 'orange'), (100, 'green')]:
            ax2.axvline(x=threshold, color=color, linestyle='--', alpha=0.5, label=f'n={threshold}')
        ax2.legend()

        # 1.3 Box Plot nach Gruppen
        ax3 = axes[1, 0]
        order = ['1-5', '6-10', '11-20', '21-50', '51-100', '101-200', '200+']
        order = [o for o in order if o in class_df['fine_bin'].values]

        if len(order) > 0:
            plot_df = class_df[class_df['fine_bin'].isin(order)]
            sns.boxplot(data=plot_df, x='fine_bin', y='recall', order=order, ax=ax3, palette='viridis')

        ax3.set_xlabel('Sample Count Group', fontsize=12)
        ax3.set_ylabel('Recall', fontsize=12)
        ax3.set_title('Recall Distribution by Sample Count Group', fontsize=14)
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(-0.05, 1.05)

        # 1.4 Bar Chart - Overall Recall per Group
        ax4 = axes[1, 1]
        if len(group_df) > 0:
            colors = plt.cm.RdYlGn(group_df['overall_recall'])
            bars = ax4.bar(group_df['bin'], group_df['overall_recall'], color=colors, edgecolor='black')
            ax4.set_xlabel('Sample Count Group', fontsize=12)
            ax4.set_ylabel('Overall Recall', fontsize=12)
            ax4.set_title('Overall Recall by Sample Count Group', fontsize=14)
            ax4.tick_params(axis='x', rotation=45)
            ax4.set_ylim(0, 1)

            # Werte auf Bars
            for bar, acc in zip(bars, group_df['overall_recall']):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                         f'{acc:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        fig1.savefig(self.output_dir / 'sample_count_vs_recall.png', dpi=150, bbox_inches='tight')
        figures.append(fig1)

        # Plot 2: Histogramm der Sample Counts
        fig2, ax = plt.subplots(figsize=(10, 6))

        ax.hist(class_df['sample_count'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Training Sample Count', fontsize=12)
        ax.set_ylabel('Number of Classes', fontsize=12)
        ax.set_title('Distribution of Training Samples per Class', fontsize=14)
        ax.axvline(x=10, color='red', linestyle='--', label='Threshold: 10')
        ax.axvline(x=50, color='orange', linestyle='--', label='Threshold: 50')
        ax.legend()

        plt.tight_layout()
        fig2.savefig(self.output_dir / 'sample_distribution.png', dpi=150, bbox_inches='tight')
        figures.append(fig2)

        return figures

    def save_results(self, class_df: pd.DataFrame, group_df: pd.DataFrame):
        """Speichert die Ergebnisse als CSV."""
        class_df.to_csv(self.output_dir / 'class_performance.csv', index=False)
        group_df.to_csv(self.output_dir / 'group_performance.csv', index=False)
        print(f"\n💾 Ergebnisse gespeichert in: {self.output_dir}")


def run_sample_count_analysis(
        model: torch.nn.Module,
        train_dataset: SignLanguageDataset,
        test_loader: torch.utils.data.DataLoader,
        vocabulary: Vocabulary,
        config: Config,
        output_dir: str = "./results/sample_analysis"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hauptfunktion zum Ausführen der Sample Count Analyse.

    Args:
        model: Trainiertes Modell
        train_dataset: Training Dataset (für Sample Counts)
        test_loader: Test DataLoader
        vocabulary: Vocabulary
        config: Config object
        output_dir: Output Verzeichnis

    Returns:
        class_df: Per-Class Ergebnisse
        group_df: Per-Group Ergebnisse
    """
    print("\n" + "=" * 80)
    print("STARTING SAMPLE COUNT ANALYSIS")
    print("=" * 80)

    analyzer = SampleCountAnalyzer(output_dir=output_dir)

    # 1. Zähle Training Samples pro Klasse
    print("\n📊 Counting training samples per class...")
    class_counts = analyzer.count_training_samples(train_dataset, vocabulary)
    print(f"   Found {len(class_counts)} classes with training samples")

    # 2. Sammle Predictions
    print("\n🔮 Collecting predictions on test set...")
    pred_sequences, target_sequences, _ = analyzer.collect_predictions(
        model=model,
        dataloader=test_loader,
        vocabulary=vocabulary,
        device=config.model.device,
        use_mixed_precision=config.training.use_mixed_precision
    )
    print(f"   Collected {len(pred_sequences)} predictions")

    # 3. Analysiere
    print("\n📈 Analyzing performance by sample count...")
    class_df, group_df = analyzer.analyze(
        pred_sequences=pred_sequences,
        target_sequences=target_sequences,
        class_counts=class_counts,
        vocabulary=vocabulary
    )

    # 4. Print Report
    analyzer.print_report(class_df, group_df)

    # 5. Erstelle Plots
    print("\n📊 Creating visualizations...")
    figures = analyzer.create_plots(class_df, group_df)

    # 6. Speichere Ergebnisse
    analyzer.save_results(class_df, group_df)

    # Schließe Figures
    for fig in figures:
        plt.close(fig)

    return class_df, group_df


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
        gamma=config.training.focal_gamma,
        label_smoothing=config.training.label_smoothing
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

        # ============================================
        # FINAL EVALUATION ON TEST SET
        # ============================================

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

        # ============================================
        # SAMPLE COUNT ANALYSIS (NEW!)
        # ============================================

        print("\n" + "=" * 80)
        print("SAMPLE COUNT VS. PERFORMANCE ANALYSIS")
        print("=" * 80)

        try:
            # Get training dataset for sample counting (without augmentation)
            train_dataset = SignLanguageDataset(
                data_dir=config.data.train_dir,
                vocabulary=vocabulary,
                config=config.data,
                split='train',
                augment=False  # No augmentation for counting
            )

            # Run analysis
            analysis_dir = checkpoint_dir / "sample_analysis"
            class_df, group_df = run_sample_count_analysis(
                model=model,
                train_dataset=train_dataset,
                test_loader=test_loader,
                vocabulary=vocabulary,
                config=config,
                output_dir=str(analysis_dir)
            )

            # Log to MLflow
            if tracking_mode == "mlflow":
                # Log CSV files
                mlflow.log_artifact(str(analysis_dir / "class_performance.csv"))
                mlflow.log_artifact(str(analysis_dir / "group_performance.csv"))

                # Log plots
                for plot_file in analysis_dir.glob("*.png"):
                    mlflow.log_artifact(str(plot_file))

                # Log key metrics per group
                for _, row in group_df.iterrows():
                    bin_name = row['bin'].replace('-', '_').replace('+', 'plus')
                    mlflow.log_metric(f"recall_bin_{bin_name}", row['overall_recall'])

            print("\n✅ Sample count analysis complete!")

        except Exception as e:
            print(f"\n⚠️  Sample count analysis failed: {e}")
            import traceback
            traceback.print_exc()

        # ============================================
        # FINAL SUMMARY
        # ============================================

        # Log model to MLflow
        if tracking_mode == "mlflow":
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