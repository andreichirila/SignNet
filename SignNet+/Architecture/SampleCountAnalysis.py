"""
SignNet: Sample Count vs. Prediction Performance Analysis
Analysiert die Erkennungsleistung basierend auf der Anzahl der Trainingssamples pro Klasse.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.cuda.amp import autocast

# Local imports
from Config import Config
from Dataset import Vocabulary, SignLanguageDataset
from Train import decode_ctc_greedy


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
            for batch in dataloader:
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
            # Für eine genauere Analyse könnte man Alignment verwenden
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
            std_recall = group_data['recall'].std()
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
        print(f"\n{'Gloss':<20} {'Samples':>8} {'Test':>6} {'Recall':>8}")
        print("-" * 50)
        for _, row in worst.iterrows():
            print(f"{row['gloss']:<20} {row['sample_count']:>8} {row['total_in_test']:>6} {row['recall']:>7.1%}")

        print("\n" + "-" * 80)
        print("BEST PERFORMING CLASSES (Top 10)")
        print("-" * 80)

        best = class_df[class_df['total_in_test'] >= 3].nlargest(10, 'recall')
        print(f"\n{'Gloss':<20} {'Samples':>8} {'Test':>6} {'Recall':>8}")
        print("-" * 50)
        for _, row in best.iterrows():
            print(f"{row['gloss']:<20} {row['sample_count']:>8} {row['total_in_test']:>6} {row['recall']:>7.1%}")

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

        # Plot 2: Heatmap - Class Count Distribution
        fig2, ax = plt.subplots(figsize=(10, 6))

        # Histogramm der Sample Counts
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
# STANDALONE TEST
# ============================================

if __name__ == "__main__":
    from Config import get_config
    from Dataset import create_dataloaders, SignLanguageDataset
    from Model import SignLanguageTransformer

    print("\n" + "=" * 80)
    print("SAMPLE COUNT ANALYSIS - STANDALONE TEST")
    print("=" * 80)

    # Config
    config = get_config(top_k=200, use_augmentation=False)
    device = config.model.device

    # Load data
    train_loader, dev_loader, test_loader, vocabulary = create_dataloaders(config)

    # Get training dataset directly for sample counting
    train_dataset = SignLanguageDataset(
        data_dir=config.data.train_dir,
        vocabulary=vocabulary,
        config=config.data,
        split='train',
        augment=False
    )

    # Load model
    print("\n🧠 Loading model...")
    model = SignLanguageTransformer(config.model)

    # Try to load checkpoint
    checkpoint_path = Path(config.training.checkpoint_dir) / "best_model.pt"
    if checkpoint_path.exists():
        print(f"   Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("   ⚠️  No checkpoint found, using random weights (for testing)")

    model = model.to(device)
    model.eval()

    # Run analysis
    class_df, group_df = run_sample_count_analysis(
        model=model,
        train_dataset=train_dataset,
        test_loader=test_loader,
        vocabulary=vocabulary,
        config=config,
        output_dir="./results/sample_analysis"
    )

    print("\n✅ Analysis complete!")