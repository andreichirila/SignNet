import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
from collections import defaultdict
import torch


class MisclassificationAnalyzer:
    """Comprehensive misclassification analysis for sign language classification."""

    def __init__(self, model, device, idx_to_word, model_save_dir="./analysis"):
        self.model = model
        self.device = device
        self.idx_to_word = idx_to_word
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)

        # Storage for analysis
        self.all_predictions = []
        self.all_labels = []
        self.all_confidences = []
        self.all_sequences = []
        self.misclassified_samples = defaultdict(list)

    def evaluate_and_collect(self, data_loader):
        """Evaluate model and collect all predictions for analysis."""
        self.model.eval()

        with torch.no_grad():
            for landmarks, labels, seq_lengths in tqdm(data_loader, desc="Collecting predictions"):
                landmarks = landmarks.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(landmarks)
                probs = torch.softmax(logits, dim=1)
                confidences, predictions = torch.max(probs, dim=1)

                self.all_predictions.extend(predictions.cpu().numpy())
                self.all_labels.extend(labels.cpu().numpy())
                self.all_confidences.extend(confidences.cpu().numpy())
                self.all_sequences.extend(seq_lengths.cpu().numpy())

        self.all_predictions = np.array(self.all_predictions)
        self.all_labels = np.array(self.all_labels)
        self.all_confidences = np.array(self.all_confidences)
        self.all_sequences = np.array(self.all_sequences)

    def get_confusion_matrix(self, normalize=True):
        """Generate and visualize confusion matrix."""
        cm = confusion_matrix(self.all_labels, self.all_predictions)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Create figure
        fig, ax = plt.subplots(figsize=(20, 20))

        num_classes = len(self.idx_to_word)
        class_names = [self.idx_to_word[i] for i in range(num_classes)]

        sns.heatmap(cm, annot=False, cmap='YlOrRd', ax=ax,
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Normalized Frequency'})

        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        cm_path = os.path.join(self.model_save_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix saved: {cm_path}")
        plt.close()

        return cm

    def get_per_class_metrics(self):
        """Detailed metrics per class."""
        report = classification_report(
            self.all_labels,
            self.all_predictions,
            target_names=[self.idx_to_word[i] for i in range(len(self.idx_to_word))],
            output_dict=True,
            zero_division=0
        )

        # Extract per-class metrics
        per_class_metrics = {}
        for class_idx in range(len(self.idx_to_word)):
            class_name = self.idx_to_word[class_idx]
            class_str = str(class_idx)

            if class_str in report:
                per_class_metrics[class_name] = {
                    'precision': report[class_str]['precision'],
                    'recall': report[class_str]['recall'],
                    'f1': report[class_str]['f1-score'],
                    'support': int(report[class_str]['support'])
                }

        return per_class_metrics

    def analyze_top_misclassifications(self, top_n=20):
        """Find most common misclassification pairs."""
        misclassified_mask = self.all_predictions != self.all_labels

        misclassified_true = self.all_labels[misclassified_mask]
        misclassified_pred = self.all_predictions[misclassified_mask]
        misclassified_conf = self.all_confidences[misclassified_mask]

        # Create confusion pairs
        confusion_pairs = defaultdict(lambda: {'count': 0, 'avg_confidence': 0})

        for true_label, pred_label, conf in zip(misclassified_true, misclassified_pred, misclassified_conf):
            key = f"{self.idx_to_word[true_label]} → {self.idx_to_word[pred_label]}"
            confusion_pairs[key]['count'] += 1
            confusion_pairs[key]['avg_confidence'] = (
                (confusion_pairs[key]['avg_confidence'] * (confusion_pairs[key]['count'] - 1) + conf)
                / confusion_pairs[key]['count']
            )

        # Sort by frequency
        sorted_pairs = sorted(
            confusion_pairs.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )[:top_n]

        print(f"\n{'='*80}")
        print("TOP MISCLASSIFICATION PAIRS")
        print(f"{'='*80}")
        print(f"{'Confusion Pair':<40} {'Count':<10} {'Avg Confidence':<15}")
        print(f"{'-'*80}")

        for pair, metrics in sorted_pairs:
            print(f"{pair:<40} {metrics['count']:<10} {metrics['avg_confidence']:<15.4f}")

        return sorted_pairs

    def analyze_confidence_vs_accuracy(self):
        """Analyze relationship between model confidence and accuracy."""
        # Bin confidences
        confidence_bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(self.all_confidences, confidence_bins)

        bin_stats = {}
        for bin_idx in np.unique(bin_indices):
            mask = bin_indices == bin_idx
            bin_accs = (self.all_predictions[mask] == self.all_labels[mask])
            bin_acc = bin_accs.mean()
            bin_count = mask.sum()
            bin_conf = self.all_confidences[mask].mean()

            bin_stats[bin_idx] = {
                'accuracy': bin_acc,
                'count': bin_count,
                'avg_confidence': bin_conf
            }

        # Plot confidence calibration
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy vs Confidence
        bin_indices_sorted = sorted(bin_stats.keys())
        accuracies = [bin_stats[b]['accuracy'] for b in bin_indices_sorted]
        confidences = [bin_stats[b]['avg_confidence'] for b in bin_indices_sorted]
        counts = [bin_stats[b]['count'] for b in bin_indices_sorted]

        scatter = ax1.scatter(confidences, accuracies, s=[c*2 for c in counts],
                             alpha=0.6, c=confidences, cmap='viridis')
        ax1.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect Calibration')
        ax1.set_xlabel('Mean Confidence', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Confidence Calibration', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Confidence')

        # Sample count per bin
        ax2.bar(range(len(counts)), counts, color='steelblue', alpha=0.7)
        ax2.set_xlabel('Confidence Bin', fontsize=12)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.set_title('Sample Distribution by Confidence', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        calib_path = os.path.join(self.model_save_dir, 'confidence_calibration.png')
        plt.savefig(calib_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confidence calibration plot saved: {calib_path}")
        plt.close()

    def analyze_sequence_length_effect(self):
        """Analyze how sequence length affects accuracy."""
        seq_len_bins = np.percentile(self.all_sequences, [0, 25, 50, 75, 100])
        bin_indices = np.digitize(self.all_sequences, seq_len_bins)

        fig, ax = plt.subplots(figsize=(12, 6))

        bin_stats = {}
        for bin_idx in np.unique(bin_indices):
            mask = bin_indices == bin_idx
            bin_acc = (self.all_predictions[mask] == self.all_labels[mask]).mean()
            bin_count = mask.sum()
            avg_seq_len = self.all_sequences[mask].mean()

            bin_stats[bin_idx] = {
                'accuracy': bin_acc,
                'count': bin_count,
                'avg_seq_len': avg_seq_len
            }

        bin_indices_sorted = sorted(bin_stats.keys())
        seq_lens = [bin_stats[b]['avg_seq_len'] for b in bin_indices_sorted]
        accuracies = [bin_stats[b]['accuracy'] for b in bin_indices_sorted]
        counts = [bin_stats[b]['count'] for b in bin_indices_sorted]

        ax.scatter(seq_lens, accuracies, s=[c*2 for c in counts], alpha=0.6)
        ax.set_xlabel('Average Sequence Length (frames)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Effect of Sequence Length on Accuracy', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        seq_path = os.path.join(self.model_save_dir, 'sequence_length_effect.png')
        plt.savefig(seq_path, dpi=150, bbox_inches='tight')
        print(f"✓ Sequence length effect plot saved: {seq_path}")
        plt.close()

    def get_hardest_classes(self):
        """Identify classes with worst performance."""
        per_class_metrics = self.get_per_class_metrics()

        # Sort by F1 score
        sorted_classes = sorted(
            per_class_metrics.items(),
            key=lambda x: x[1]['f1'],
            reverse=False
        )

        print(f"\n{'='*80}")
        print("HARDEST CLASSES (Lowest F1 Score)")
        print(f"{'='*80}")
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print(f"{'-'*80}")

        for class_name, metrics in sorted_classes[:10]:
            print(f"{class_name:<20} {metrics['precision']:<12.3f} "
                  f"{metrics['recall']:<12.3f} {metrics['f1']:<12.3f} "
                  f"{metrics['support']:<10.0f}")

        return sorted_classes

    def generate_full_report(self, data_loader):
        """Run complete analysis pipeline."""
        print("\n" + "="*80)
        print("COMPREHENSIVE MISCLASSIFICATION ANALYSIS")
        print("="*80)

        print("\n[1/6] Collecting predictions...")
        self.evaluate_and_collect(data_loader)

        overall_acc = (self.all_predictions == self.all_labels).mean()
        print(f"  Overall Accuracy: {overall_acc:.2%}")
        print(f"  Total Samples: {len(self.all_labels)}")
        print(f"  Misclassified: {(self.all_predictions != self.all_labels).sum()}")

        print("\n[2/6] Analyzing per-class metrics...")
        per_class_metrics = self.get_per_class_metrics()

        print("\n[3/6] Generating confusion matrix...")
        self.get_confusion_matrix()

        print("\n[4/6] Analyzing top misclassification pairs...")
        self.analyze_top_misclassifications(top_n=20)

        print("\n[5/6] Analyzing confidence calibration...")
        self.analyze_confidence_vs_accuracy()

        print("\n[6/6] Analyzing sequence length effects...")
        self.analyze_sequence_length_effect()

        print("\n[7/7] Identifying hardest classes...")
        self.get_hardest_classes()

        print("\n" + "="*80)
        print(f"Analysis complete! Results saved to: {self.model_save_dir}")
        print("="*80)
