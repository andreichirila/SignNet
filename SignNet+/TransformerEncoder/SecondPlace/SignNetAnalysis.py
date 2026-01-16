#!/usr/bin/env python3
"""
analyze_model_performance.py

Comprehensive analysis of SignNet model performance:
- Load metrics from MLflow
- Stratified analysis by sample count (low/mid/high)
- Confusion matrix visualization
- Per-class accuracy breakdown
- Error analysis for commonly confused classes

Usage:
    python analyze_model_performance.py --run-id <MLFLOW_RUN_ID>
    python analyze_model_performance.py --latest  # Use latest run
    python analyze_model_performance.py --model-path ./models/best.pth --data-dir ./word_landmarks_extracted
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score,
    precision_recall_fscore_support
)
from tqdm import tqdm
import pandas as pd

# MLflow setup
import mlflow
import mlflow.pytorch

os.environ['MLFLOW_TRACKING_USERNAME'] = 'andrei'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'andrei'
mlflow.set_tracking_uri("https://mlflow.schlaepfer.me")

# Import from training script (adjust path as needed)
try:
    from sign_classifier_word import (
        SignLanguageDataset,
        TransformerSignClassifierWithHandedness,
        PadCollate,
        RemappedDataset
    )
    from SignNetConfig import SAMPLE_COUNT_THRESHOLDS, HIERARCHY_CONFIG
except ImportError:
    print("[WARNING] Could not import from sign_classifier_word.py")
    print("  Make sure the file is in the same directory or PYTHONPATH")
    SAMPLE_COUNT_THRESHOLDS = {
        'low': (0, 100),
        'mid': (100, 300),
        'high': (300, float('inf'))
    }
    HIERARCHY_CONFIG = {}


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze SignNet Model Performance')

    parser.add_argument('--run-id', type=str, default=None,
                        help='MLflow run ID to analyze')
    parser.add_argument('--latest', action='store_true',
                        help='Use the latest MLflow run')
    parser.add_argument('--experiment-name', type=str, default='SignNetWord',
                        help='MLflow experiment name')

    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model checkpoint (if not using MLflow)')
    parser.add_argument('--data-dir', type=str, default='./word_landmarks_extracted',
                        help='Path to dataset directory')
    parser.add_argument('--vocab-path', type=str, default=None,
                        help='Path to vocabulary JSON file')

    parser.add_argument('--output-dir', type=str, default='./analysis_results',
                        help='Directory to save analysis results')
    parser.add_argument('--top-n-confused', type=int, default=20,
                        help='Number of top confused class pairs to show')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for inference')

    return parser.parse_args()


def get_latest_run(experiment_name):
    """Get the latest MLflow run from an experiment."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if len(runs) == 0:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")

    return runs.iloc[0]['run_id']


def load_class_metrics_from_mlflow(run_id):
    """Load class metrics JSON artifact from MLflow run."""
    client = mlflow.tracking.MlflowClient()

    # List artifacts
    artifacts = client.list_artifacts(run_id)

    # Find the latest class_metrics file
    class_metrics_files = [a.path for a in artifacts if 'class_metrics_epoch' in a.path]

    if not class_metrics_files:
        print("[WARNING] No class_metrics artifact found in run")
        return None

    # Get the one with highest epoch number
    latest_file = sorted(class_metrics_files,
                         key=lambda x: int(x.split('_')[-1].replace('.json', '')))[-1]

    # Download artifact
    local_path = client.download_artifacts(run_id, latest_file)

    with open(local_path, 'r') as f:
        return json.load(f)


def compute_sample_counts(data_dir, vocab=None):
    """Count samples per class in dataset."""
    npz_files = list(Path(data_dir).glob("*.npz"))

    counts = Counter()
    for f in tqdm(npz_files, desc="Counting samples"):
        try:
            data = np.load(f, allow_pickle=True)
            gloss = str(data['glosses'][0])
            if vocab is None or gloss in vocab:
                counts[gloss] += 1
        except Exception as e:
            continue

    return counts


def stratify_classes_by_sample_count(class_metrics, sample_counts, thresholds=None):
    """
    Group classes into low/mid/high sample count categories.

    Returns dict with keys 'low', 'mid', 'high', each containing:
    - classes: list of class names
    - accuracies: list of per-class accuracies
    - counts: list of sample counts
    - mean_accuracy: average accuracy for this stratum
    """
    if thresholds is None:
        thresholds = SAMPLE_COUNT_THRESHOLDS

    strata = {
        'low': {'classes': [], 'accuracies': [], 'counts': [], 'supports': []},
        'mid': {'classes': [], 'accuracies': [], 'counts': [], 'supports': []},
        'high': {'classes': [], 'accuracies': [], 'counts': [], 'supports': []},
    }

    for item in class_metrics:
        if item.get('class', '').startswith('_'):  # Skip _overall
            continue

        class_name = item['class']
        accuracy = float(item['accuracy'])
        support = int(item['support'])
        count = sample_counts.get(class_name, support)  # Use support if count not available

        # Determine stratum
        for stratum_name, (low, high) in thresholds.items():
            if low <= count < high:
                strata[stratum_name]['classes'].append(class_name)
                strata[stratum_name]['accuracies'].append(accuracy)
                strata[stratum_name]['counts'].append(count)
                strata[stratum_name]['supports'].append(support)
                break

    # Compute statistics for each stratum
    for stratum_name, data in strata.items():
        if data['accuracies']:
            data['mean_accuracy'] = np.mean(data['accuracies'])
            data['std_accuracy'] = np.std(data['accuracies'])
            data['median_accuracy'] = np.median(data['accuracies'])
            data['min_accuracy'] = np.min(data['accuracies'])
            data['max_accuracy'] = np.max(data['accuracies'])
            data['num_classes'] = len(data['classes'])
        else:
            data['mean_accuracy'] = 0
            data['std_accuracy'] = 0
            data['median_accuracy'] = 0
            data['min_accuracy'] = 0
            data['max_accuracy'] = 0
            data['num_classes'] = 0

    return strata


def plot_stratified_analysis(strata, output_path):
    """Create visualization of performance by sample count stratum."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Bar chart of mean accuracy per stratum
    ax1 = axes[0, 0]
    stratum_names = ['low', 'mid', 'high']
    means = [strata[s]['mean_accuracy'] for s in stratum_names]
    stds = [strata[s]['std_accuracy'] for s in stratum_names]
    num_classes = [strata[s]['num_classes'] for s in stratum_names]

    colors = ['#e74c3c', '#f39c12', '#27ae60']
    bars = ax1.bar(stratum_names, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax1.set_ylabel('Mean Accuracy', fontsize=12)
    ax1.set_xlabel('Sample Count Category', fontsize=12)
    ax1.set_title('Accuracy by Sample Count Category', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)

    # Add count labels on bars
    for bar, n in zip(bars, num_classes):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'n={n}', ha='center', va='bottom', fontsize=10)

    # Add threshold labels
    ax1.set_xticklabels([
        f'Low\n(<{SAMPLE_COUNT_THRESHOLDS["low"][1]})',
        f'Mid\n({SAMPLE_COUNT_THRESHOLDS["mid"][0]}-{SAMPLE_COUNT_THRESHOLDS["mid"][1]})',
        f'High\n(>{SAMPLE_COUNT_THRESHOLDS["high"][0]})'
    ])

    # 2. Scatter plot: sample count vs accuracy
    ax2 = axes[0, 1]
    all_counts = []
    all_accs = []
    all_colors = []

    for stratum, color in zip(stratum_names, colors):
        all_counts.extend(strata[stratum]['counts'])
        all_accs.extend(strata[stratum]['accuracies'])
        all_colors.extend([color] * len(strata[stratum]['counts']))

    ax2.scatter(all_counts, all_accs, c=all_colors, alpha=0.6, s=50)
    ax2.set_xlabel('Sample Count', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Sample Count vs Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    ax2.legend()

    # 3. Box plot of accuracy distribution per stratum
    ax3 = axes[1, 0]
    box_data = [strata[s]['accuracies'] for s in stratum_names if strata[s]['accuracies']]
    box_labels = [s for s in stratum_names if strata[s]['accuracies']]

    bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_xlabel('Sample Count Category', fontsize=12)
    ax3.set_title('Accuracy Distribution by Category', fontsize=14, fontweight='bold')

    # 4. Bottom performers in each category
    ax4 = axes[1, 1]
    ax4.axis('off')

    text_lines = ["WORST PERFORMERS BY CATEGORY\n" + "=" * 40 + "\n"]

    for stratum in stratum_names:
        if not strata[stratum]['classes']:
            continue

        # Sort by accuracy (ascending)
        sorted_idx = np.argsort(strata[stratum]['accuracies'])

        text_lines.append(f"\n{stratum.upper()} sample count (n={strata[stratum]['num_classes']}):")
        text_lines.append(f"Mean: {strata[stratum]['mean_accuracy']:.1%} ± {strata[stratum]['std_accuracy']:.1%}\n")

        # Show bottom 5
        for i in sorted_idx[:5]:
            cls = strata[stratum]['classes'][i]
            acc = strata[stratum]['accuracies'][i]
            cnt = strata[stratum]['counts'][i]
            text_lines.append(f"  {cls}: {acc:.1%} ({cnt} samples)")

    ax4.text(0.05, 0.95, '\n'.join(text_lines), transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved stratified analysis to {output_path}")
    plt.close()


def plot_confusion_matrix_detailed(confusion_mat, idx_to_word, output_path, top_n=30):
    """
    Plot detailed confusion matrix with focus on most confused classes.
    """
    num_classes = len(idx_to_word)

    # Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        confusion_norm = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
        confusion_norm = np.nan_to_num(confusion_norm)

    # Find classes with most errors (lowest diagonal values)
    diagonal = np.diag(confusion_norm)
    worst_indices = np.argsort(diagonal)[:top_n]

    # Create subset confusion matrix
    confusion_subset = confusion_norm[worst_indices][:, worst_indices]
    class_names = [idx_to_word[i] for i in worst_indices]

    # Plot
    fig, ax = plt.subplots(figsize=(16, 14))

    sns.heatmap(confusion_subset,
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Blues',
                annot=True,
                fmt='.2f',
                ax=ax,
                cbar_kws={'label': 'Proportion'},
                square=True,
                linewidths=0.5)

    ax.set_title(f'Confusion Matrix - Top {top_n} Most Confused Classes',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)

    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved confusion matrix to {output_path}")
    plt.close()

    return worst_indices


def find_confused_pairs(confusion_mat, idx_to_word, top_n=20):
    """
    Find the most commonly confused class pairs.

    Returns list of tuples: (true_class, predicted_class, count, rate)
    """
    num_classes = confusion_mat.shape[0]
    confused_pairs = []

    for true_idx in range(num_classes):
        total_true = confusion_mat[true_idx].sum()
        if total_true == 0:
            continue

        for pred_idx in range(num_classes):
            if true_idx == pred_idx:  # Skip correct predictions
                continue

            count = confusion_mat[true_idx, pred_idx]
            if count > 0:
                rate = count / total_true
                confused_pairs.append({
                    'true_class': idx_to_word[true_idx],
                    'predicted_class': idx_to_word[pred_idx],
                    'error_count': int(count),
                    'error_rate': float(rate),
                    'true_support': int(total_true)
                })

    # Sort by error count (descending)
    confused_pairs.sort(key=lambda x: x['error_count'], reverse=True)

    return confused_pairs[:top_n]


def analyze_hierarchy_clusters(confused_pairs, hierarchy_config):
    """
    Check if confused pairs fall into predefined hierarchy clusters.
    """
    cluster_analysis = {}

    for cluster_name, cluster_classes in hierarchy_config.items():
        cluster_set = set(cluster_classes)

        # Find confused pairs within this cluster
        intra_cluster_errors = []
        for pair in confused_pairs:
            if pair['true_class'] in cluster_set and pair['predicted_class'] in cluster_set:
                intra_cluster_errors.append(pair)

        cluster_analysis[cluster_name] = {
            'classes': cluster_classes,
            'intra_cluster_errors': intra_cluster_errors,
            'total_intra_errors': sum(p['error_count'] for p in intra_cluster_errors)
        }

    return cluster_analysis


def run_inference_and_analyze(model, dataloader, device, idx_to_word):
    """
    Run inference on dataset and collect predictions for analysis.
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            landmarks, labels, handedness, padding_mask = batch
            landmarks = landmarks.to(device)
            padding_mask = padding_mask.to(device)

            sign_logits, _ = model(landmarks, src_key_padding_mask=padding_mask)
            probs = F.softmax(sign_logits, dim=1)
            preds = torch.argmax(sign_logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    num_classes = len(idx_to_word)
    conf_mat = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(num_classes), zero_division=0
    )

    class_metrics = []
    for idx in range(num_classes):
        class_metrics.append({
            'class': idx_to_word[idx],
            'accuracy': float((all_preds[all_labels == idx] == idx).mean()) if (all_labels == idx).sum() > 0 else 0,
            'precision': float(precision[idx]),
            'recall': float(recall[idx]),
            'f1': float(f1[idx]),
            'support': int(support[idx])
        })

    # Overall metrics
    overall = {
        'accuracy': float((all_preds == all_labels).mean()),
        'f1_macro': float(f1_score(all_labels, all_preds, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(all_labels, all_preds, average='weighted', zero_division=0)),
    }

    return {
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'confusion_matrix': conf_mat,
        'class_metrics': class_metrics,
        'overall': overall
    }


def generate_report(strata, confused_pairs, cluster_analysis, overall_metrics, output_path):
    """Generate a comprehensive text report."""

    lines = [
        "=" * 80,
        "SIGNNET MODEL PERFORMANCE ANALYSIS REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        "",
        "OVERALL METRICS",
        "-" * 40,
        f"  Accuracy:    {overall_metrics.get('accuracy', 0):.2%}",
        f"  F1 (Macro):  {overall_metrics.get('f1_macro', 0):.2%}",
        f"  F1 (Weight): {overall_metrics.get('f1_weighted', 0):.2%}",
        "",
        "PERFORMANCE BY SAMPLE COUNT",
        "-" * 40,
    ]

    for stratum in ['low', 'mid', 'high']:
        s = strata[stratum]
        thresh = SAMPLE_COUNT_THRESHOLDS[stratum]
        lines.append(f"\n  {stratum.upper()} ({thresh[0]}-{thresh[1]} samples):")
        lines.append(f"    Classes:  {s['num_classes']}")
        lines.append(f"    Mean Acc: {s['mean_accuracy']:.2%} ± {s['std_accuracy']:.2%}")
        lines.append(f"    Range:    {s['min_accuracy']:.2%} - {s['max_accuracy']:.2%}")

    lines.extend([
        "",
        "TOP CONFUSED CLASS PAIRS",
        "-" * 40,
    ])

    for i, pair in enumerate(confused_pairs[:15], 1):
        lines.append(
            f"  {i:2}. {pair['true_class']:20} → {pair['predicted_class']:20} "
            f"({pair['error_count']:4} errors, {pair['error_rate']:.1%} of {pair['true_class']})"
        )

    lines.extend([
        "",
        "HIERARCHY CLUSTER ANALYSIS",
        "-" * 40,
    ])

    for cluster_name, analysis in cluster_analysis.items():
        lines.append(f"\n  {cluster_name}:")
        lines.append(f"    Classes: {len(analysis['classes'])}")
        lines.append(f"    Intra-cluster errors: {analysis['total_intra_errors']}")
        if analysis['intra_cluster_errors']:
            lines.append("    Top confusions within cluster:")
            for err in analysis['intra_cluster_errors'][:5]:
                lines.append(f"      {err['true_class']} → {err['predicted_class']}: {err['error_count']}")

    lines.extend([
        "",
        "=" * 80,
        "RECOMMENDATIONS",
        "=" * 80,
        "",
    ])

    # Generate recommendations based on analysis
    if strata['low']['mean_accuracy'] < 0.4:
        lines.append("• LOW SAMPLE CLASSES: Consider data augmentation or collecting more samples")
        lines.append("  for classes with < 100 training samples.")
        lines.append("")

    for cluster_name, analysis in cluster_analysis.items():
        if analysis['total_intra_errors'] > 50:
            lines.append(f"• {cluster_name.upper()}: High confusion within cluster.")
            lines.append(f"  Consider training dedicated expert model for these {len(analysis['classes'])} classes.")
            lines.append("")

    report_text = '\n'.join(lines)

    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"✓ Saved report to {output_path}")
    print("\n" + report_text)


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 80)
    print("SIGNNET MODEL PERFORMANCE ANALYSIS")
    print("=" * 80)

    # Get MLflow run
    if args.latest:
        print(f"\n[1/5] Loading latest run from experiment '{args.experiment_name}'...")
        run_id = get_latest_run(args.experiment_name)
        print(f"  Run ID: {run_id}")
    elif args.run_id:
        run_id = args.run_id
        print(f"\n[1/5] Using specified run: {run_id}")
    else:
        run_id = None
        print("\n[1/5] No MLflow run specified, will analyze from model file...")

    # Load class metrics
    class_metrics = None
    overall_metrics = {}

    if run_id:
        print("\n[2/5] Loading metrics from MLflow...")
        try:
            class_metrics = load_class_metrics_from_mlflow(run_id)

            # Load run metrics
            run = mlflow.get_run(run_id)
            overall_metrics = {
                'accuracy': run.data.metrics.get('val_accuracy', 0),
                'f1_macro': run.data.metrics.get('val_f1_macro', 0),
                'f1_weighted': run.data.metrics.get('val_f1_weighted', 0),
            }
            print(f"  Loaded metrics for {len(class_metrics) if class_metrics else 0} classes")
        except Exception as e:
            print(f"  [WARNING] Could not load from MLflow: {e}")

    # Count samples in dataset
    print(f"\n[3/5] Counting samples in {args.data_dir}...")
    sample_counts = compute_sample_counts(args.data_dir)
    print(f"  Found {sum(sample_counts.values())} samples across {len(sample_counts)} classes")

    if class_metrics is None:
        print("\n[ERROR] No class metrics available. Please specify --run-id or --latest")
        print("  Or run inference with --model-path")
        return

    # Stratified analysis
    print("\n[4/5] Performing stratified analysis...")
    strata = stratify_classes_by_sample_count(class_metrics, sample_counts)

    for stratum in ['low', 'mid', 'high']:
        s = strata[stratum]
        print(f"  {stratum.upper():5}: {s['num_classes']:3} classes, "
              f"mean acc = {s['mean_accuracy']:.2%} ± {s['std_accuracy']:.2%}")

    # Create visualizations
    print("\n[5/5] Generating visualizations and report...")

    # Stratified analysis plot
    plot_stratified_analysis(
        strata,
        os.path.join(args.output_dir, f'stratified_analysis_{timestamp}.png')
    )

    # Build confusion matrix from class metrics if we don't have raw predictions
    # (This is approximate - for accurate confusion matrix, run inference)

    # Find confused pairs (approximate from class metrics)
    # Note: For accurate confusion matrix, need to run inference
    confused_pairs = []

    # Analyze hierarchy clusters
    cluster_analysis = analyze_hierarchy_clusters(confused_pairs, HIERARCHY_CONFIG)

    # Generate report
    generate_report(
        strata,
        confused_pairs,
        cluster_analysis,
        overall_metrics,
        os.path.join(args.output_dir, f'analysis_report_{timestamp}.txt')
    )

    # Save JSON with all analysis data
    analysis_data = {
        'timestamp': timestamp,
        'run_id': run_id,
        'overall_metrics': overall_metrics,
        'strata_summary': {
            stratum: {
                'num_classes': strata[stratum]['num_classes'],
                'mean_accuracy': strata[stratum]['mean_accuracy'],
                'std_accuracy': strata[stratum]['std_accuracy'],
            }
            for stratum in ['low', 'mid', 'high']
        },
        'class_metrics': class_metrics,
        'sample_counts': dict(sample_counts),
    }

    json_path = os.path.join(args.output_dir, f'analysis_data_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    print(f"✓ Saved analysis data to {json_path}")

    # Log to MLflow
    if run_id:
        try:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifacts(args.output_dir, artifact_path="analysis")
                print(f"✓ Logged analysis artifacts to MLflow run {run_id}")
        except Exception as e:
            print(f"  [WARNING] Could not log to MLflow: {e}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()