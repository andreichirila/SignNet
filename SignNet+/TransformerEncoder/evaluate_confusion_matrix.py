#!/usr/bin/env python3
"""
evaluate_confusion_matrix.py

Full evaluation script that:
1. Loads trained model
2. Runs inference on validation set
3. Creates detailed confusion matrix
4. Identifies most confused class pairs
5. Analyzes performance by sample count stratum

Usage:
    python evaluate_confusion_matrix.py \
        --model-path ./models_balanced/sign_classifier_best_enhanced.pth \
        --data-dir ./word_landmarks_extracted \
        --vocab-path ./main_vocab.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

# MLflow
import mlflow

os.environ['MLFLOW_TRACKING_USERNAME'] = 'andrei'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'andrei'
mlflow.set_tracking_uri("https://mlflow.schlaepfer.me")

from SignNetConfig import (
    MAIN_MODEL_CONFIG,
    SAMPLE_COUNT_THRESHOLDS,
    HIERARCHY_CONFIG
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model with confusion matrix')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--vocab-path', type=str, required=True,
                        help='Path to vocabulary JSON')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='Output directory')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--val-indices-path', type=str, default=None,
                        help='Path to validation indices .npy file')
    return parser.parse_args()


# ==================== MODEL DEFINITION ====================
class TransformerSignClassifierWithHandedness(nn.Module):
    """Same architecture as training script."""

    def __init__(self, input_size, hidden_size, num_classes, num_layers=2,
                 num_heads=4, dim_feedforward=512, dropout_rate=0.3,
                 attention_dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 2048, hidden_size))
        nn.init.normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=attention_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_sign = nn.Linear(hidden_size, num_classes)
        self.fc_handedness = nn.Linear(hidden_size, 4)

    def forward(self, landmarks, src_key_padding_mask=None):
        B, T, D = landmarks.shape
        x = self.input_proj(landmarks)

        if T > self.pos_embedding.size(1):
            raise ValueError(f"Sequence length {T} exceeds max {self.pos_embedding.size(1)}")

        pos_emb = self.pos_embedding[:, :T, :]
        x = x + pos_emb
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).float().unsqueeze(-1)
            x_masked = x * mask
            lengths = mask.sum(dim=1).clamp(min=1.0)
            pooled = x_masked.sum(dim=1) / lengths
        else:
            pooled = x.mean(dim=1)

        pooled = self.dropout(pooled)
        sign_logits = self.fc_sign(pooled)
        handedness_logits = self.fc_handedness(pooled)

        return sign_logits, handedness_logits


# ==================== DATASET ====================
class SimpleSignDataset(Dataset):
    """Simplified dataset for evaluation."""

    def __init__(self, npz_dir, word_to_idx, indices=None):
        self.npz_dir = Path(npz_dir)
        self.npz_files = sorted(self.npz_dir.glob("*.npz"))
        self.word_to_idx = word_to_idx

        if indices is not None:
            self.npz_files = [self.npz_files[i] for i in indices if i < len(self.npz_files)]

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        data = np.load(self.npz_files[idx], allow_pickle=True)
        landmarks = data["landmarks"].astype(np.float32)
        gloss = str(data["glosses"][0])
        label = self.word_to_idx.get(gloss, -1)

        handedness = 3  # Default: NONE
        if "handedness" in data:
            h_data = data["handedness"]
            left = sum(1 for h in h_data if "LEFT" in str(h))
            right = sum(1 for h in h_data if "RIGHT" in str(h))
            if left > 0 and right == 0:
                handedness = 0
            elif right > 0 and left == 0:
                handedness = 1
            elif left > 0 and right > 0:
                handedness = 2

        return (torch.from_numpy(landmarks).float(),
                torch.tensor(label, dtype=torch.long),
                torch.tensor(handedness, dtype=torch.long),
                gloss)


def collate_fn(batch):
    landmarks_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    handedness = torch.stack([item[2] for item in batch])
    glosses = [item[3] for item in batch]

    lengths = [lm.shape[0] for lm in landmarks_list]
    max_len = max(lengths)

    padded = []
    for lm in landmarks_list:
        pad_size = max_len - lm.shape[0]
        if pad_size > 0:
            lm = F.pad(lm, (0, 0, 0, pad_size), value=0.0)
        padded.append(lm)

    landmarks = torch.stack(padded)

    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        if l < max_len:
            mask[i, l:] = True

    return landmarks, labels, handedness, mask, glosses


def run_inference(model, dataloader, device):
    """Run inference and collect all predictions."""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_glosses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            landmarks, labels, handedness, mask, glosses = batch
            landmarks = landmarks.to(device)
            mask = mask.to(device)

            sign_logits, _ = model(landmarks, src_key_padding_mask=mask)
            probs = F.softmax(sign_logits, dim=1)
            preds = torch.argmax(sign_logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_glosses.extend(glosses)

    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs),
        'glosses': all_glosses
    }


def plot_confusion_matrix(conf_mat, class_names, output_path, top_n=40):
    """Plot confusion matrix for top-N most confused classes."""

    # Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        conf_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        conf_norm = np.nan_to_num(conf_norm)

    # Find worst performing classes (lowest diagonal)
    diagonal = np.diag(conf_norm)
    worst_idx = np.argsort(diagonal)[:top_n]

    # Subset
    conf_subset = conf_norm[worst_idx][:, worst_idx]
    names_subset = [class_names[i] for i in worst_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(18, 16))

    sns.heatmap(conf_subset,
                xticklabels=names_subset,
                yticklabels=names_subset,
                cmap='Blues',
                annot=True,
                fmt='.2f',
                ax=ax,
                square=True,
                linewidths=0.3,
                annot_kws={'size': 7})

    ax.set_title(f'Confusion Matrix - Top {top_n} Most Confused Classes\n(Normalized by True Class)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)

    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved confusion matrix to {output_path}")
    plt.close()


def plot_full_confusion_matrix(conf_mat, class_names, output_path):
    """Plot full confusion matrix (may be large)."""
    n = len(class_names)

    # Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        conf_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        conf_norm = np.nan_to_num(conf_norm)

    fig, ax = plt.subplots(figsize=(max(20, n * 0.3), max(18, n * 0.25)))

    sns.heatmap(conf_norm,
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Blues',
                ax=ax,
                square=True,
                cbar_kws={'label': 'Proportion', 'shrink': 0.5})

    ax.set_title(f'Full Confusion Matrix ({n} classes)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)

    plt.xticks(rotation=90, ha='right', fontsize=5)
    plt.yticks(rotation=0, fontsize=5)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved full confusion matrix to {output_path}")
    plt.close()


def find_top_confused_pairs(conf_mat, idx_to_word, top_n=30):
    """Find most confused class pairs."""
    pairs = []

    for true_idx in range(conf_mat.shape[0]):
        total = conf_mat[true_idx].sum()
        if total == 0:
            continue

        for pred_idx in range(conf_mat.shape[1]):
            if true_idx == pred_idx:
                continue

            count = conf_mat[true_idx, pred_idx]
            if count > 0:
                pairs.append({
                    'true_class': idx_to_word[true_idx],
                    'pred_class': idx_to_word[pred_idx],
                    'count': int(count),
                    'rate': count / total,
                    'support': int(total)
                })

    pairs.sort(key=lambda x: x['count'], reverse=True)
    return pairs[:top_n]


def analyze_by_sample_count(results, sample_counts, idx_to_word, thresholds=None):
    """Stratified analysis by sample count."""
    if thresholds is None:
        thresholds = SAMPLE_COUNT_THRESHOLDS

    strata = {k: {'correct': 0, 'total': 0, 'classes': set()} for k in thresholds}

    preds = results['predictions']
    labels = results['labels']

    for pred, label in zip(preds, labels):
        if label < 0:
            continue

        class_name = idx_to_word.get(label, 'UNKNOWN')
        count = sample_counts.get(class_name, 0)

        # Find stratum
        for stratum, (low, high) in thresholds.items():
            if low <= count < high:
                strata[stratum]['total'] += 1
                strata[stratum]['classes'].add(class_name)
                if pred == label:
                    strata[stratum]['correct'] += 1
                break

    # Compute accuracies
    for stratum in strata:
        total = strata[stratum]['total']
        correct = strata[stratum]['correct']
        strata[stratum]['accuracy'] = correct / total if total > 0 else 0
        strata[stratum]['num_classes'] = len(strata[stratum]['classes'])
        del strata[stratum]['classes']  # Remove set for JSON serialization

    return strata


def plot_stratum_analysis(strata, output_path):
    """Plot accuracy by sample count stratum."""
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(strata.keys())
    accs = [strata[n]['accuracy'] for n in names]
    totals = [strata[n]['total'] for n in names]
    num_classes = [strata[n]['num_classes'] for n in names]

    colors = ['#e74c3c', '#f39c12', '#27ae60']
    bars = ax.bar(names, accs, color=colors, alpha=0.8, edgecolor='black')

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Sample Count Category', fontsize=12)
    ax.set_title('Validation Accuracy by Sample Count Category', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)

    # Add labels
    for bar, acc, total, nc in zip(bars, accs, totals, num_classes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{acc:.1%}\n({nc} classes, {total} samples)',
                ha='center', va='bottom', fontsize=9)

    # Add threshold info
    labels = []
    for name in names:
        low, high = SAMPLE_COUNT_THRESHOLDS[name]
        if high == float('inf'):
            labels.append(f'{name}\n(≥{low})')
        else:
            labels.append(f'{name}\n({low}-{high})')
    ax.set_xticklabels(labels)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved stratum analysis to {output_path}")
    plt.close()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load vocabulary
    print(f"\n[1/6] Loading vocabulary from {args.vocab_path}...")
    with open(args.vocab_path, 'r') as f:
        vocab = json.load(f)

    word_to_idx = vocab['word_to_idx']
    idx_to_word = {int(k): v for k, v in vocab['idx_to_word'].items()}
    num_classes = vocab['num_classes']
    print(f"  {num_classes} classes")

    # Load model
    print(f"\n[2/6] Loading model from {args.model_path}...")
    model = TransformerSignClassifierWithHandedness(
        input_size=MAIN_MODEL_CONFIG['input_size'],
        hidden_size=MAIN_MODEL_CONFIG['hidden_size'],
        num_classes=num_classes,
        num_layers=MAIN_MODEL_CONFIG['num_layers'],
        num_heads=MAIN_MODEL_CONFIG['num_heads'],
        dim_feedforward=MAIN_MODEL_CONFIG['dim_feedforward'],
        dropout_rate=0.0,  # No dropout during inference
        attention_dropout=0.0
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"  Loaded from 'model_state_dict' key")
    else:
        state_dict = checkpoint

    # Handle torch.compile _orig_mod. prefix
    has_orig_mod = any(k.startswith("_orig_mod.") for k in state_dict.keys())
    if has_orig_mod:
        print(f"  Detected torch.compile checkpoint, removing '_orig_mod.' prefix...")
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "")
            new_state_dict[new_key] = v
        state_dict = new_state_dict

    # Verify state dict matches model
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    if missing:
        print(f"  ⚠️  Missing keys: {list(missing)[:5]}...")
    if unexpected:
        print(f"  ⚠️  Unexpected keys: {list(unexpected)[:5]}...")

    # Load state dict
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"  ✓ Model loaded successfully (strict=True)")
    except RuntimeError as e:
        print(f"  ⚠️  Strict loading failed: {e}")
        print(f"  Trying with strict=False...")
        model.load_state_dict(state_dict, strict=False)
        print(f"  ✓ Model loaded with strict=False")

    model.eval()

    # Load dataset
    print(f"\n[3/6] Loading dataset from {args.data_dir}...")

    # Get validation indices
    if args.val_indices_path and os.path.exists(args.val_indices_path):
        val_indices = np.load(args.val_indices_path)
        print(f"  Loaded {len(val_indices)} validation indices")
    else:
        # Create split
        all_files = sorted(Path(args.data_dir).glob("*.npz"))
        all_labels = []
        for f in all_files:
            data = np.load(f, allow_pickle=True)
            gloss = str(data['glosses'][0])
            if gloss in word_to_idx:
                all_labels.append(gloss)
            else:
                all_labels.append('_UNKNOWN_')

        indices = list(range(len(all_files)))
        _, val_indices = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=all_labels
        )
        print(f"  Created validation split: {len(val_indices)} samples")

    dataset = SimpleSignDataset(args.data_dir, word_to_idx, indices=val_indices)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=4)

    # Count samples per class
    print(f"\n[4/6] Counting samples per class...")
    sample_counts = Counter()
    for f in Path(args.data_dir).glob("*.npz"):
        try:
            data = np.load(f, allow_pickle=True)
            gloss = str(data['glosses'][0])
            if gloss in word_to_idx:
                sample_counts[gloss] += 1
        except:
            continue
    print(f"  {len(sample_counts)} classes in vocabulary")

    # Run inference
    print(f"\n[5/6] Running inference...")
    results = run_inference(model, dataloader, device)

    # Filter out invalid labels (-1)
    valid_mask = results['labels'] >= 0
    results['predictions'] = results['predictions'][valid_mask]
    results['labels'] = results['labels'][valid_mask]
    results['probabilities'] = results['probabilities'][valid_mask]

    print(f"  {len(results['labels'])} valid samples")

    # Compute metrics
    accuracy = (results['predictions'] == results['labels']).mean()
    print(f"\n  Overall Accuracy: {accuracy:.2%}")

    # Confusion matrix
    print(f"\n[6/6] Generating analysis...")

    conf_mat = confusion_matrix(results['labels'], results['predictions'],
                                labels=range(num_classes))

    class_names = [idx_to_word.get(i, f'UNK_{i}') for i in range(num_classes)]

    # Plot confusion matrices
    plot_confusion_matrix(conf_mat, class_names,
                          os.path.join(args.output_dir, f'confusion_top40_{timestamp}.png'),
                          top_n=40)

    plot_full_confusion_matrix(conf_mat, class_names,
                               os.path.join(args.output_dir, f'confusion_full_{timestamp}.png'))

    # Top confused pairs
    confused_pairs = find_top_confused_pairs(conf_mat, idx_to_word, top_n=50)

    print("\n" + "=" * 60)
    print("TOP 20 CONFUSED CLASS PAIRS")
    print("=" * 60)
    for i, pair in enumerate(confused_pairs[:20], 1):
        print(f"{i:2}. {pair['true_class']:20} → {pair['pred_class']:20} "
              f"({pair['count']:4} errors, {pair['rate']:.1%})")

    # Stratum analysis
    strata = analyze_by_sample_count(results, sample_counts, idx_to_word)

    print("\n" + "=" * 60)
    print("ACCURACY BY SAMPLE COUNT")
    print("=" * 60)
    for name in ['low', 'mid', 'high']:
        s = strata[name]
        thresh = SAMPLE_COUNT_THRESHOLDS[name]
        print(f"  {name.upper():5} ({thresh[0]}-{thresh[1]}): "
              f"{s['accuracy']:.2%} ({s['num_classes']} classes, {s['total']} samples)")

    plot_stratum_analysis(strata, os.path.join(args.output_dir, f'stratum_analysis_{timestamp}.png'))

    # Classification report
    report = classification_report(results['labels'], results['predictions'],
                                   labels=range(num_classes),
                                   target_names=class_names,
                                   output_dict=True,
                                   zero_division=0)

    # Save results
    output_data = {
        'timestamp': timestamp,
        'model_path': args.model_path,
        'data_dir': args.data_dir,
        'num_samples': len(results['labels']),
        'num_classes': num_classes,
        'overall_accuracy': float(accuracy),
        'strata_analysis': strata,
        'top_confused_pairs': confused_pairs,
        'per_class_metrics': [
            {
                'class': class_names[i],
                'precision': report[class_names[i]]['precision'],
                'recall': report[class_names[i]]['recall'],
                'f1': report[class_names[i]]['f1-score'],
                'support': report[class_names[i]]['support'],
                'sample_count': sample_counts.get(class_names[i], 0)
            }
            for i in range(num_classes)
        ]
    }

    json_path = os.path.join(args.output_dir, f'evaluation_results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n✓ Saved results to {json_path}")

    # Log to MLflow
    try:
        mlflow.set_experiment("SignNetWord")
        with mlflow.start_run(run_name=f"evaluation_{timestamp}"):
            mlflow.log_param("model_path", args.model_path)
            mlflow.log_param("num_samples", len(results['labels']))
            mlflow.log_metric("val_accuracy", accuracy)
            mlflow.log_metric("low_sample_accuracy", strata['low']['accuracy'])
            mlflow.log_metric("mid_sample_accuracy", strata['mid']['accuracy'])
            mlflow.log_metric("high_sample_accuracy", strata['high']['accuracy'])
            mlflow.log_artifacts(args.output_dir)
            print(f"✓ Logged to MLflow")
    except Exception as e:
        print(f"  [WARNING] MLflow logging failed: {e}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()