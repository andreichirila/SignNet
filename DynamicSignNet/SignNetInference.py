import os
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from datetime import datetime
from collections import Counter

# Import der Projekt-Module
from SignNetWord import TransformerSignClassifier, SignLanguageDataset, PadCollate, RemappedDataset
from SignNetConfig import MAIN_MODEL_CONFIG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path, config, num_classes):
    """Lädt das trainierte Modell und bereinigt ggf. Präfixe."""
    model = TransformerSignClassifier(
        input_size=config['input_size'], hidden_size=config['hidden_size'],
        num_classes=num_classes, num_layers=config['num_layers'],
        num_heads=config['num_heads'], dim_feedforward=config['dim_feedforward']
    )
    state_dict = torch.load(path, map_location=DEVICE)
    if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
    clean_sd = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd, strict=False)
    return model.to(DEVICE).eval()

def plot_accuracy_per_bucket(all_labels, all_preds, train_counts, out_path):
    """Visualisiert die Accuracy gruppiert nach Trainings-Häufigkeit."""
    bucket_results = {
        "Rare (< 100)": {"correct": 0, "total": 0},
        "Medium (100-300)": {"correct": 0, "total": 0},
        "Frequent (> 300)": {"correct": 0, "total": 0}
    }
    for label, pred in zip(all_labels, all_preds):
        count = train_counts.get(label, 0)
        b_name = "Rare (< 100)" if count < 100 else ("Medium (100-300)" if count <= 300 else "Frequent (> 300)")
        bucket_results[b_name]["total"] += 1
        if label == pred: bucket_results[b_name]["correct"] += 1

    names, accs = [], []
    for name in ["Rare (< 100)", "Medium (100-300)", "Frequent (> 300)"]:
        res = bucket_results[name]
        if res["total"] > 0:
            names.append(f"{name}\n(n={res['total']})"); accs.append(res["correct"] / res["total"])

    plt.figure(figsize=(9, 6))
    bars = plt.bar(names, accs, color=sns.color_palette("viridis", len(accs)), edgecolor='black', alpha=0.8)
    plt.ylim(0, 1.0); plt.ylabel("Accuracy (Top-1)"); plt.title("Genauigkeit nach Klassen-Häufigkeit")
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{bar.get_height():.1%}", ha='center', fontweight='bold')
    plt.savefig(out_path / "accuracy_per_bucket.png"); plt.close()

def plot_confidence_distribution(full_logits, all_labels, all_preds, out_path):
    """Visualisiert die Verteilung der Softmax-Konfidenz."""
    probs = torch.softmax(full_logits, dim=1)
    confidences, _ = torch.max(probs, dim=1)
    confidences = confidences.numpy()
    correct_conf = [confidences[i] for i in range(len(all_labels)) if all_labels[i] == all_preds[i]]
    wrong_conf = [confidences[i] for i in range(len(all_labels)) if all_labels[i] != all_preds[i]]
    plt.figure(figsize=(10, 6))
    sns.histplot(correct_conf, bins=30, color='green', label='Korrekt', kde=True, alpha=0.5)
    sns.histplot(wrong_conf, bins=30, color='red', label='Falsch', kde=True, alpha=0.5)
    plt.title("Konfidenz-Verteilung"); plt.xlabel("Konfidenz"); plt.legend()
    plt.savefig(out_path / "confidence_distribution.png"); plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='./word_landmarks_extracted')
    args = parser.parse_args()

    # 1. Setup & Split (Reproduktion des Trainings-Splits)
    with open('model/main_vocab.json', 'r') as f:
        vocab = json.load(f); word_to_idx = vocab['word_to_idx']
        idx_to_word = {int(k): v for k, v in vocab['idx_to_word'].items()}
    
    base_dataset = SignLanguageDataset(args.data_dir, debug=False, augment=False)
    old_to_new = {base_dataset.word_to_idx[w]: i for w, i in word_to_idx.items() if w in base_dataset.word_to_idx}
    filtered_indices = [i for i in range(len(base_dataset)) if base_dataset[i][1].item() in old_to_new]
    filtered_labels = [base_dataset.idx_to_word[base_dataset[i][1].item()] for i in filtered_indices]
    train_indices, val_indices = train_test_split(filtered_indices, test_size=0.2, random_state=42, stratify=filtered_labels)
    
    train_counts_raw = Counter([base_dataset[i][1].item() for i in train_indices])
    train_counts = {old_to_new[old_idx]: count for old_idx, count in train_counts_raw.items()}

    val_subset = RemappedDataset(base_dataset, val_indices, old_to_new)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=64, collate_fn=PadCollate(), shuffle=False)

    # 2. Inferenz
    config = MAIN_MODEL_CONFIG.copy(); sample_lms, _ = val_subset[0]
    config['input_size'] = sample_lms.shape[-1]
    model = load_model(args.model_path, config, len(word_to_idx))

    all_logits, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Inferenz"):
            landmarks, labels, mask = batch
            logits = model(landmarks.to(DEVICE), mask.to(DEVICE))
            all_logits.append(logits.cpu()); all_labels.extend(labels.tolist())
            all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())

    full_logits = torch.cat(all_logits)
    out = Path(f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M')}"); out.mkdir(exist_ok=True)

    # 3. AUSWERTUNGEN
    # A) Report
    report = classification_report(all_labels, all_preds, target_names=[idx_to_word[i] for i in range(len(idx_to_word))], output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose(); df_report.to_csv(out / "full_report.csv")

    # B) Top-K
    ks = range(1, 11)
    k_accs = [sum([1 for i in range(len(all_labels)) if all_labels[i] in torch.topk(full_logits[i], k=k).indices]) / len(all_labels) for k in ks]
    plt.figure(figsize=(8, 5)); plt.plot(ks, k_accs, 'b-o'); plt.title("Top-K Accuracy"); plt.savefig(out / "topk_curve.png"); plt.close()

    # C) Accuracy per Bucket (NEU)
    plot_accuracy_per_bucket(all_labels, all_preds, train_counts, out)

    # D) Confusion Matrix (Die 10 schwierigsten)
    worst_10 = df_report.drop(['accuracy', 'macro avg', 'weighted avg']).sort_values(by='f1-score').head(10).index.tolist()
    worst_indices = [word_to_idx[w] for w in worst_10]
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(idx_to_word)))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm[np.ix_(worst_indices, worst_indices)], annot=True, fmt='d', xticklabels=worst_10, yticklabels=worst_10, cmap='Blues')
    plt.title("Confusion Matrix: 10 schwierigste Klassen"); plt.savefig(out / "confusion_matrix_worst10.png"); plt.close()

    # E) Top 10 Verwechslungen (Textdatei)
    miscounts = [f"{idx_to_word[all_labels[i]]} -> {idx_to_word[all_preds[i]]}" for i in range(len(all_labels)) if all_labels[i] != all_preds[i]]
    with open(out / "misclassifications.txt", "w") as f:
        f.write("Häufigste Verwechslungen:\n")
        for pair, count in Counter(miscounts).most_common(10): f.write(f"{pair}: {count}\n")

    # F) Konfidenz
    plot_confidence_distribution(full_logits, all_labels, all_preds, out)
    print(f"\nErfolgreich abgeschlossen. Ergebnisse in: {out}")

if __name__ == "__main__":
    main()