import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from datetime import datetime
import json

# Import classes from your main training script
from SignNetWord import (
    TransformerSignClassifierWithHandedness,
    SignLanguageDataset,
    PadCollate,
    load_data_by_type,
    build_topk_vocabulary
)

from SignNetConfig import (
    MAIN_MODEL_CONFIG,
    EXPERT_MODEL_CONFIG,
    HIERARCHY_CONFIG
)

# ==================== CONFIGURATION ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description='SignNet Hierarchical Inference')
    parser.add_argument('--data-dir', type=str, default='./word_landmarks_extracted')
    parser.add_argument('--dataset-type', type=str, default='flat', choices=['flat', 'split'])
    parser.add_argument('--main-model-path', type=str, default='./models/sign_classifier_final_enhanced.pth')
    parser.add_argument('--direction-expert-path', type=str, default='./models/direction_expert.pth')
    parser.add_argument('--kommen-expert-path', type=str, default='./models/kommen_expert.pth')
    parser.add_argument('--weather-expert-path', type=str, default='./models/weather_expert.pth')
    parser.add_argument('--main-model-uri', type=str, default='models:/Production/1')
    parser.add_argument('--direction-expert-uri', type=str, default='models:/SignClassifier_DirectionExpert/1')
    parser.add_argument('--training-run-id', type=str, required=False, help='MLflow run ID from training that contains val_indices.npy')
    parser.add_argument('--main-only', action='store_true', help='Run inference using main model only (skip experts)')
    return parser.parse_args()

class HierarchicalClassifier(nn.Module):
    def __init__(self, root_model, expert_models, hierarchy_config, root_idx_to_word, expert_dicts):
        super().__init__()
        self.root_model = root_model
        self.expert_models = nn.ModuleDict(expert_models)
        self.config = hierarchy_config
        self.root_idx_to_word = root_idx_to_word
        self.expert_dicts = expert_dicts

        self.class_to_expert = {}
        for expert_name, classes in self.config.items():
            for cls in classes:
                self.class_to_expert[cls] = expert_name

    def forward(self, landmarks, padding_mask=None):
        root_logits, _ = self.root_model(landmarks, padding_mask)
        root_probs = torch.softmax(root_logits, dim=1)
        
        final_preds = []
        used_expert = []

        # CHANGE: Analyze Top-5 to calculate "Group Probabilities"
        K_check = 5
        topk_conf, topk_idx = torch.topk(root_probs, k=K_check, dim=1)

        for i in range(len(landmarks)):
            # Default: Top-1 from Main Model
            best_main_idx = topk_idx[i, 0].item()
            pred_class_name = self.root_idx_to_word[best_main_idx]
            
            # 1. Calculate Sum of Probabilities for each Expert Group
            expert_group_probs = {}
            max_non_expert_prob = 0.0
            
            for k in range(K_check):
                prob = topk_conf[i, k].item()
                idx = topk_idx[i, k].item()
                name = self.root_idx_to_word[idx]
                
                # Check if this class belongs to an active expert
                is_expert_class = False
                if name in self.class_to_expert:
                    exp_name = self.class_to_expert[name]
                    if exp_name in self.expert_models:
                        expert_group_probs[exp_name] = expert_group_probs.get(exp_name, 0.0) + prob
                        is_expert_class = True
                
                # If not an expert class, update the "Best Alternative" score
                if not is_expert_class:
                    if prob > max_non_expert_prob:
                        max_non_expert_prob = prob

            # 2. Decide Trigger: Is the Group Prob > Best Non-Expert Prob?
            candidate_expert = None
            best_group_prob = -1.0
            
            for exp_name, group_prob in expert_group_probs.items():
                # STRICTER: Group must beat alternatives by at least 10% margin AND have 25% mass
                if group_prob > 0.25 and group_prob > (max_non_expert_prob + 0.10):
                    if group_prob > best_group_prob:
                        best_group_prob = group_prob
                        candidate_expert = exp_name
            
            # 3. Execute with Confidence-Based Arbitration
            if candidate_expert:
                expert_model = self.expert_models[candidate_expert]
                expert_dict = self.expert_dicts[candidate_expert]

                expert_logits, _ = expert_model(landmarks[i:i+1], padding_mask[i:i+1] if padding_mask is not None else None)
                expert_prob = torch.softmax(expert_logits, dim=1)
                
                # Get expert prediction
                e_conf, e_pred_idx = torch.max(expert_prob, dim=1)
                expert_pred_name = expert_dict['idx_to_word'][e_pred_idx.item()]
                
                # CHANGE: Confidence-weighted decision
                # Compare expert confidence vs. main model Top-1 confidence
                main_top1_conf = topk_conf[i, 0].item()
                expert_conf = e_conf.item()
                
                # Use expert ONLY if it's more confident than main model
                # This prevents weak expert guesses from overriding strong main predictions
                if expert_conf > main_top1_conf:
                    final_preds.append(expert_pred_name)
                    used_expert.append(candidate_expert)
                else:
                    # Main model was more confident, trust it
                    final_preds.append(pred_class_name)
                    used_expert.append(f"{candidate_expert} (Main>Expert)")
            else:
                final_preds.append(pred_class_name)
                used_expert.append("None")

        return final_preds, used_expert

def load_model(model_class, path_or_uri, config, num_classes, device):
    print(f"Loading model from {path_or_uri}...")

    # Initialize architecture
    model = model_class(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_classes=num_classes,
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dim_feedforward=config['dim_feedforward']
    )

    # Check if it's an MLflow URI (starts with 'models:/')
    if path_or_uri.startswith("models:/"):
        try:
            # Download the state_dict artifact from MLflow
            # This assumes you logged the model as a PyTorch model or saved state_dict as artifact
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=path_or_uri)

            # If downloaded path is a directory (standard mlflow model format), find the state_dict
            if os.path.isdir(local_path):
                # Look for common state dict filenames
                possible_files = ["state_dict.pth", "checkpoint.pth", "data/model.pth"]
                found = False
                for f in possible_files:
                    full_p = os.path.join(local_path, f)
                    if os.path.exists(full_p):
                        checkpoint = torch.load(full_p, map_location=device)
                        found = True
                        break
                if not found:
                    # Fallback: try loading using mlflow.pytorch.load_model (returns full object)
                    # This might be safer if you logged using mlflow.pytorch.log_model
                    loaded_model = mlflow.pytorch.load_model(path_or_uri, map_location=device)
                    # Copy weights to our initialized architecture (safer than using loaded object directly if arch differs)
                    model.load_state_dict(loaded_model.state_dict())
                    model.to(device)
                    model.eval()
                    return model
            else:
                # It's a direct file path downloaded
                checkpoint = torch.load(local_path, map_location=device)

        except Exception as e:
            print(f"[ERROR] Failed to load from MLflow: {e}")
            raise e
    else:
        # Standard local file load
        checkpoint = torch.load(path_or_uri, map_location=device)

    # Process state_dict (handle 'module.' prefix etc)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict() # If it's a model object

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()

    # ---------------- MLFLOW SETUP ----------------
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'roman'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'SignNet'
    mlflow.set_tracking_uri("https://mlflow.schlaepfer.me")
    mlflow.set_experiment("Hierarchical_Evaluation")

    timestamp = datetime.now().strftime('%m%d_%H%M')
    run_name = f"{'MainModel' if args.main_only else 'Hierarchical'}_Eval_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        # Log Parameters
        mlflow.log_params({
            "main_model": Path(args.main_model_path).name,
            "main_only": args.main_only,
            "dataset_type": args.dataset_type,
            "hierarchy_config": "None" if args.main_only else str(HIERARCHY_CONFIG)
        })

        # 1. Load Vocabulary
        if args.training_run_id:
            # Load vocabulary from MLflow
            print("Downloading main vocabulary from MLflow...")
            main_vocab_path = mlflow.artifacts.download_artifacts(
                run_id=args.training_run_id,
                artifact_path="main_vocab.json"
            )
            
            with open(main_vocab_path, 'r') as f:
                vocab_data = json.load(f)
            
            main_word_to_idx = vocab_data['word_to_idx']
            main_idx_to_word = {int(k): v for k, v in vocab_data['idx_to_word'].items()}
            num_classes_main = vocab_data['num_classes']
            
            print(f"Main Vocabulary (from MLflow): {num_classes_main} classes")
        else:
            # Fallback: Build vocabulary from files (old method)
            vocab_filename = 'main_vocab.json'
            with open(vocab_filename, 'r') as f:
                vocab_dict = json.load(f)

            main_word_to_idx = vocab_dict['word_to_idx']
            main_idx_to_word = {int(k): v for k, v in vocab_dict['idx_to_word'].items()}
            num_classes_main = vocab_dict['num_classes']

            print(f"Loaded vocabulary with {num_classes_main} classes")

        # 2. Load Models
        main_model = load_model(TransformerSignClassifierWithHandedness, args.main_model_path, MAIN_MODEL_CONFIG, num_classes_main, DEVICE)

        # Skip expert loading if --main-only
        if args.main_only:
            print("\n[INFO] Running in MAIN MODEL ONLY mode. Skipping expert models.")
            expert_models = {}
            expert_dicts = {}
            hierarchical_model = None
        else:
            expert_models = {}
            expert_dicts = {}

            print("\nLoading Expert Models...")
            for expert_name, expert_classes in HIERARCHY_CONFIG.items():
                # 1. Resolve Path
                arg_attr = f"{expert_name}_path"
                if hasattr(args, arg_attr):
                    model_path = getattr(args, arg_attr)
                else:
                    model_path = f"./models/{expert_name}.pth"

                if not os.path.exists(model_path):
                    continue

                # 2. Resolve Vocabulary
                vocab_path = Path(model_path).with_name(f"{Path(model_path).stem}_vocab.json")
                
                if vocab_path.exists():
                    print(f"  [i] Found vocab file for {expert_name}: {vocab_path.name}")
                    with open(vocab_path, 'r') as f:
                        vocab_data = json.load(f)
                    
                    if 'idx_to_word' in vocab_data:
                        expert_idx_to_word = {int(k): v for k, v in vocab_data['idx_to_word'].items()}
                        expert_word_to_idx = vocab_data['word_to_idx']
                    else:
                        expert_word_to_idx = vocab_data
                        expert_idx_to_word = {v: k for k, v in expert_word_to_idx.items()}
                else:
                    print(f"  [!] No vocab file found for {expert_name}. Falling back to sorted config classes.")
                    vocab_list = sorted(expert_classes)
                    expert_word_to_idx = {w: i for i, w in enumerate(vocab_list)}
                    expert_idx_to_word = {i: w for i, w in enumerate(vocab_list)}

                # 3. Load Model
                num_classes_expert = len(expert_idx_to_word)
                expert_dicts[expert_name] = {
                    'word_to_idx': expert_word_to_idx,
                    'idx_to_word': expert_idx_to_word
                }
                
                try:
                    expert_models[expert_name] = load_model(
                        TransformerSignClassifierWithHandedness, 
                        model_path, 
                        EXPERT_MODEL_CONFIG, 
                        num_classes_expert, 
                        DEVICE
                    )
                    print(f"  [+] Loaded {expert_name} ({num_classes_expert} classes)")
                except Exception as e:
                    print(f"  [x] Failed to load {expert_name}: {e}")

            hierarchical_model = HierarchicalClassifier(main_model, expert_models, HIERARCHY_CONFIG, main_idx_to_word, expert_dicts)

        # 3. Load Data
        print("Loading validation indices...")
        artifact_path = "val_indices.npy"
        val_indices = np.load(artifact_path)
        print(f"Loaded {len(val_indices)} validation samples")

        if args.dataset_type == 'flat':
            data_root = args.data_dir
        else:
            data_root = os.path.join(args.data_dir, "train")

        base_dataset = SignLanguageDataset(data_root, debug=False)

        old_to_new_idx = {}
        for word, model_idx in main_word_to_idx.items():
            if word in base_dataset.word_to_idx:
                dataset_original_idx = base_dataset.word_to_idx[word]
                old_to_new_idx[dataset_original_idx] = model_idx

        from SignNetWord import RemappedDataset
        val_subset = RemappedDataset(base_dataset, val_indices.tolist(), old_to_new_idx)
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=64,
            collate_fn=PadCollate(),
            shuffle=False
        )

        # 4. Run Evaluation
        if args.main_only:
            # ========== MAIN MODEL ONLY EVALUATION ==========
            print("\nStarting Main Model Evaluation...")
            all_preds = []
            all_labels = []
            all_confidences = []

            with torch.no_grad():
                for batch in tqdm(val_loader):
                    landmarks, labels, _, padding_mask = batch
                    landmarks = landmarks.to(DEVICE)

                    logits, _ = main_model(landmarks, padding_mask)
                    probs = torch.softmax(logits, dim=1)
                    conf, preds_idx = torch.max(probs, dim=1)

                    for i in range(len(landmarks)):
                        gt_idx = labels[i].item()
                        gt_name = main_idx_to_word[gt_idx]
                        all_labels.append(gt_name)

                        pred_idx = preds_idx[i].item()
                        pred_name = main_idx_to_word[pred_idx]
                        all_preds.append(pred_name)
                        all_confidences.append(conf[i].item())

            # Compute Metrics
            accuracy = accuracy_score(all_labels, all_preds)
            avg_confidence = np.mean(all_confidences)

            # Top-K Accuracy (recompute with full logits)
            print("\nComputing Top-K Accuracy...")
            top1_correct = 0
            top3_correct = 0
            top5_correct = 0
            total = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Top-K"):
                    landmarks, labels, _, padding_mask = batch
                    landmarks = landmarks.to(DEVICE)

                    logits, _ = main_model(landmarks, padding_mask)
                    
                    for i in range(len(landmarks)):
                        gt_idx = labels[i].item()
                        _, topk_indices = torch.topk(logits[i], k=5)
                        topk_list = topk_indices.tolist()

                        total += 1
                        if gt_idx == topk_list[0]:
                            top1_correct += 1
                        if gt_idx in topk_list[:3]:
                            top3_correct += 1
                        if gt_idx in topk_list[:5]:
                            top5_correct += 1

            top1_acc = top1_correct / total
            top3_acc = top3_correct / total
            top5_acc = top5_correct / total

            # Log to MLflow
            mlflow.log_metrics({
                "accuracy": accuracy,
                "top1_accuracy": top1_acc,
                "top3_accuracy": top3_acc,
                "top5_accuracy": top5_acc,
                "avg_confidence": avg_confidence
            })

            # Print Results
            print(f"\n{'='*60}")
            print(f"MAIN MODEL RESULTS")
            print(f"{'='*60}")
            print(f"Top-1 Accuracy:    {top1_acc:.4%}")
            print(f"Top-3 Accuracy:    {top3_acc:.4%}")
            print(f"Top-5 Accuracy:    {top5_acc:.4%}")
            print(f"Avg Confidence:    {avg_confidence:.4f}")
            print(f"{'='*60}")

            # Classification Report (optional, can be verbose)
            print("\nClassification Report (Top-20 classes):")
            report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
            
            # Sort by support and show top 20
            class_metrics = [(k, v) for k, v in report.items() if k not in ['accuracy', 'macro avg', 'weighted avg']]
            class_metrics.sort(key=lambda x: x[1]['support'], reverse=True)
            
            print(f"{'Class':<20} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'Support':<10}")
            print("-" * 65)
            for cls_name, metrics in class_metrics[:20]:
                print(f"{cls_name:<20} | {metrics['precision']:.4f}     | {metrics['recall']:.4f}     | {metrics['f1-score']:.4f}     | {int(metrics['support'])}")

            # ========== CONFUSION MATRIX FOR 50 WORST CLASSES ==========
            print("\n" + "="*60)
            print("CONFUSION MATRIX: 50 WORST PERFORMING CLASSES")
            print("="*60)
            
            # Sort by F1-score (ascending) to find worst classes, filter out classes with 0 support
            worst_classes = [(k, v) for k, v in report.items() 
                            if k not in ['accuracy', 'macro avg', 'weighted avg'] and v['support'] > 0]
            worst_classes.sort(key=lambda x: x[1]['f1-score'])
            worst_50_names = [cls_name for cls_name, _ in worst_classes[:50]]
            
            print(f"Worst 50 classes (by F1-score): {worst_50_names}")
            
            # Filter predictions and labels to only include worst 50 classes
            filtered_labels = []
            filtered_preds = []
            for gt, pred in zip(all_labels, all_preds):
                if gt in worst_50_names:
                    filtered_labels.append(gt)
                    filtered_preds.append(pred if pred in worst_50_names else "OTHER")
            
            # Add "OTHER" to the label list if any predictions fell outside worst 50
            worst_50_with_other = worst_50_names + (["OTHER"] if "OTHER" in filtered_preds else [])
            
            # Compute confusion matrix
            cm = confusion_matrix(filtered_labels, filtered_preds, labels=worst_50_with_other)
            
            # Plot confusion matrix
            plt.figure(figsize=(20, 18))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={'size': 6},
                       xticklabels=worst_50_with_other,
                       yticklabels=worst_50_names)
            plt.title('Confusion Matrix: 50 Worst Performing Classes (by F1-score)')
            plt.xlabel('Predicted')
            plt.ylabel('True Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save and log to MLflow
            cm_path = f"confusion_matrix_worst50_{timestamp}.png"
            plt.savefig(cm_path, dpi=200, bbox_inches='tight')
            mlflow.log_artifact(cm_path)
            plt.close()
            print(f"Confusion matrix saved to: {cm_path}")
            
            # Print text version of confusion matrix (most confused pairs)
            print("\nMost Common Misclassifications (True -> Predicted: Count):")
            print("-" * 50)
            misclassifications = []
            for i, true_label in enumerate(worst_50_names):
                for j, pred_label in enumerate(worst_50_with_other):
                    if true_label != pred_label and cm[i, j] > 0:
                        misclassifications.append((true_label, pred_label, cm[i, j]))
            
            misclassifications.sort(key=lambda x: x[2], reverse=True)
            for true_lbl, pred_lbl, count in misclassifications[:25]:
                print(f"  {true_lbl:<20} -> {pred_lbl:<20}: {count}")

            # ========== FULL CONFUSION MATRIX (ALL CLASSES) ==========
            print("\n" + "="*60)
            print("CONFUSION MATRIX: ALL CLASSES")
            print("="*60)
            
            # Get all unique class names sorted by support (descending)
            all_class_names = [cls_name for cls_name, _ in class_metrics]
            num_classes = len(all_class_names)
            print(f"Total classes: {num_classes}")
            
            # Compute full confusion matrix
            cm_full = confusion_matrix(all_labels, all_preds, labels=all_class_names)
            
            # Plot full confusion matrix (no annotations due to size)
            plt.figure(figsize=(30, 28))
            sns.heatmap(cm_full, cmap='Blues', cbar=True,
                       xticklabels=all_class_names,
                       yticklabels=all_class_names)
            plt.title(f'Full Confusion Matrix: All {num_classes} Classes')
            plt.xlabel('Predicted')
            plt.ylabel('True Label')
            plt.xticks(rotation=90, ha='center', fontsize=4)
            plt.yticks(rotation=0, fontsize=4)
            plt.tight_layout()
            
            # Save and log to MLflow
            cm_full_path = f"confusion_matrix_full_{timestamp}.png"
            plt.savefig(cm_full_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(cm_full_path)
            plt.close()
            print(f"Full confusion matrix saved to: {cm_full_path}")
            
            # Also save as numpy array for later analysis
            cm_full_npy_path = f"confusion_matrix_full_{timestamp}.npy"
            np.save(cm_full_npy_path, cm_full)
            mlflow.log_artifact(cm_full_npy_path)
            
            # Save class names mapping
            cm_classes_path = f"confusion_matrix_classes_{timestamp}.json"
            with open(cm_classes_path, 'w') as f:
                json.dump({'classes': all_class_names, 'num_classes': num_classes}, f)
            mlflow.log_artifact(cm_classes_path)
            print(f"Class mapping saved to: {cm_classes_path}")
            
            # Print overall misclassification statistics
            print("\nTop 30 Most Common Misclassifications (All Classes):")
            print("-" * 60)
            all_misclassifications = []
            for i, true_label in enumerate(all_class_names):
                for j, pred_label in enumerate(all_class_names):
                    if true_label != pred_label and cm_full[i, j] > 0:
                        all_misclassifications.append((true_label, pred_label, cm_full[i, j]))
            
            all_misclassifications.sort(key=lambda x: x[2], reverse=True)
            for true_lbl, pred_lbl, count in all_misclassifications[:30]:
                print(f"  {true_lbl:<25} -> {pred_lbl:<25}: {count}")

        else:
            # ========== HIERARCHICAL EVALUATION (Original Code) ==========
            print("\nStarting Hierarchical Evaluation with Diagnostics...")
            all_preds_hierarchical = []
            all_preds_baseline = []
            all_labels = []
            expert_usage_stats = Counter()

            oracle_stats = {name: {'total': 0, 'main_correct': 0, 'expert_standalone_correct': 0, 'routed_correctly': 0} 
                           for name in HIERARCHY_CONFIG.keys()}

            with torch.no_grad():
                for batch in tqdm(val_loader):
                    landmarks, labels, _, padding_mask = batch
                    landmarks = landmarks.to(DEVICE)

                    logits, _ = main_model(landmarks, padding_mask)
                    base_probs = torch.softmax(logits, dim=1)
                    base_conf, base_preds_idx = torch.max(base_probs, dim=1)

                    hier_preds, hier_experts_used = hierarchical_model(landmarks, padding_mask)

                    for i in range(len(landmarks)):
                        gt_idx = labels[i].item()
                        gt_name = main_idx_to_word[gt_idx]
                        all_labels.append(gt_name)

                        pred_idx = base_preds_idx[i].item()
                        pred_name = main_idx_to_word[pred_idx]
                        all_preds_baseline.append(pred_name)

                        final_pred = hier_preds[i]
                        expert_used = hier_experts_used[i]
                        
                        all_preds_hierarchical.append(final_pred)
                        expert_usage_stats[expert_used] += 1

                        if gt_name in hierarchical_model.class_to_expert:
                            target_expert = hierarchical_model.class_to_expert[gt_name]
                            stats = oracle_stats[target_expert]
                            stats['total'] += 1

                            if pred_name == gt_name:
                                stats['main_correct'] += 1

                            if expert_used == target_expert:
                                stats['routed_correctly'] += 1

                            if target_expert in hierarchical_model.expert_models:
                                exp_model = hierarchical_model.expert_models[target_expert]
                                exp_dict = hierarchical_model.expert_dicts[target_expert]
                                
                                e_logits, _ = exp_model(landmarks[i:i+1], padding_mask[i:i+1] if padding_mask is not None else None)
                                e_pred_idx = torch.argmax(e_logits, dim=1)
                                oracle_pred_name = exp_dict['idx_to_word'][e_pred_idx.item()]
                                
                                if oracle_pred_name == gt_name:
                                    stats['expert_standalone_correct'] += 1

            acc_base = accuracy_score(all_labels, all_preds_baseline)
            acc_hier = accuracy_score(all_labels, all_preds_hierarchical)
            net_improvement = acc_hier - acc_base

            mlflow.log_metric("accuracy_baseline", acc_base)
            mlflow.log_metric("accuracy_hierarchical", acc_hier)
            mlflow.log_metric("net_improvement", net_improvement)

            print(f"\n{'='*60}")
            print(f"OVERALL RESULTS")
            print(f"{'='*60}")
            print(f"Baseline Accuracy:     {acc_base:.4%}")
            print(f"Hierarchical Accuracy: {acc_hier:.4%}")
            print(f"Net Improvement:       {net_improvement:+.4%}")

            print(f"\n{'='*60}")
            print(f"ROOT CAUSE ANALYSIS (Oracle Diagnostics)")
            print(f"{'='*60}")
            print(f"{'Expert Name':<20} | {'Samples':<8} | {'Main Acc':<10} | {'Expert Acc':<10} | {'Routing Acc':<10} | {'Potential Gain':<10}")
            print("-" * 85)

            for expert_name, stats in oracle_stats.items():
                total = stats['total']
                if total == 0:
                    print(f"{expert_name:<20} | {total:<8} | N/A")
                    continue

                main_acc = stats['main_correct'] / total
                expert_acc = stats['expert_standalone_correct'] / total
                routing_acc = stats['routed_correctly'] / total
                gain = expert_acc - main_acc

                mlflow.log_metric(f"{expert_name}_main_acc", main_acc)
                mlflow.log_metric(f"{expert_name}_expert_acc", expert_acc)
                mlflow.log_metric(f"{expert_name}_routing_acc", routing_acc)
                mlflow.log_metric(f"{expert_name}_potential_gain", gain)

                print(f"{expert_name:<20} | {total:<8} | {main_acc:.2%}     | {expert_acc:.2%}     | {routing_acc:.2%}     | {gain:+.2%}")

            print("-" * 85)

            # ========== CONFUSION MATRIX FOR 50 WORST CLASSES (HIERARCHICAL) ==========
            print("\n" + "="*60)
            print("CONFUSION MATRIX: 50 WORST PERFORMING CLASSES (Hierarchical)")
            print("="*60)
            
            # Get classification report for hierarchical predictions
            report_hier = classification_report(all_labels, all_preds_hierarchical, output_dict=True, zero_division=0)
            
            # Sort by F1-score (ascending) to find worst classes
            worst_classes = [(k, v) for k, v in report_hier.items() 
                            if k not in ['accuracy', 'macro avg', 'weighted avg'] and v['support'] > 0]
            worst_classes.sort(key=lambda x: x[1]['f1-score'])
            worst_50_names = [cls_name for cls_name, _ in worst_classes[:50]]
            
            print(f"Worst 50 classes (by F1-score): {worst_50_names}")
            
            # Filter predictions and labels to only include worst 50 classes
            filtered_labels = []
            filtered_preds = []
            for gt, pred in zip(all_labels, all_preds_hierarchical):
                if gt in worst_50_names:
                    filtered_labels.append(gt)
                    filtered_preds.append(pred if pred in worst_50_names else "OTHER")
            
            # Add "OTHER" to the label list if any predictions fell outside worst 50
            worst_50_with_other = worst_50_names + (["OTHER"] if "OTHER" in filtered_preds else [])
            
            # Compute confusion matrix
            cm = confusion_matrix(filtered_labels, filtered_preds, labels=worst_50_with_other)
            
            # Plot confusion matrix
            plt.figure(figsize=(20, 18))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', annot_kws={'size': 6},
                       xticklabels=worst_50_with_other,
                       yticklabels=worst_50_names)
            plt.title('Confusion Matrix: 50 Worst Performing Classes - Hierarchical (by F1-score)')
            plt.xlabel('Predicted')
            plt.ylabel('True Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save and log to MLflow
            cm_path = f"confusion_matrix_worst50_hierarchical_{timestamp}.png"
            plt.savefig(cm_path, dpi=200, bbox_inches='tight')
            mlflow.log_artifact(cm_path)
            plt.close()
            print(f"Confusion matrix saved to: {cm_path}")
            
            # Print text version of confusion matrix (most confused pairs)
            print("\nMost Common Misclassifications (True -> Predicted: Count):")
            print("-" * 50)
            misclassifications = []
            for i, true_label in enumerate(worst_50_names):
                for j, pred_label in enumerate(worst_50_with_other):
                    if true_label != pred_label and cm[i, j] > 0:
                        misclassifications.append((true_label, pred_label, cm[i, j]))
            
            misclassifications.sort(key=lambda x: x[2], reverse=True)
            for true_lbl, pred_lbl, count in misclassifications[:25]:
                print(f"  {true_lbl:<20} -> {pred_lbl:<20}: {count}")

            # ========== FULL CONFUSION MATRIX (ALL CLASSES) - HIERARCHICAL ==========
            print("\n" + "="*60)
            print("CONFUSION MATRIX: ALL CLASSES (Hierarchical)")
            print("="*60)
            
            # Get all unique class names sorted by support (descending)
            class_metrics_hier = [(k, v) for k, v in report_hier.items() if k not in ['accuracy', 'macro avg', 'weighted avg']]
            class_metrics_hier.sort(key=lambda x: x[1]['support'], reverse=True)
            all_class_names = [cls_name for cls_name, _ in class_metrics_hier]
            num_classes = len(all_class_names)
            print(f"Total classes: {num_classes}")
            
            # Compute full confusion matrix
            cm_full = confusion_matrix(all_labels, all_preds_hierarchical, labels=all_class_names)
            
            # Plot full confusion matrix (no annotations due to size)
            plt.figure(figsize=(30, 28))
            sns.heatmap(cm_full, cmap='Reds', cbar=True,
                       xticklabels=all_class_names,
                       yticklabels=all_class_names)
            plt.title(f'Full Confusion Matrix: All {num_classes} Classes - Hierarchical')
            plt.xlabel('Predicted')
            plt.ylabel('True Label')
            plt.xticks(rotation=90, ha='center', fontsize=4)
            plt.yticks(rotation=0, fontsize=4)
            plt.tight_layout()
            
            # Save and log to MLflow
            cm_full_path = f"confusion_matrix_full_hierarchical_{timestamp}.png"
            plt.savefig(cm_full_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(cm_full_path)
            plt.close()
            print(f"Full confusion matrix saved to: {cm_full_path}")
            
            # Also save as numpy array for later analysis
            cm_full_npy_path = f"confusion_matrix_full_hierarchical_{timestamp}.npy"
            np.save(cm_full_npy_path, cm_full)
            mlflow.log_artifact(cm_full_npy_path)
            
            # Save class names mapping
            cm_classes_path = f"confusion_matrix_classes_hierarchical_{timestamp}.json"
            with open(cm_classes_path, 'w') as f:
                json.dump({'classes': all_class_names, 'num_classes': num_classes}, f)
            mlflow.log_artifact(cm_classes_path)
            print(f"Class mapping saved to: {cm_classes_path}")
            
            # Print overall misclassification statistics
            print("\nTop 30 Most Common Misclassifications (All Classes):") 
            print("-" * 60)
            all_misclassifications = []
            for i, true_label in enumerate(all_class_names):
                for j, pred_label in enumerate(all_class_names):
                    if true_label != pred_label and cm_full[i, j] > 0:
                        all_misclassifications.append((true_label, pred_label, cm_full[i, j]))
            
            all_misclassifications.sort(key=lambda x: x[2], reverse=True)
            for true_lbl, pred_lbl, count in all_misclassifications[:30]:
                print(f"  {true_lbl:<25} -> {pred_lbl:<25}: {count}")


if __name__ == "__main__":
    main()
