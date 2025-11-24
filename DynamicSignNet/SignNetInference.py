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
        
        # CHANGE: Get Top-K predictions (e.g., Top 3)
        K = 3
        topk_conf, topk_idx = torch.topk(root_probs, k=K, dim=1)

        final_preds = []
        used_expert = []

        for i in range(len(landmarks)):
            # Default to Top-1
            best_main_idx = topk_idx[i, 0].item()
            pred_class_name = self.root_idx_to_word[best_main_idx]
            
            candidate_expert = None
            
            # Check if ANY of the Top-K predictions trigger an expert
            for k in range(K):
                curr_idx = topk_idx[i, k].item()
                curr_name = self.root_idx_to_word[curr_idx]
                
                if curr_name in self.class_to_expert:
                    expert_name = self.class_to_expert[curr_name]
                    if expert_name in self.expert_models:
                        candidate_expert = expert_name
                        break # Found an expert, stop looking
            
            # If an expert was found in Top-K, use it
            if candidate_expert:
                expert_model = self.expert_models[candidate_expert]
                expert_dict = self.expert_dicts[candidate_expert]

                expert_logits, _ = expert_model(landmarks[i:i+1], padding_mask[i:i+1] if padding_mask is not None else None)
                expert_prob = torch.softmax(expert_logits, dim=1)
                
                # Get expert prediction
                e_conf, e_pred_idx = torch.max(expert_prob, dim=1)
                expert_pred_name = expert_dict['idx_to_word'][e_pred_idx.item()]
                
                # OPTIONAL: Safety Check
                # Only use expert if it is confident enough (e.g. > 50%)
                # otherwise fall back to main model's Top-1
                if e_conf.item() > 0.4: 
                    final_preds.append(expert_pred_name)
                    used_expert.append(candidate_expert)
                else:
                    final_preds.append(pred_class_name)
                    used_expert.append(f"{candidate_expert} (Low Conf)")
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
    run_name = f"Hierarchical_Eval_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        # Log Parameters
        mlflow.log_params({
            "main_model": Path(args.main_model_path).name,
            "direction_expert": Path(args.direction_expert_path).name,
            "dataset_type": args.dataset_type,
            "hierarchy_config": str(HIERARCHY_CONFIG)
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
            # Load from file
            with open(vocab_filename, 'r') as f:
                vocab_dict = json.load(f)

            # Extract components
            main_word_to_idx = vocab_dict['word_to_idx']
            main_idx_to_word = {int(k): v for k, v in vocab_dict['idx_to_word'].items()}
            num_classes_main = vocab_dict['num_classes']

            print(f"Loaded vocabulary with {num_classes_main} classes")


        # 2. Load Models
        main_model = load_model(TransformerSignClassifierWithHandedness, args.main_model_path, MAIN_MODEL_CONFIG, num_classes_main, DEVICE)

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
                # Skip if model file doesn't exist
                continue

            # 2. Resolve Vocabulary (CRITICAL FIX)
            # Try to find a vocab file: {model_name}_vocab.json
            vocab_path = Path(model_path).with_name(f"{Path(model_path).stem}_vocab.json")
            
            if vocab_path.exists():
                print(f"  [i] Found vocab file for {expert_name}: {vocab_path.name}")
                with open(vocab_path, 'r') as f:
                    vocab_data = json.load(f)
                
                # Handle standard vocab format
                if 'idx_to_word' in vocab_data:
                    expert_idx_to_word = {int(k): v for k, v in vocab_data['idx_to_word'].items()}
                    expert_word_to_idx = vocab_data['word_to_idx']
                else:
                    # Fallback if it's just a simple dict
                    expert_word_to_idx = vocab_data
                    expert_idx_to_word = {v: k for k, v in expert_word_to_idx.items()}
            else:
                print(f"  [!] No vocab file found for {expert_name}. Falling back to sorted config classes (Risk of mismatch!).")
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
        print("Downloading validation indices from MLflow...")
        # artifact_path = mlflow.artifacts.download_artifacts(
        #     run_id=args.training_run_id,
        #     artifact_path="val_indices.npy"
        # )
        artifact_path = "val_indices.npy"  # Corrected to just the filename
        val_indices = np.load(artifact_path)
        print(f"Loaded {len(val_indices)} validation samples")

        # Rest of the code remains the same
        if args.dataset_type == 'flat':
            data_root = args.data_dir
        else:
            data_root = os.path.join(args.data_dir, "train")

        base_dataset = SignLanguageDataset(data_root, debug=False)

        # Build mapping old_label_idx (Dataset) -> new_label_idx (Model/Vocab)
        old_to_new_idx = {}
        
        # Iterate directly over the loaded vocabulary dict to get the TRUE model indices
        for word, model_idx in main_word_to_idx.items():
            # Check if this word exists in the raw dataset
            if word in base_dataset.word_to_idx:
                dataset_original_idx = base_dataset.word_to_idx[word]
                
                # Map: Dataset ID -> Model ID
                old_to_new_idx[dataset_original_idx] = model_idx


        from SignNetWord import RemappedDataset
        val_subset = RemappedDataset(base_dataset, val_indices.tolist(), old_to_new_idx)
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=64,
            collate_fn=PadCollate(),
            shuffle=False
        )

        # 4. Run Evaluation with ORACLE DIAGNOSTICS
        print("\nStarting Hierarchical Evaluation with Diagnostics...")
        all_preds_hierarchical = []
        all_preds_baseline = []
        all_labels = []
        expert_usage_stats = Counter()

        # Storage for diagnostic stats
        # Structure: {expert_name: {'total': 0, 'main_correct': 0, 'expert_standalone_correct': 0, 'routed_correctly': 0}}
        oracle_stats = {name: {'total': 0, 'main_correct': 0, 'expert_standalone_correct': 0, 'routed_correctly': 0} 
                       for name in HIERARCHY_CONFIG.keys()}

        with torch.no_grad():
            for batch in tqdm(val_loader):
                landmarks, labels, _, padding_mask = batch
                landmarks = landmarks.to(DEVICE)

                # 1. Run Main Model (for baseline metrics)
                logits, _ = main_model(landmarks, padding_mask)
                base_probs = torch.softmax(logits, dim=1)
                base_conf, base_preds_idx = torch.max(base_probs, dim=1)

                # 2. Run Hierarchical Model (The Real System)
                # This uses the Top-K logic defined in HierarchicalClassifier.forward()
                hier_preds, hier_experts_used = hierarchical_model(landmarks, padding_mask)

                # Process batch item by item for granular analysis
                for i in range(len(landmarks)):
                    # --- Ground Truth ---
                    gt_idx = labels[i].item()
                    gt_name = main_idx_to_word[gt_idx]
                    all_labels.append(gt_name)

                    # --- Main Model Prediction ---
                    pred_idx = base_preds_idx[i].item()
                    pred_name = main_idx_to_word[pred_idx]
                    all_preds_baseline.append(pred_name)

                    # --- Hierarchical Prediction ---
                    final_pred = hier_preds[i]
                    expert_used = hier_experts_used[i]
                    
                    all_preds_hierarchical.append(final_pred)
                    expert_usage_stats[expert_used] += 1

                    # --- ORACLE DIAGNOSTICS (The "What If" Analysis) ---
                    # We analyze samples that ACTUALLY belong to an expert domain
                    if gt_name in hierarchical_model.class_to_expert:
                        target_expert = hierarchical_model.class_to_expert[gt_name]
                        stats = oracle_stats[target_expert]
                        stats['total'] += 1

                        # 1. Did Main Model get it right?
                        if pred_name == gt_name:
                            stats['main_correct'] += 1

                        # 2. Was it routed correctly? 
                        # Check if the system actually decided to use the correct expert
                        # (Matches if expert was triggered AND confident enough)
                        if expert_used == target_expert:
                            stats['routed_correctly'] += 1

                        # 3. Expert Standalone Accuracy (Oracle)
                        # Force run the correct expert regardless of main model prediction
                        if target_expert in hierarchical_model.expert_models:
                            exp_model = hierarchical_model.expert_models[target_expert]
                            exp_dict = hierarchical_model.expert_dicts[target_expert]
                            
                            e_logits, _ = exp_model(landmarks[i:i+1], padding_mask[i:i+1] if padding_mask is not None else None)
                            e_pred_idx = torch.argmax(e_logits, dim=1)
                            oracle_pred_name = exp_dict['idx_to_word'][e_pred_idx.item()]
                            
                            if oracle_pred_name == gt_name:
                                stats['expert_standalone_correct'] += 1

        # 5. Metrics & Logging
        acc_base = accuracy_score(all_labels, all_preds_baseline)
        acc_hier = accuracy_score(all_labels, all_preds_hierarchical)
        net_improvement = acc_hier - acc_base

        # Log overall metrics to MLflow
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
            
            # "Potential Gain" is how much better the expert is than the main model
            # If negative, the expert is worse than the main model -> Retrain Expert
            gain = expert_acc - main_acc

            # Log per-expert metrics to MLflow
            mlflow.log_metric(f"{expert_name}_main_acc", main_acc)
            mlflow.log_metric(f"{expert_name}_expert_acc", expert_acc)
            mlflow.log_metric(f"{expert_name}_routing_acc", routing_acc)
            mlflow.log_metric(f"{expert_name}_potential_gain", gain)

            print(f"{expert_name:<20} | {total:<8} | {main_acc:.2%}     | {expert_acc:.2%}     | {routing_acc:.2%}     | {gain:+.2%}")

        print("-" * 85)


if __name__ == "__main__":
    main()
