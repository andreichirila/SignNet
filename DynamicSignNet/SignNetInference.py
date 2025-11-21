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
    parser.add_argument('--main-model-path', type=str, default='./models_balanced/sign_classifier_best_enhanced.pth')
    parser.add_argument('--direction-expert-path', type=str, default='./models_balanced/expert_direction_expert_sign_classifier_best_enhanced.pth')
    parser.add_argument('--main-model-uri', type=str, default='models:/Production/1')
    parser.add_argument('--direction-expert-uri', type=str, default='models:/SignClassifier_DirectionExpert/1')
    parser.add_argument('--training-run-id', type=str, required=True, help='MLflow run ID from training that contains val_indices.npy')
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
        top1_conf, top1_idx = torch.max(root_probs, dim=1)

        final_preds = []
        used_expert = []

        for i in range(len(landmarks)):
            pred_class_name = self.root_idx_to_word[top1_idx[i].item()]

            if pred_class_name in self.class_to_expert:
                expert_name = self.class_to_expert[pred_class_name]

                if expert_name in self.expert_models:
                    expert_model = self.expert_models[expert_name]
                    expert_dict = self.expert_dicts[expert_name]

                    expert_logits, _ = expert_model(landmarks[i:i+1], padding_mask[i:i+1] if padding_mask is not None else None)
                    expert_prob = torch.softmax(expert_logits, dim=1)
                    _, expert_pred_idx = torch.max(expert_prob, dim=1)

                    expert_pred_name = expert_dict['idx_to_word'][expert_pred_idx.item()]

                    final_preds.append(expert_pred_name)
                    used_expert.append(expert_name)
                else:
                    final_preds.append(pred_class_name)
                    used_expert.append("None (Missing)")
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
    mlflow.set_tracking_uri("http://mlflow.schlaepfer.me")
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
        if args.dataset_type == 'flat':
            npz_files = sorted(Path(args.data_dir).glob("*.npz"))
        else:
            npz_files = sorted((Path(args.data_dir) / "train").glob("*.npz"))

        main_vocab_words, _ = build_topk_vocabulary(npz_files, K=170, min_samples=70, debug=False)
        sorted_main_vocab = sorted(list(main_vocab_words))
        main_word_to_idx = {w: i for i, w in enumerate(sorted_main_vocab)}
        main_idx_to_word = {i: w for i, w in enumerate(sorted_main_vocab)}
        num_classes_main = len(main_vocab_words)
        print(f"Main Vocabulary: {num_classes_main} classes")

        # 2. Load Models
        main_model = load_model(TransformerSignClassifierWithHandedness, args.main_model_path, MAIN_MODEL_CONFIG, num_classes_main, DEVICE)

        expert_models = {}
        expert_dicts = {}

        # Load Direction Expert
        if os.path.exists(args.direction_expert_path):
            vocab_list = sorted(HIERARCHY_CONFIG['direction_expert'])
            expert_dicts['direction_expert'] = {
                'word_to_idx': {w: i for i, w in enumerate(vocab_list)},
                'idx_to_word': {i: w for i, w in enumerate(vocab_list)}
            }
            expert_models['direction_expert'] = load_model(TransformerSignClassifierWithHandedness, args.direction_expert_path, EXPERT_MODEL_CONFIG, len(vocab_list), DEVICE)
            print("Loaded Direction Expert")

        hierarchical_model = HierarchicalClassifier(main_model, expert_models, HIERARCHY_CONFIG, main_idx_to_word, expert_dicts)

        # 3. Load Data
        print("Downloading validation indices from MLflow...")
        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=args.training_run_id,
            artifact_path="val_indices.npy"
        )
        val_indices = np.load(artifact_path)
        print(f"Loaded {len(val_indices)} validation samples")

        # Rest of the code remains the same
        if args.dataset_type == 'flat':
            data_root = args.data_dir
        else:
            data_root = os.path.join(args.data_dir, "train")

        base_dataset = SignLanguageDataset(data_root, debug=False)

        # Build mapping old_label_idx -> new_label_idx (vocab index)
        old_to_new_idx = {}
        for new_idx, word in enumerate(sorted_main_vocab):
            if word in base_dataset.word_to_idx:
                old_idx = base_dataset.word_to_idx[word]
                old_to_new_idx[old_idx] = new_idx

        from SignNetWord import RemappedDataset
        val_subset = RemappedDataset(base_dataset, val_indices.tolist(), old_to_new_idx)
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=64,
            collate_fn=PadCollate(),
            shuffle=False
        )

        # 4. Run Evaluation
        print("\nStarting Hierarchical Evaluation...")
        all_preds_hierarchical = []
        all_preds_baseline = []
        all_labels = []
        expert_usage_stats = Counter()

        with torch.no_grad():
            for batch in tqdm(val_loader):
                landmarks, labels, _, padding_mask = batch
                landmarks = landmarks.to(DEVICE)

                batch_labels_names = [main_idx_to_word[l.item()] for l in labels]
                all_labels.extend(batch_labels_names)

                # Hierarchical
                preds, experts_used = hierarchical_model(landmarks, padding_mask)
                all_preds_hierarchical.extend(preds)
                expert_usage_stats.update(experts_used)

                # Baseline
                logits, _ = main_model(landmarks, padding_mask)
                base_preds_idx = torch.argmax(logits, dim=1)
                base_preds_names = [main_idx_to_word[i.item()] for i in base_preds_idx]
                all_preds_baseline.extend(base_preds_names)

        # 5. Metrics & Logging
        acc_base = accuracy_score(all_labels, all_preds_baseline)
        acc_hier = accuracy_score(all_labels, all_preds_hierarchical)

        # Log Global Metrics
        mlflow.log_metric("acc_baseline", acc_base)
        mlflow.log_metric("acc_hierarchical", acc_hier)
        mlflow.log_metric("acc_improvement", acc_hier - acc_base)

        print(f"Baseline: {acc_base:.4%}")
        print(f"Hierarchical: {acc_hier:.4%}")

        # Log Expert Stats
        for expert, count in expert_usage_stats.items():
            mlflow.log_metric(f"usage_{expert}", count)

        # Detailed Analysis (Direction)
        dir_classes = HIERARCHY_CONFIG['direction_expert']
        dir_indices = [i for i, label in enumerate(all_labels) if label in dir_classes]

        if dir_indices:
            dir_labels = [all_labels[i] for i in dir_indices]
            dir_base = [all_preds_baseline[i] for i in dir_indices]
            dir_hier = [all_preds_hierarchical[i] for i in dir_indices]

            acc_dir_base = accuracy_score(dir_labels, dir_base)
            acc_dir_hier = accuracy_score(dir_labels, dir_hier)

            mlflow.log_metric("acc_direction_baseline", acc_dir_base)
            mlflow.log_metric("acc_direction_hierarchical", acc_dir_hier)
            mlflow.log_metric("acc_direction_improvement", acc_dir_hier - acc_dir_base)

            print(f"Direction Improvement: {acc_dir_base:.4%} -> {acc_dir_hier:.4%}")

if __name__ == "__main__":
    main()
