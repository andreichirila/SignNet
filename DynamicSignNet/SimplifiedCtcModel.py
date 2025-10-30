import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import CTCLoss
import glob
import json
import csv
from datetime import datetime
from pathlib import Path
import math
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from torchinfo import summary
import platform
import psutil
import random
import subprocess
from torch.amp import autocast, GradScaler
from collections import Counter

# ==================== DATASET (Simplified for debugging) ====================

class LandmarkDataset(Dataset):
    """Dataset loader - simplified for debugging"""
    def __init__(self, landmarks_dir, vocab_file=None, build_vocab=False, 
                 augment=False, max_samples=None, random_subset=False, seed=42,
                 compute_stats=False, stats_file=None, min_frequency=1):
        self.landmarks_dir = landmarks_dir
        self.samples = sorted(glob.glob(os.path.join(landmarks_dir, "*.npz")))
        self.augment = augment
        self.seed = seed
        self.min_frequency = min_frequency

        # Limit samples
        if max_samples is not None:
            if random_subset:
                random.seed(seed)
                self.samples = random.sample(self.samples, 
                                           min(max_samples, len(self.samples)))
                print(f"✓ Random subset: {len(self.samples)} samples (seed={seed})")
            else:
                self.samples = self.samples[:max_samples]
                print(f"✓ First {len(self.samples)} samples")
                
        if build_vocab:
            self.gloss_vocab = self._build_vocab()
            self._save_vocab(vocab_file)
        else:
            self.gloss_vocab = self._load_vocab(vocab_file)

        self.idx_to_gloss = {v: k for k, v in self.gloss_vocab.items()}
        
        # Compute class weights
        self.class_weights = self._compute_class_weights()
        
        # Compute normalization statistics
        if compute_stats:
            self.mean, self.std = self._compute_normalization_stats()
            self._save_stats(stats_file)
        elif stats_file and os.path.exists(stats_file):
            self.mean, self.std = self._load_stats(stats_file)
        else:
            print("⚠ Warning: Using zero mean and unit std.")
            self.mean = 0.0
            self.std = 1.0
    
    def _compute_class_weights(self):
        """Compute class weights - DISABLED for sanity check"""
        print("ℹ️ Class weighting DISABLED for sanity check")
        num_classes = len(self.gloss_vocab)
        return torch.ones(num_classes)
    
    def save_dataset_labels(self, output_file="dataset_labels.csv"):
        """Save all labels in the dataset for inspection"""
        print(f"\n{'='*50}")
        print("📋 Saving Dataset Labels for Inspection")
        print(f"{'='*50}")
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Sample_ID', 'Filename', 'Num_Frames', 'Num_Glosses', 
                           'Frames_Per_Gloss', 'Glosses', 'Gloss_IDs'])
            
            for idx, sample_path in enumerate(tqdm(self.samples, desc="Extracting labels")):
                try:
                    data = np.load(sample_path, allow_pickle=True)
                    landmarks = data['landmarks']
                    glosses = data['glosses']
                    
                    num_frames = landmarks.shape[0]
                    num_glosses = len(glosses)
                    frames_per_gloss = num_frames / num_glosses if num_glosses > 0 else 0
                    
                    # Convert glosses to IDs
                    gloss_ids = []
                    for g in glosses:
                        g_str = str(g)
                        gloss_ids.append(self.gloss_vocab.get(g_str, self.gloss_vocab['<unk>']))
                    
                    glosses_str = ' '.join([str(g) for g in glosses])
                    gloss_ids_str = ' '.join([str(gid) for gid in gloss_ids])
                    
                    writer.writerow([
                        idx,
                        os.path.basename(sample_path),
                        num_frames,
                        num_glosses,
                        f"{frames_per_gloss:.2f}",
                        glosses_str,
                        gloss_ids_str
                    ])
                except Exception as e:
                    print(f"⚠️ Error loading {sample_path}: {e}")
                    continue
        
        print(f"✓ Saved dataset labels to: {output_file}")
        
        # Print summary statistics
        print(f"\n{'='*50}")
        print("📊 Dataset Statistics")
        print(f"{'='*50}")
        self._print_dataset_stats()
    
    def _print_dataset_stats(self):
        """Print statistics about the dataset"""
        all_glosses = []
        all_num_frames = []
        all_num_glosses = []
        all_fps_ratios = []
        
        for sample_path in self.samples:
            try:
                data = np.load(sample_path, allow_pickle=True)
                landmarks = data['landmarks']
                glosses = data['glosses']
                
                all_num_frames.append(landmarks.shape[0])
                all_num_glosses.append(len(glosses))
                all_fps_ratios.append(landmarks.shape[0] / max(len(glosses), 1))
                all_glosses.extend([str(g) for g in glosses])
            except:
                continue
        
        print(f"Total samples: {len(self.samples)}")
        print(f"Total unique glosses: {len(set(all_glosses))}")
        print(f"\nFrames statistics:")
        print(f"  Min: {min(all_num_frames)}")
        print(f"  Max: {max(all_num_frames)}")
        print(f"  Mean: {np.mean(all_num_frames):.1f}")
        print(f"  Median: {np.median(all_num_frames):.1f}")
        print(f"\nGlosses per sample:")
        print(f"  Min: {min(all_num_glosses)}")
        print(f"  Max: {max(all_num_glosses)}")
        print(f"  Mean: {np.mean(all_num_glosses):.1f}")
        print(f"  Median: {np.median(all_num_glosses):.1f}")
        print(f"\nFrames per gloss ratio:")
        print(f"  Min: {min(all_fps_ratios):.2f}")
        print(f"  Max: {max(all_fps_ratios):.2f}")
        print(f"  Mean: {np.mean(all_fps_ratios):.2f}")
        print(f"  Median: {np.median(all_fps_ratios):.2f}")
        
        # Count gloss frequencies
        gloss_counts = Counter(all_glosses)
        print(f"\nTop 20 most frequent glosses:")
        for i, (gloss, count) in enumerate(gloss_counts.most_common(20), 1):
            print(f"  {i:2d}. {gloss:20s} {count:4d} times")
        
        print(f"\nTop 20 rarest glosses (that passed filtering):")
        for i, (gloss, count) in enumerate(list(gloss_counts.most_common())[-20:], 1):
            print(f"  {i:2d}. {gloss:20s} {count:4d} times")
            
    def _compute_normalization_stats(self):
        """Compute mean and std for landmark normalization"""
        print("Computing normalization statistics...")
        all_landmarks = []
        
        sample_size = min(len(self.samples), max(20, len(self.samples) // 10))
        sampled_files = random.sample(self.samples, sample_size)
        
        for sample_path in tqdm(sampled_files, desc="Computing stats"):
            data = np.load(sample_path, allow_pickle=True)
            landmarks = data['landmarks'].astype(np.float32)
            landmarks = self._normalize_landmarks(landmarks)
            all_landmarks.append(landmarks)
        
        all_landmarks = np.concatenate(all_landmarks, axis=0)
        mean = np.mean(all_landmarks, axis=0)
        std = np.std(all_landmarks, axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        
        print(f"✓ Computed statistics from {sample_size} samples")
        return mean, std
    
    def _save_stats(self, stats_file):
        if stats_file:
            np.savez(stats_file, mean=self.mean, std=self.std)
            print(f"✓ Normalization stats saved to {stats_file}")
    
    def _load_stats(self, stats_file):
        data = np.load(stats_file)
        print(f"✓ Normalization stats loaded from {stats_file}")
        return data['mean'], data['std']

    def _build_vocab(self):
        """Build vocabulary with frequency filtering"""
        gloss_counts = Counter()
        for sample_path in tqdm(self.samples, desc="Building vocab"):
            data = np.load(sample_path, allow_pickle=True)
            gloss_counts.update(data['glosses'])

        # Filter rare glosses
        filtered_glosses = {g for g, count in gloss_counts.items() if count >= self.min_frequency}
        
        gloss_to_idx = {
            '<pad>': 0, 
            '<blank>': 1, 
            '<sos>': 2, 
            '<eos>': 3,
            '<unk>': 4
        }
        
        for idx, gloss in enumerate(sorted(filtered_glosses)):
            gloss_to_idx[gloss] = idx + 5

        print(f"✓ Vocabulary size: {len(gloss_to_idx)} (from {len(gloss_counts)} unique glosses)")
        
        return gloss_to_idx

    def _save_vocab(self, vocab_file):
        if vocab_file:
            with open(vocab_file, 'w') as f:
                json.dump(self.gloss_vocab, f, indent=2)
            print(f"✓ Vocabulary saved to {vocab_file}")

    def _load_vocab(self, vocab_file):
        with open(vocab_file, 'r') as f:
            vocab = json.load(f)
        print(f"✓ Vocabulary loaded from {vocab_file}, size: {len(vocab)}")
        return vocab
        
    def _normalize_landmarks(self, landmarks):
        """Preprocessing pipeline for MediaPipe landmarks"""
        num_frames = landmarks.shape[0]
        feature_dim = landmarks.shape[1]
        num_landmarks = feature_dim // 3
        landmarks_3d = landmarks.reshape(num_frames, num_landmarks, 3)
        
        processed = []
        
        for frame_idx in range(num_frames):
            frame_landmarks = landmarks_3d[frame_idx].copy()
            
            hands_start, hands_end = 0, 42
            face_start, face_end = 42, 520
            pose_start, pose_end = 520, 553
            
            # Hands normalization
            hands = frame_landmarks[hands_start:hands_end].copy()
            if not np.all(np.isclose(hands, 0, atol=1e-6)):
                left_hand = hands[0:21].copy()
                right_hand = hands[21:42].copy()
                
                if not np.all(np.isclose(left_hand, 0, atol=1e-6)):
                    left_hand = left_hand - left_hand[0]
                
                if not np.all(np.isclose(right_hand, 0, atol=1e-6)):
                    right_hand = right_hand - right_hand[0]
                
                hands = np.concatenate([left_hand, right_hand])
                frame_landmarks[hands_start:hands_end] = hands
            
            # Face normalization
            face = frame_landmarks[face_start:face_end].copy()
            if not np.all(np.isclose(face, 0, atol=1e-6)):
                reference = face[1].copy()
                face = face - reference
                frame_landmarks[face_start:face_end] = face
            
            # Pose normalization
            pose = frame_landmarks[pose_start:pose_end].copy()
            if not np.all(np.isclose(pose, 0, atol=1e-6)):
                left_shoulder = pose[11].copy()
                right_shoulder = pose[12].copy()
                reference = (left_shoulder + right_shoulder) / 2.0
                pose = pose - reference
                frame_landmarks[pose_start:pose_end] = pose
            
            # Scale normalization
            parts = [
                (hands_start, hands_end, "hands"),
                (face_start, face_end, "face"),
                (pose_start, pose_end, "pose")
            ]
            
            for start, end, name in parts:
                part = frame_landmarks[start:end].copy()
                distances = np.linalg.norm(part, axis=-1)
                max_dist = np.max(distances)
                
                if max_dist > 1e-6:
                    part = part / max_dist
                    frame_landmarks[start:end] = part
            
            processed.append(frame_landmarks)
        
        processed = np.array(processed)
        processed = processed.reshape(num_frames, -1)
        
        return processed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx], allow_pickle=True)
        landmarks = data['landmarks'].astype(np.float32)

        # Apply preprocessing
        landmarks = self._normalize_landmarks(landmarks)
        landmarks = (landmarks - self.mean) / self.std
        landmarks = np.clip(landmarks, -5.0, 5.0)

        # NO AUGMENTATION for sanity check
        landmarks = torch.FloatTensor(landmarks)
        
        # Handle unknown glosses
        glosses = []
        for g in data['glosses']:
            g_str = str(g)
            glosses.append(self.gloss_vocab.get(g_str, self.gloss_vocab['<unk>']))
        
        return landmarks, torch.LongTensor(glosses)

def collate_fn(batch):
    """Collate function for variable-length sequences"""
    landmarks, glosses = zip(*batch)

    max_len = max(lm.shape[0] for lm in landmarks)
    feature_dim = landmarks[0].shape[1]
    padded_landmarks = torch.zeros(len(landmarks), max_len, feature_dim)
    landmark_lengths = []

    for i, lm in enumerate(landmarks):
        padded_landmarks[i, :lm.shape[0]] = lm
        landmark_lengths.append(lm.shape[0])

    max_gloss_len = max(len(g) for g in glosses)
    padded_glosses = torch.zeros(len(glosses), max_gloss_len).long()
    gloss_lengths = []

    for i, g in enumerate(glosses):
        padded_glosses[i, :len(g)] = g
        gloss_lengths.append(len(g))

    return (padded_landmarks, torch.LongTensor(landmark_lengths),
            padded_glosses, torch.LongTensor(gloss_lengths))

# ==================== EVEN SIMPLER MODEL ====================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MinimalCTCModel(nn.Module):
    """Minimal CTC model for debugging"""
    def __init__(self, input_dim, num_glosses, d_model=128, nhead=2,
                 num_encoder_layers=2, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_glosses = num_glosses
        self.blank_id = 1

        # Simple input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Minimal transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Simple CTC head
        self.ctc_head = nn.Linear(d_model, num_glosses)

        self._init_weights()

    def _init_weights(self):
        """Conservative initialization"""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
            elif 'bias' in name:
                nn.init.zeros_(p)
        
        # CRITICAL: Small positive bias for blank token only
        with torch.no_grad():
            self.ctc_head.bias[self.blank_id] = 0.5

    def forward(self, src, src_lengths):
        """Forward pass"""
        # Input projection
        x = self.input_proj(src)
        x = self.input_norm(x)
        x = F.relu(x)

        # Positional encoding
        x = self.pos_encoder(x)

        # Create padding mask
        src_mask = self._generate_padding_mask(src_lengths, src.size(1)).to(src.device)

        # Encode
        memory = self.encoder(x, src_key_padding_mask=src_mask)

        # CTC output
        ctc_logits = self.ctc_head(memory)

        return ctc_logits

    def _generate_padding_mask(self, lengths, max_len):
        """Generate padding mask"""
        batch_size = len(lengths)
        mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths.unsqueeze(1)
        return mask

# ==================== SIMPLE CTC LOSS ====================

class SimpleCTCLoss(nn.Module):
    """Simple CTC Loss without complications"""
    def __init__(self, blank_id=1):
        super().__init__()
        self.blank_id = blank_id
        self.ctc_criterion = CTCLoss(blank=blank_id, zero_infinity=True, reduction='mean')

    def forward(self, ctc_logits, targets, src_lengths, tgt_lengths):
        """Compute simple CTC loss"""
        ctc_logits_fp32 = ctc_logits.float()
        ctc_logits_t = ctc_logits_fp32.transpose(0, 1)
        ctc_log_probs = F.log_softmax(ctc_logits_t, dim=-1)
        
        loss = self.ctc_criterion(ctc_log_probs, targets, src_lengths, tgt_lengths)
        
        return loss

# ==================== DECODING ====================

def decode_predictions(ctc_output, lengths, vocab, blank_id=1):
    """Decode CTC output using greedy decoding"""
    batch_size = ctc_output.size(0)
    predictions = torch.argmax(ctc_output, dim=-1)
    
    decoded = []
    for i in range(batch_size):
        pred = predictions[i, :lengths[i]].cpu().numpy()
        
        pred_seq = []
        prev_token = None
        for token in pred:
            if token == blank_id:
                prev_token = None
                continue
            
            if token != prev_token:
                pred_seq.append(int(token))
                prev_token = token
        
        glosses = [vocab.get(p, '<unk>') for p in pred_seq]
        decoded.append(glosses)
    
    return decoded

# ==================== METRICS ====================

def compute_wer(predictions, targets):
    """Compute Word Error Rate"""
    errors = 0
    total_words = 0

    for pred, tgt in zip(predictions, targets):
        pred_words = pred if isinstance(pred, list) else pred.split()
        tgt_words = tgt if isinstance(tgt, list) else tgt.split()

        d = [[0] * (len(tgt_words) + 1) for _ in range(len(pred_words) + 1)]

        for i in range(len(pred_words) + 1):
            d[i][0] = i
        for j in range(len(tgt_words) + 1):
            d[0][j] = j

        for i in range(1, len(pred_words) + 1):
            for j in range(1, len(tgt_words) + 1):
                if pred_words[i-1] == tgt_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1

        errors += d[len(pred_words)][len(tgt_words)]
        total_words += len(tgt_words)

    return errors / max(total_words, 1)

# ==================== TRAINING ====================

def train_epoch(model, train_loader, optimizer, criterion, device, vocab, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    blank_counts = []
    non_blank_counts = []

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    for batch_idx, (landmarks, landmark_lengths, glosses, gloss_lengths) in enumerate(pbar):
        landmarks = landmarks.to(device)
        glosses = glosses.to(device)
        landmark_lengths = landmark_lengths.to(device)
        gloss_lengths = gloss_lengths.to(device)

        ctc_logits = model(landmarks, landmark_lengths)
        loss = criterion(ctc_logits, glosses, landmark_lengths, gloss_lengths)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️ WARNING: Invalid loss at batch {batch_idx}")
            continue

        optimizer.zero_grad()
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        
        with torch.no_grad():
            preds = torch.argmax(ctc_logits, dim=-1)
            blank_ratio = (preds == 1).float().mean().item()
            blank_counts.append(blank_ratio)
            non_blank_counts.append(1 - blank_ratio)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/num_batches:.4f}',
            'blank%': f'{blank_ratio*100:.1f}',
            'grad': f'{grad_norm:.3f}'
        })

    avg_blank = np.mean(blank_counts) * 100
    print(f"\n📊 Epoch {epoch+1} Stats: Avg blank predictions: {avg_blank:.1f}%")
    
    return total_loss / num_batches

def validate(model, val_loader, criterion, device, vocab, idx_to_gloss,
             save_predictions=True, output_dir="validation_outputs", epoch=None):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0

    all_predictions = []
    all_targets = []
    
    error_samples = []
    
    blank_ratios = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}" if epoch is not None else "Validation")

        for batch_idx, (landmarks, landmark_lengths, glosses, gloss_lengths) in enumerate(pbar):
            landmarks = landmarks.to(device)
            glosses = glosses.to(device)
            landmark_lengths = landmark_lengths.to(device)
            gloss_lengths = gloss_lengths.to(device)

            ctc_logits = model(landmarks, landmark_lengths)
            loss = criterion(ctc_logits, glosses, landmark_lengths, gloss_lengths)

            total_loss += loss.item()
            num_batches += 1
            
            preds_tokens = torch.argmax(ctc_logits, dim=-1)
            blank_ratio = (preds_tokens == 1).float().mean().item()
            blank_ratios.append(blank_ratio)

            preds = decode_predictions(ctc_logits, landmark_lengths, idx_to_gloss, blank_id=1)
            all_predictions.extend(preds)

            batch_targets = []
            for i in range(glosses.size(0)):
                target = glosses[i, :gloss_lengths[i]].cpu().numpy()
                special_tokens = {0, 1, 2, 3}
                target_glosses = [idx_to_gloss.get(int(t), '<unk>') 
                                  for t in target 
                                  if int(t) not in special_tokens]
                batch_targets.append(target_glosses)
            all_targets.extend(batch_targets)
            
            for i, (pred, tgt) in enumerate(zip(preds, batch_targets)):
                if pred != tgt:
                    error_samples.append({
                        'target': ' '.join(tgt),
                        'prediction': ' '.join(pred) if pred else '<EMPTY>',
                        'target_len': len(tgt),
                        'pred_len': len(pred)
                    })

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / num_batches
    wer = compute_wer(all_predictions, all_targets)

    correct = sum(1 for p, t in zip(all_predictions, all_targets) if p == t)
    accuracy = correct / len(all_predictions) if all_predictions else 0
    
    non_empty = sum(1 for p in all_predictions if len(p) > 0)
    non_empty_rate = non_empty / len(all_predictions) if all_predictions else 0
    
    avg_blank = np.mean(blank_ratios) * 100

    print(f"\n📊 Validation Stats:")
    print(f"  Avg blank predictions: {avg_blank:.1f}%")
    print(f"  Non-empty predictions: {non_empty}/{len(all_predictions)} ({non_empty_rate*100:.1f}%)")

    if save_predictions and epoch is not None:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        error_file = output_path / f"errors_epoch{epoch:03d}_{timestamp}.txt"
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(f"ERROR ANALYSIS - Epoch {epoch}\n")
            f.write(f"Total errors: {len(error_samples)}/{len(all_predictions)}\n")
            f.write(f"WER: {wer:.4f}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Non-empty rate: {non_empty_rate:.4f}\n")
            f.write(f"Avg blank%: {avg_blank:.1f}\n\n")
            f.write("="*80 + "\n\n")
            
            for i, error in enumerate(error_samples[:50]):
                f.write(f"Error {i+1}:\n")
                f.write(f"  Target:     {error['target']}\n")
                f.write(f"  Prediction: {error['prediction']}\n")
                f.write(f"  Lengths:    {error['target_len']} -> {error['pred_len']}\n")
                f.write("-"*80 + "\n")
        
        print(f"✓ Saved error analysis to: {error_file}")

    mlflow.log_metric("val_loss", avg_loss)
    mlflow.log_metric("val_wer", wer)
    mlflow.log_metric("val_accuracy", accuracy)
    mlflow.log_metric("val_non_empty_rate", non_empty_rate)
    mlflow.log_metric("val_blank_ratio", avg_blank / 100)

    return avg_loss, wer, accuracy, non_empty_rate

def train_model(model, train_loader, val_loader, num_epochs, device, vocab, idx_to_gloss, 
                save_dir='checkpoints'):
    """Full training loop"""
    os.makedirs(save_dir, exist_ok=True)

    criterion = SimpleCTCLoss(blank_id=1)
    
    base_lr = 5e-3
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=base_lr,
        eps=1e-8
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    print(f"\n{'='*50}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*50}")
    print(f"Learning Rate: {base_lr:.2e}")
    print(f"Optimizer: Adam")
    print(f"Scheduler: StepLR (decay every 5 epochs)")
    print(f"Loss: Simple CTC")
    print(f"{'='*50}\n")

    best_non_empty = 0.0
    best_accuracy = 0.0
    patience_counter = 0
    max_patience = 10

    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, 
            device, vocab, epoch
        )
        print(f"✓ Train Loss: {train_loss:.4f}")

        val_loss, wer, accuracy, non_empty_rate = validate(
            model, val_loader, criterion, 
            device, vocab, idx_to_gloss,
            save_predictions=True,
            output_dir="validation_outputs",
            epoch=epoch
        )
        print(f"✓ Val Loss: {val_loss:.4f}, WER: {wer:.4f}, Accuracy: {accuracy:.4f}, Non-empty: {non_empty_rate:.4f}")

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_wer", wer, step=epoch)
        mlflow.log_metric("val_accuracy", accuracy, step=epoch)
        mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)
        
        scheduler.step()

        if non_empty_rate > best_non_empty:
            best_non_empty = non_empty_rate
            best_accuracy = accuracy
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'wer': wer,
                'accuracy': accuracy,
                'non_empty_rate': non_empty_rate
            }
            checkpoint_path = os.path.join(save_dir, 'best_model.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved best model: Non-empty={non_empty_rate:.4f}, Accuracy={accuracy:.4f}")
            mlflow.log_metric("best_non_empty", non_empty_rate)
            mlflow.log_metric("best_accuracy", accuracy)
        else:
            patience_counter += 1
            
        if non_empty_rate > 0.5:
            print(f"✓ Progress! Model producing non-empty sequences ({non_empty_rate*100:.1f}%)")
            
        if patience_counter >= max_patience:
            print(f"⚠ Early stopping triggered after {epoch+1} epochs")
            break

    return best_accuracy, best_non_empty

# ==================== MAIN ====================

def main():
    # Configuration
    LANDMARKS_TRAIN = "./landmarks_train"
    LANDMARKS_DEV = "./landmarks_dev"
    VOCAB_FILE = "vocab_debug.json"
    STATS_FILE = "normalization_stats_debug.npz"
    
    # MINIMAL SETUP
    TRAIN_SAMPLES = 50
    VAL_SAMPLES = 20
    BATCH_SIZE = 4
    NUM_EPOCHS = 50
    INPUT_DIM = 1659
    D_MODEL = 128
    NHEAD = 2
    NUM_ENCODER_LAYERS = 2
    DROPOUT = 0.0
    MIN_FREQUENCY = 1
    SEED = 42
    
    # Set random seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    EXPERIMENT_NAME = "SignNetDebug"
    RUN_NAME = "minimal_ctc_debug"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")

    # MLflow setup
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'roman'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'SignNet'
    mlflow.set_tracking_uri("https://mlflow.schlaepfer.me")
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(log_system_metrics=True, run_name=RUN_NAME):
        mlflow.log_params({
            "mode": "MINIMAL_DEBUG",
            "train_samples": TRAIN_SAMPLES,
            "val_samples": VAL_SAMPLES,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_encoder_layers": NUM_ENCODER_LAYERS,
            "dropout": DROPOUT,
            "learning_rate": 5e-3,
            "loss": "simple_ctc"
        })

        # Create datasets
        print("\n" + "="*50)
        print("📂 Loading Datasets")
        print("="*50)

        compute_stats = not os.path.exists(STATS_FILE)

        train_dataset = LandmarkDataset(
            LANDMARKS_TRAIN,
            vocab_file=VOCAB_FILE,
            build_vocab=True,
            augment=False,
            compute_stats=compute_stats,
            stats_file=STATS_FILE,
            seed=SEED,
            min_frequency=MIN_FREQUENCY,
            max_samples=TRAIN_SAMPLES,
            random_subset=True
        )
        
        # SAVE DATASET LABELS
        train_dataset.save_dataset_labels("train_labels.csv")

        val_dataset = LandmarkDataset(
            LANDMARKS_DEV,
            vocab_file=VOCAB_FILE,
            build_vocab=False,
            augment=False,
            compute_stats=False,
            stats_file=STATS_FILE,
            seed=SEED,
            min_frequency=MIN_FREQUENCY,
            max_samples=VAL_SAMPLES,
            random_subset=True
        )
        
        # SAVE VAL DATASET LABELS
        val_dataset.save_dataset_labels("val_labels.csv")
        
        print(f"\n✓ Dataset Summary:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Vocabulary size: {len(train_dataset.gloss_vocab)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        # Create model
        print("\n" + "="*50)
        print("🏗️  Creating Minimal Model")
        print("="*50)

        vocab_size = len(train_dataset.gloss_vocab)
        model = MinimalCTCModel(
            input_dim=INPUT_DIM,
            num_glosses=vocab_size,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            dropout=DROPOUT
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Total parameters: {total_params:,}")

        # Train model
        print("\n" + "="*50)
        print("🎯 Starting Training")
        print("="*50)

        best_accuracy, best_non_empty = train_model(
            model,
            train_loader,
            val_loader,
            NUM_EPOCHS,
            device,
            train_dataset.gloss_vocab,
            train_dataset.idx_to_gloss
        )

        print(f"\n{'='*50}")
        print(f"✅ Training Complete!")
        print(f"{'='*50}")
        print(f"Best Non-empty Rate: {best_non_empty:.4f}")
        print(f"Best Accuracy: {best_accuracy:.4f}")
        print(f"{'='*50}")
        
        print(f"\n📋 Dataset label files created:")
        print(f"  - train_labels.csv")
        print(f"  - val_labels.csv")

if __name__ == "__main__":
    main()
