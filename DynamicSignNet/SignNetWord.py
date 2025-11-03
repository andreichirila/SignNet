import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn import CTCLoss
import glob
import json
import math
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from torchinfo import summary
import platform
import psutil
import random
from collections import defaultdict


# ==================== DATASET ====================

class LandmarkDataset(Dataset):
    """Dataset loader for preprocessed landmarks with train/val/test split support"""

    def __init__(self, landmarks_dir=None, vocab_file=None, build_vocab=False,
                 augment=False, all_dirs_for_vocab=None, max_samples=None,
                 random_subset=False, samples_list=None):
        """
        Args:
            landmarks_dir: Directory containing .npz files (if loading from directory)
            vocab_file: Path to vocabulary JSON file
            build_vocab: Whether to build vocabulary from scratch
            augment: Whether to apply augmentation
            all_dirs_for_vocab: Not used anymore (kept for compatibility)
            max_samples: Maximum number of samples to use
            random_subset: Whether to randomly sample max_samples
            samples_list: Pre-filtered list of sample paths (for train/val/test splits)
        """
        if samples_list is not None:
            # Use pre-filtered samples (for train/val/test splits)
            self.samples = samples_list
            self.landmarks_dir = None
        else:
            # Load from directory
            self.landmarks_dir = landmarks_dir
            self.samples = sorted(glob.glob(os.path.join(landmarks_dir, "*.npz")))

        self.augment = augment

        # Initialize augmentation
        if self.augment:
            self.temporal_aug = AdvancedTemporalAugmentation(prob=0.8)

        # Limit samples
        if max_samples is not None and samples_list is None:
            if random_subset:
                random.seed(42)
                self.samples = random.sample(self.samples,
                                             min(max_samples, len(self.samples)))
                print(f"Random subset: {len(self.samples)} samples (seed=42)")
            else:
                self.samples = self.samples[:max_samples]
                print(f"First {len(self.samples)} samples")

        if build_vocab:
            self.gloss_vocab = self._build_vocab()
            self._save_vocab(vocab_file)
        else:
            self.gloss_vocab = self._load_vocab(vocab_file)

        self.idx_to_gloss = {v: k for k, v in self.gloss_vocab.items()}

    def _build_vocab(self):
        """Build vocabulary from all glosses in current samples"""
        glosses = set()
        for sample_path in tqdm(self.samples, desc="Building vocab"):
            data = np.load(sample_path, allow_pickle=True)
            glosses.update(data['glosses'])

        gloss_to_idx = {
            '<pad>': 0,
            '<blank>': 1,
            '<sos>': 2,
            '<eos>': 3,
            '<unk>': 4
        }

        for idx, gloss in enumerate(sorted(glosses)):
            gloss_to_idx[gloss] = idx + 5

        print(f"Vocabulary size: {len(gloss_to_idx)}")
        return gloss_to_idx

    def _save_vocab(self, vocab_file):
        """Save vocabulary to JSON file"""
        if vocab_file:
            with open(vocab_file, 'w') as f:
                json.dump(self.gloss_vocab, f, indent=2)
            print(f"Vocabulary saved to {vocab_file}")

    def _load_vocab(self, vocab_file):
        """Load vocabulary from JSON file"""
        with open(vocab_file, 'r') as f:
            vocab = json.load(f)
        print(f"Vocabulary loaded from {vocab_file}, size: {len(vocab)}")
        return vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx], allow_pickle=True)
        landmarks = data['landmarks']
        handedness = data.get('handedness', None)

        # Convert handedness strings to numeric encoding BEFORE augmentation
        if handedness is not None:
            mapping = {'LEFT': 0, 'RIGHT': 1, 'NONE': 2}
            handedness = np.vectorize(lambda x: mapping.get(x, 2))(handedness)
        else:
            handedness = np.zeros((landmarks.shape[0], 2), dtype=np.int64)

        # Apply augmentation to BOTH landmarks and handedness
        if self.augment:
            landmarks, handedness = self.temporal_aug(landmarks, handedness)

        # Convert to tensors AFTER augmentation
        landmarks = torch.FloatTensor(landmarks)
        handedness = torch.LongTensor(handedness)

        glosses = []
        for g in data['glosses']:
            g_str = str(g)
            glosses.append(self.gloss_vocab.get(g_str, self.gloss_vocab['<unk>']))

        return landmarks, handedness, torch.LongTensor(glosses)


def collate_fn(batch):
    landmarks, handedness, glosses = zip(*batch)

    max_len = max(lm.shape[0] for lm in landmarks)
    feature_dim = landmarks[0].shape[1]

    padded_landmarks = torch.zeros(len(landmarks), max_len, feature_dim)
    padded_handedness = torch.full((len(landmarks), max_len, 2), 2, dtype=torch.long)

    landmark_lengths = []
    gloss_lengths = []

    max_gloss_len = max(len(g) for g in glosses)
    padded_glosses = torch.zeros(len(glosses), max_gloss_len).long()

    for i in range(len(landmarks)):
        lm = landmarks[i]
        hd = handedness[i]
        g = glosses[i]

        padded_landmarks[i, :lm.shape[0]] = lm
        landmark_lengths.append(lm.shape[0])

        padded_handedness[i, :hd.shape[0]] = hd

        padded_glosses[i, :len(g)] = g
        gloss_lengths.append(len(g))

    return (padded_landmarks, torch.LongTensor(landmark_lengths),
            padded_handedness, padded_glosses, torch.LongTensor(gloss_lengths))


# ==================== MODEL ====================

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


class SignLanguageTranslator(nn.Module):
    """Transformer-based Sign Language Translation Model"""

    def __init__(self, input_dim, num_glosses, d_model=384, nhead=8,
                 num_encoder_layers=6, dropout=0.5):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + 6, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # Temporal convolution for local feature extraction
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # CTC head for gloss prediction
        self.ctc_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_glosses)
        )

        self.d_model = d_model

    def forward(self, src, handedness, src_lengths):
        # One-hot encode handedness
        handedness_onehot = torch.nn.functional.one_hot(handedness, num_classes=3).float()
        handedness_onehot = handedness_onehot.view(handedness.shape[0], handedness.shape[1], -1)

        x = torch.cat([src, handedness_onehot], dim=2)
        x = self.input_proj(x)

        # Temporal convolution
        x_conv = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv

        # Positional encoding
        x = self.pos_encoder(x)

        # Create padding mask
        src_mask = self._generate_padding_mask(src_lengths, src.size(1)).to(src.device)

        # Encode
        memory = self.encoder(x, src_key_padding_mask=src_mask)

        # CTC prediction
        ctc_logits = self.ctc_head(memory)

        return ctc_logits

    def _generate_padding_mask(self, lengths, max_len):
        """Generate padding mask for variable length sequences"""
        batch_size = len(lengths)
        mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths.unsqueeze(1)
        return mask


# ==================== DATA AUGMENTATION ====================

class AdvancedTemporalAugmentation:
    """More aggressive augmentation to reduce overfitting"""

    def __init__(self, prob=0.8):
        self.prob = prob

    def __call__(self, landmarks, handedness=None):
        """
        Apply temporal augmentation to landmarks and optionally handedness.
        """
        if np.random.random() > self.prob:
            if handedness is not None:
                return landmarks, handedness
            return landmarks

        num_frames = landmarks.shape[0]

        # Speed variation (aggressive)
        speed = np.random.uniform(0.65, 1.35)
        new_num_frames = max(1, int(num_frames / speed))
        indices = np.linspace(0, num_frames - 1, new_num_frames)
        indices = np.array([min(int(i), num_frames - 1) for i in indices])

        # Apply same indices to landmarks and handedness
        landmarks = landmarks[indices]
        if handedness is not None:
            handedness = handedness[indices]

        # Stronger spatial noise (only to landmarks)
        noise = np.random.normal(0, 0.015, landmarks.shape)
        landmarks = landmarks + noise

        # Random frame dropout
        if np.random.random() < 0.3:
            keep_ratio = np.random.uniform(0.9, 0.95)
            mask = np.random.rand(landmarks.shape[0]) < keep_ratio
            if mask.sum() > 0:
                landmarks = landmarks[mask]
                if handedness is not None:
                    handedness = handedness[mask]

        if handedness is not None:
            return landmarks, handedness
        return landmarks


# ==================== STRATEGY 1: WEIGHTED SAMPLING ====================

def create_balanced_sampler(dataset):
    """
    Create a weighted sampler to balance class distribution during training.
    """
    class_counts = {}
    sample_classes = []

    for sample_path in tqdm(dataset.samples, desc="Analyzing class distribution"):
        data = np.load(sample_path, allow_pickle=True)
        gloss = str(data['glosses'][0]) if len(data['glosses']) > 0 else '<unk>'

        sample_classes.append(gloss)
        class_counts[gloss] = class_counts.get(gloss, 0) + 1

    # Calculate weights: inverse frequency
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[cls] for cls in sample_classes]

    # Print statistics
    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print(f"Total classes: {len(class_counts)}")
    print(f"Total samples: {len(sample_classes)}")

    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 most frequent classes:")
    for cls, count in sorted_classes[:10]:
        print(f"  {cls}: {count} samples ({count / len(sample_classes) * 100:.2f}%)")

    print(f"\nTop 10 least frequent classes:")
    for cls, count in sorted_classes[-10:]:
        print(f"  {cls}: {count} samples ({count / len(sample_classes) * 100:.2f}%)")
    print("=" * 70 + "\n")

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler, class_counts


# ==================== STRATEGY 2: FOCAL LOSS ====================

class FocalCTCLoss(nn.Module):
    """Focal CTC Loss: Down-weights easy examples, focuses on hard ones"""

    def __init__(self, blank=1, gamma=2.0, alpha=0.25, zero_infinity=True):
        super().__init__()
        self.blank = blank
        self.gamma = gamma
        self.alpha = alpha
        self.zero_infinity = zero_infinity
        self.ctc = CTCLoss(blank=blank, reduction='none', zero_infinity=zero_infinity)

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # Get CTC loss per sample
        ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)

        # Apply focal weight
        p_t = torch.exp(-ctc_loss)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ctc_loss

        return focal_loss.mean()


# ==================== STRATEGY 3: CLASS-AWARE AUGMENTATION ====================

class ClassAwareAugmentation:
    """Apply stronger augmentation to rare classes"""

    def __init__(self, class_counts, rare_threshold=10):
        self.class_counts = class_counts
        self.rare_threshold = rare_threshold
        self.strong_aug = AdvancedTemporalAugmentation(prob=0.95)
        self.normal_aug = AdvancedTemporalAugmentation(prob=0.8)

    def __call__(self, landmarks, handedness, gloss):
        is_rare = self.class_counts.get(gloss, 0) <= self.rare_threshold

        if is_rare:
            return self.strong_aug(landmarks, handedness)
        else:
            return self.normal_aug(landmarks, handedness)


# ==================== STRATEGY 4: OVERSAMPLING ====================

def create_oversampled_samples(samples, target_min_samples=15):
    """
    Oversample rare classes by duplicating their samples.
    """
    class_to_samples = {}

    for sample_path in tqdm(samples, desc="Analyzing for oversampling"):
        data = np.load(sample_path, allow_pickle=True)
        gloss = str(data['glosses'][0]) if len(data['glosses']) > 0 else '<unk>'

        if gloss not in class_to_samples:
            class_to_samples[gloss] = []
        class_to_samples[gloss].append(sample_path)

    new_samples = []
    oversampled_classes = []

    for gloss, class_samples in class_to_samples.items():
        num_samples = len(class_samples)

        if num_samples < target_min_samples:
            repeats = (target_min_samples + num_samples - 1) // num_samples
            oversampled = class_samples * repeats
            oversampled = oversampled[:target_min_samples]
            new_samples.extend(oversampled)
            oversampled_classes.append(f"{gloss}: {num_samples} → {len(oversampled)}")
        else:
            new_samples.extend(class_samples)

    if oversampled_classes:
        print("\n" + "=" * 70)
        print("OVERSAMPLED CLASSES")
        print("=" * 70)
        for info in oversampled_classes[:20]:  # Show first 20
            print(f"  {info}")
        if len(oversampled_classes) > 20:
            print(f"  ... and {len(oversampled_classes) - 20} more")
        print("=" * 70 + "\n")

    print(f"Total samples after oversampling: {len(samples)} → {len(new_samples)}\n")

    return new_samples


# ==================== DECODING ====================

def decode_predictions_greedy(ctc_output, lengths, idx_to_gloss, blank_id=1):
    """Fast greedy CTC decoding"""
    batch_size = ctc_output.size(0)
    predictions = torch.argmax(ctc_output, dim=-1)

    decoded = []
    for i in range(batch_size):
        pred = predictions[i, :lengths[i]]

        pred_seq = []
        prev = None
        for p in pred:
            p = p.item()
            if p != blank_id and p != prev:
                pred_seq.append(p)
            prev = p

        glosses = [idx_to_gloss.get(p, '<unk>') for p in pred_seq]
        decoded.append(glosses)

    return decoded


def decode_predictions_hybrid(ctc_output, lengths, idx_to_gloss, blank_id=1,
                              beam_width=30, confidence_threshold=0.4, use_length_penalty=True):
    """Hybrid decoding: Beam search + confidence filtering"""
    batch_size = ctc_output.size(0)
    log_probs = F.log_softmax(ctc_output, dim=-1)
    max_probs = torch.max(F.softmax(ctc_output, dim=-1), dim=-1)[0]

    decoded = []

    for batch_idx in range(batch_size):
        seq_len = lengths[batch_idx]
        probs = log_probs[batch_idx, :seq_len, :]
        conf = max_probs[batch_idx, :seq_len]

        beam = [(tuple(), 0.0, 0)]

        for t in range(len(probs)):
            new_beam = {}

            if conf[t].item() < confidence_threshold:
                for prefix, score, length in beam:
                    key = (prefix, length)
                    if key not in new_beam:
                        new_beam[key] = score - 0.1
                beam = [(p, s, l) for (p, l), s in
                        sorted(new_beam.items(), key=lambda x: x[1], reverse=True)[:beam_width]]
                continue

            for c_idx in range(ctc_output.size(2)):
                for prefix, score, length in beam:
                    if c_idx == blank_id:
                        new_prefix = prefix
                        new_length = length
                    else:
                        gloss = idx_to_gloss.get(c_idx, '<unk>')

                        if len(prefix) > 0 and prefix[-1] == gloss:
                            new_prefix = prefix
                            new_length = length
                        else:
                            new_prefix = prefix + (gloss,)
                            new_length = length + 1

                    ctc_score = probs[t, c_idx].item()
                    length_penalty = 0.0

                    if use_length_penalty and new_length == 0:
                        length_penalty = -0.1

                    new_score = score + ctc_score + length_penalty

                    key = (new_prefix, new_length)
                    if key not in new_beam or new_beam[key] < new_score:
                        new_beam[key] = new_score

            beam = [(p, s, l) for (p, l), s in sorted(new_beam.items(), key=lambda x: x[1], reverse=True)[:beam_width]]

        best_seq = beam[0][0] if beam else tuple()
        decoded.append(list(best_seq))

    return decoded


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
                if pred_words[i - 1] == tgt_words[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]) + 1

        errors += d[len(pred_words)][len(tgt_words)]
        total_words += len(tgt_words)

    return errors / max(total_words, 1)


class EarlyStopping:
    """Early stopping with patience"""

    def __init__(self, patience=8, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_wer = float('inf')
        self.best_epoch = 0

    def __call__(self, val_wer, epoch):
        if val_wer < (self.best_val_wer - self.min_delta):
            self.best_val_wer = val_wer
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                print(f"Best WER was {self.best_val_wer:.4f} at epoch {self.best_epoch}")
                return True
        return False


# ==================== TRAINING ====================

def train_epoch(model, train_loader, optimizer, criterion, device, vocab, epoch, scheduler):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (landmarks, landmark_lengths, handedness, glosses, gloss_lengths) in enumerate(pbar):
        landmarks = landmarks.to(device)
        handedness = handedness.to(device)
        glosses = glosses.to(device)
        landmark_lengths = landmark_lengths.to(device)
        gloss_lengths = gloss_lengths.to(device)

        optimizer.zero_grad()

        ctc_logits = model(landmarks, handedness, landmark_lengths)

        ctc_logits = ctc_logits.transpose(0, 1)
        log_probs = F.log_softmax(ctc_logits, dim=-1)

        loss = criterion(log_probs, glosses, landmark_lengths, gloss_lengths)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        if batch_idx % 50 == 0:
            step = epoch * len(train_loader) + batch_idx
            mlflow.log_metric("batch_loss", loss.item(), step=step)

    return total_loss / num_batches


def validate(model, val_loader, criterion, device, vocab, idx_to_gloss, use_beam_search=False):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for landmarks, landmark_lengths, handedness, glosses, gloss_lengths in pbar:
            landmarks = landmarks.to(device)
            handedness = handedness.to(device)
            glosses = glosses.to(device)
            landmark_lengths = landmark_lengths.to(device)
            gloss_lengths = gloss_lengths.to(device)

            ctc_logits = model(landmarks, handedness, landmark_lengths)

            ctc_logits_t = ctc_logits.transpose(0, 1)
            log_probs = F.log_softmax(ctc_logits_t, dim=-1)
            loss = criterion(log_probs, glosses, landmark_lengths, gloss_lengths)

            total_loss += loss.item()
            num_batches += 1

            if use_beam_search:
                predictions = decode_predictions_hybrid(
                    ctc_logits, landmark_lengths, idx_to_gloss,
                    blank_id=1, beam_width=30, confidence_threshold=0.4,
                    use_length_penalty=True
                )
            else:
                predictions = decode_predictions_greedy(
                    ctc_logits, landmark_lengths, idx_to_gloss, blank_id=1
                )

            for i in range(glosses.size(0)):
                target = glosses[i, :gloss_lengths[i]].cpu().numpy()
                target_glosses = [idx_to_gloss.get(int(t), '<unk>') for t in target]
                all_predictions.append(predictions[i])
                all_targets.append(target_glosses)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    wer = compute_wer(all_predictions, all_targets)

    return avg_loss, wer


def train_model(model, train_loader, val_loader, num_epochs, device, vocab, idx_to_gloss, save_dir='checkpoints'):
    """Full training loop"""
    os.makedirs(save_dir, exist_ok=True)

    criterion = CTCLoss(blank=1, zero_infinity=True)
    # Uncomment for Focal Loss (Strategy 2):
    # criterion = FocalCTCLoss(blank=1, gamma=2.0, alpha=0.25, zero_infinity=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        betas=(0.9, 0.999),
        weight_decay=2e-4
    )

    def get_lr_scheduler(optimizer, num_epochs, steps_per_epoch):
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = int(0.1 * total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return max(0.1, (1.0 - progress) ** 0.5)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scheduler = get_lr_scheduler(optimizer, num_epochs, len(train_loader))
    early_stopping = EarlyStopping(patience=8, min_delta=0.001)
    best_wer = float('inf')

    for epoch in range(num_epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'=' * 50}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, vocab, epoch, scheduler)
        print(f"Train Loss: {train_loss:.4f}")

        val_loss, val_wer = validate(model, val_loader, criterion, device, vocab, idx_to_gloss, use_beam_search=False)
        print(f"Val Loss: {val_loss:.4f}, Val WER: {val_wer:.4f} (greedy decoder - fast)")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_wer": val_wer,
            "learning_rate": optimizer.param_groups[0]['lr']
        }, step=epoch)

        if val_wer < best_wer:
            best_wer = val_wer
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_wer': val_wer,
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pt'))
            print(f"✓ Saved best model with WER: {best_wer:.4f}")
            mlflow.log_metric("best_wer", best_wer)

        if early_stopping(val_wer, epoch):
            break

    return best_wer


def evaluate_with_beam_search(model, val_loader, criterion, device, vocab, idx_to_gloss):
    """Final evaluation with beam search decoder"""
    print("\n" + "=" * 70)
    print("FINAL EVALUATION WITH BEAM SEARCH DECODER")
    print("=" * 70)

    val_loss, val_wer = validate(model, val_loader, criterion, device, vocab, idx_to_gloss, use_beam_search=True)

    print(f"Final Val Loss: {val_loss:.4f}")
    print(f"Final Val WER (Hybrid Beam Search): {val_wer:.4f}")
    print("=" * 70 + "\n")

    return val_loss, val_wer


def generate_model_summary(model, input_dim, device, batch_size=8, seq_length=100):
    """Generate comprehensive model summary"""
    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)

    dummy_input = torch.randn(batch_size, seq_length, input_dim).to(device)
    dummy_lengths = torch.tensor([seq_length] * batch_size).to(device)
    dummy_handedness = torch.randint(0, 3, (batch_size, seq_length, 2)).to(device)

    model_stats = summary(
        model,
        input_data=(dummy_input, dummy_handedness, dummy_lengths),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
        verbose=0
    )

    print(model_stats)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 ** 2

    with torch.no_grad():
        output = model(dummy_input, dummy_handedness, dummy_lengths)

    print("\n" + "=" * 70)
    print("DETAILED STATISTICS")
    print("=" * 70)
    print(f"Total Parameters:           {total_params:>15,}")
    print(f"Trainable Parameters:       {trainable_params:>15,}")
    print(f"Non-trainable Parameters:   {non_trainable_params:>15,}")
    print(f"Model Size:                 {size_mb:>15.2f} MB")
    print(f"Input Shape (landmarks):    {tuple(dummy_input.shape)}")
    print(f"Input Shape (handedness):   {tuple(dummy_handedness.shape)}")
    print(f"Output Shape:               {tuple(output.shape)}")
    print("=" * 70 + "\n")

    summary_str = str(model_stats)

    summary_dict = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "model_size_mb": size_mb,
        "input_shape_landmarks": list(dummy_input.shape),
        "input_shape_handedness": list(dummy_handedness.shape),
        "output_shape": list(output.shape)
    }

    return summary_str, summary_dict


def split_dataset(landmarks_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split dataset into train, validation, and test sets"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    all_samples = sorted(glob.glob(os.path.join(landmarks_dir, "*.npz")))

    print(f"\nDataset Split Configuration:")
    print(f"{'=' * 50}")
    print(f"Total samples: {len(all_samples)}")
    print(f"Train ratio: {train_ratio:.1%}")
    print(f"Val ratio: {val_ratio:.1%}")
    print(f"Test ratio: {test_ratio:.1%}")
    print(f"Random seed: {seed}")

    random.seed(seed)
    random.shuffle(all_samples)

    n_total = len(all_samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]

    print(f"\nSplit Results:")
    print(f"{'=' * 50}")
    print(f"Train samples: {len(train_samples)} ({len(train_samples) / n_total:.1%})")
    print(f"Val samples: {len(val_samples)} ({len(val_samples) / n_total:.1%})")
    print(f"Test samples: {len(test_samples)} ({len(test_samples) / n_total:.1%})")
    print(f"{'=' * 50}\n")

    return train_samples, val_samples, test_samples


# ==================== MAIN ====================

def main():
    # Configuration
    LANDMARKS_DIR = "./word_landmarks_extracted"
    VOCAB_FILE = "vocab.json"
    BATCH_SIZE = 16
    NUM_EPOCHS = 150
    INPUT_DIM = 1659

    # Dataset split ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    SPLIT_SEED = 42

    # Model configuration
    D_MODEL = 384
    NHEAD = 8
    NUM_LAYERS = 6
    DROPOUT = 0.5

    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 2e-4

    # CLASS IMBALANCE STRATEGIES - CONFIGURE HERE
    USE_WEIGHTED_SAMPLING = True  # Strategy 1: Recommended!
    USE_FOCAL_LOSS = False  # Strategy 2: Set True to use focal loss
    USE_OVERSAMPLING = True  # Strategy 4: Oversample rare classes
    OVERSAMPLE_TARGET = 15  # Minimum samples per class after oversampling

    # MLflow configuration
    EXPERIMENT_NAME = "SignNetWord"
    RUN_NAME = "transformer_ctc_balanced_training"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.environ['MLFLOW_TRACKING_USERNAME'] = 'roman'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'SignNet'
    mlflow.set_tracking_uri("https://mlflow.schlaepfer.me")

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(log_system_metrics=True, run_name=RUN_NAME):
        # System information
        mlflow.log_param("python_version", platform.python_version())
        mlflow.log_param("pytorch_version", torch.__version__)
        mlflow.log_param("os", platform.system())
        mlflow.log_param("cpu_count", os.cpu_count())
        mlflow.log_param("total_ram_gb", round(psutil.virtual_memory().total / (1024 ** 3), 2))

        if torch.cuda.is_available():
            mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
            mlflow.log_param("gpu_count", torch.cuda.device_count())
            mlflow.log_param("cuda_version", torch.version.cuda)
            mlflow.log_param("cudnn_version", torch.backends.cudnn.version())
            mlflow.log_param("gpu_memory_gb", round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2))

        # Log hyperparameters
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "input_dim": INPUT_DIM,
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
            "scheduler": "LambdaLR_with_warmup",
            "decoder_training": "greedy_fast",
            "decoder_final": "hybrid_beam_search",
            "augmentation": "advanced_temporal",
            "loss_function": "FocalCTC" if USE_FOCAL_LOSS else "CTC",
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "split_seed": SPLIT_SEED,
            "use_weighted_sampling": USE_WEIGHTED_SAMPLING,
            "use_focal_loss": USE_FOCAL_LOSS,
            "use_oversampling": USE_OVERSAMPLING,
            "oversample_target": OVERSAMPLE_TARGET if USE_OVERSAMPLING else None
        })

        mlflow.log_param("device", str(device))
        mlflow.log_param("cuda_available", torch.cuda.is_available())

        # Split dataset
        print("\n" + "=" * 50)
        print("Splitting Dataset")
        print("=" * 50)

        train_samples, val_samples, test_samples = split_dataset(
            LANDMARKS_DIR,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            seed=SPLIT_SEED
        )

        # STRATEGY 4: Oversampling (optional)
        if USE_OVERSAMPLING:
            print("\n" + "=" * 50)
            print("Applying Oversampling Strategy")
            print("=" * 50)
            train_samples = create_oversampled_samples(train_samples, target_min_samples=OVERSAMPLE_TARGET)

        # Build vocabulary from ALL data
        print("\n" + "=" * 50)
        print("Building Vocabulary from ALL Data")
        print("=" * 50)

        all_samples = train_samples + val_samples + test_samples
        vocab_builder_dataset = LandmarkDataset(
            samples_list=all_samples,
            vocab_file=VOCAB_FILE,
            build_vocab=True,
            augment=False
        )

        # Create train/val/test datasets
        print("\n" + "=" * 50)
        print("Loading Train/Val/Test Datasets")
        print("=" * 50)

        train_dataset = LandmarkDataset(
            samples_list=train_samples,
            vocab_file=VOCAB_FILE,
            build_vocab=False,
            augment=True
        )

        val_dataset = LandmarkDataset(
            samples_list=val_samples,
            vocab_file=VOCAB_FILE,
            build_vocab=False,
            augment=False
        )

        test_dataset = LandmarkDataset(
            samples_list=test_samples,
            vocab_file=VOCAB_FILE,
            build_vocab=False,
            augment=False
        )

        mlflow.log_params({
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
            "total_samples": len(train_dataset) + len(val_dataset) + len(test_dataset),
            "vocab_size": len(train_dataset.gloss_vocab)
        })

        mlflow.log_artifact(VOCAB_FILE)

        use_pin_memory = torch.cuda.is_available()

        # STRATEGY 1: Weighted Sampling (optional)
        if USE_WEIGHTED_SAMPLING:
            print("\n" + "=" * 50)
            print("Creating Balanced Sampler (Strategy 1)")
            print("=" * 50)

            train_sampler, class_counts = create_balanced_sampler(train_dataset)

            mlflow.log_params({
                "num_classes": len(class_counts),
                "min_class_samples": min(class_counts.values()),
                "max_class_samples": max(class_counts.values()),
                "avg_samples_per_class": np.mean(list(class_counts.values()))
            })

            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                sampler=train_sampler,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=use_pin_memory
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=use_pin_memory
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=use_pin_memory
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=use_pin_memory
        )

        # Create model
        print("\n" + "=" * 50)
        print("Creating Model")
        print("=" * 50)

        num_glosses = len(train_dataset.gloss_vocab)
        model = SignLanguageTranslator(
            input_dim=INPUT_DIM,
            num_glosses=num_glosses,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_encoder_layers=NUM_LAYERS,
            dropout=DROPOUT
        ).to(device)

        summary_str, summary_dict = generate_model_summary(
            model,
            INPUT_DIM,
            device,
            batch_size=BATCH_SIZE,
            seq_length=100
        )

        mlflow.log_params(summary_dict)

        model_summary_path = "model_summary.txt"
        with open(model_summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_str)
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("DETAILED STATISTICS\n")
            f.write("=" * 70 + "\n")
            for key, value in summary_dict.items():
                f.write(f"{key}: {value}\n")
        mlflow.log_artifact(model_summary_path)

        # Train model
        print("\n" + "=" * 50)
        print("Starting Training")
        print("=" * 50)

        best_wer_greedy = train_model(
            model,
            train_loader,
            val_loader,
            NUM_EPOCHS,
            device,
            train_dataset.gloss_vocab,
            train_dataset.idx_to_gloss
        )

        print(f"\n{'=' * 50}")
        print(f"Training Complete! Best Greedy WER: {best_wer_greedy:.4f}")
        print(f"{'=' * 50}")

        # Load best model
        checkpoint = torch.load(os.path.join('checkpoints', 'best_model.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded best model for final evaluation")

        # Final evaluation with beam search
        print("\n" + "=" * 70)
        print("FINAL VALIDATION EVALUATION (Beam Search)")
        print("=" * 70)

        val_loss_beam, val_wer_beam = evaluate_with_beam_search(
            model,
            val_loader,
            CTCLoss(blank=1, zero_infinity=True),
            device,
            train_dataset.gloss_vocab,
            train_dataset.idx_to_gloss
        )

        print("\n" + "=" * 70)
        print("FINAL TEST EVALUATION (Beam Search)")
        print("=" * 70)

        test_loss_beam, test_wer_beam = evaluate_with_beam_search(
            model,
            test_loader,
            CTCLoss(blank=1, zero_infinity=True),
            device,
            train_dataset.gloss_vocab,
            train_dataset.idx_to_gloss
        )

        # Log final metrics
        mlflow.log_metric("final_best_wer_greedy", best_wer_greedy)
        mlflow.log_metric("final_val_wer_beam_search", val_wer_beam)
        mlflow.log_metric("final_test_wer_beam_search", test_wer_beam)
        mlflow.log_metric("val_wer_improvement_percent",
                          ((best_wer_greedy - val_wer_beam) / best_wer_greedy) * 100)

        print(f"\n{'=' * 70}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'=' * 70}")
        print(f"Best Validation WER (Greedy):      {best_wer_greedy:.4f}")
        print(f"Final Validation WER (Beam):       {val_wer_beam:.4f}")
        print(f"Final Test WER (Beam):             {test_wer_beam:.4f}")
        print(f"Improvement (Val Greedy→Beam):     {((best_wer_greedy - val_wer_beam) / best_wer_greedy) * 100:.2f}%")
        print(f"{'=' * 70}\n")

        # Save model
        mlflow.pytorch.log_model(model, "model")

        mlflow.set_tags({
            "model_type": "transformer",
            "task": "sign_language_translation",
            "dataset": "custom_word_level",
            "decoder_training": "greedy",
            "decoder_final": "hybrid_beam_search",
            "class_balancing": "weighted_sampling" if USE_WEIGHTED_SAMPLING else "none",
            "status": "completed"
        })


if __name__ == "__main__":
    main()
