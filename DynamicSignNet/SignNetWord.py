import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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


# ==================== DATASET ====================

class LandmarkDataset(Dataset):
    """Dataset loader for preprocessed landmarks"""

    def __init__(self, landmarks_dir, vocab_file=None, build_vocab=False,
                 augment=False, all_dirs_for_vocab=None, max_samples=None, random_subset=False):
        self.landmarks_dir = landmarks_dir
        self.samples = sorted(glob.glob(os.path.join(landmarks_dir, "*.npz")))
        self.augment = augment

        # Initialize augmentation
        if self.augment:
            self.temporal_aug = AdvancedTemporalAugmentation(prob=0.8)

        # Limit samples
        if max_samples is not None:
            if random_subset:
                random.seed(42)
                self.samples = random.sample(self.samples,
                                             min(max_samples, len(self.samples)))
                print(f"Random subset: {len(self.samples)} samples (seed=42)")
            else:
                self.samples = self.samples[:max_samples]
                print(f"First {len(self.samples)} samples")

        if build_vocab:
            if all_dirs_for_vocab:
                self.gloss_vocab = self._build_vocab_from_multiple_dirs(all_dirs_for_vocab)
            else:
                self.gloss_vocab = self._build_vocab()
            self._save_vocab(vocab_file)
        else:
            self.gloss_vocab = self._load_vocab(vocab_file)

        self.idx_to_gloss = {v: k for k, v in self.gloss_vocab.items()}

    def _build_vocab_from_multiple_dirs(self, directories):
        """Build vocabulary from multiple directories (train, dev, test)"""
        glosses = set()
        for directory in directories:
            samples = sorted(glob.glob(os.path.join(directory, "*.npz")))
            for sample_path in tqdm(samples, desc=f"Building vocab from {directory}"):
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

        print(f"Vocabulary size: {len(gloss_to_idx)} (from {len(directories)} directories)")
        return gloss_to_idx

    def _build_vocab(self):
        """Build vocabulary from all glosses in dataset"""
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

        # Apply augmentation during training
        if self.augment:
            landmarks = self.temporal_aug(landmarks)

        landmarks = torch.FloatTensor(landmarks)

        # Optional: convert handedness strings to numeric encoding
        # e.g., LEFT=0, RIGHT=1, NONE=2
        if handedness is not None:
            mapping = {'LEFT': 0, 'RIGHT': 1, 'NONE': 2}
            hand_enc = np.vectorize(lambda x: mapping.get(x, 2))(handedness)
            handedness = torch.LongTensor(hand_enc)
        else:
            handedness = torch.LongTensor(np.zeros((landmarks.shape[0], 2)))  # default NONE

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
    padded_handedness = torch.full((len(landmarks), max_len, 2), 2, dtype=torch.long)  # Pad with 'NONE' = 2

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
            nn.Linear(input_dim + 6, d_model),  # input_dim=1659, +6 for handedness one-hot
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

        x = torch.cat([src, handedness_onehot], dim=2)  # 1659 + 6 = 1665
        x = self.input_proj(x)  # Projects from 1665 to d_model

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

    def __call__(self, landmarks):
        if np.random.random() > self.prob:
            return landmarks

        # Speed variation (aggressive)
        speed = np.random.uniform(0.65, 1.35)
        num_frames = landmarks.shape[0]
        new_num_frames = max(1, int(num_frames / speed))
        indices = np.linspace(0, num_frames - 1, new_num_frames)
        landmarks = np.array([landmarks[min(int(i), num_frames - 1)] for i in indices])

        # Stronger spatial noise
        noise = np.random.normal(0, 0.015, landmarks.shape)
        landmarks = landmarks + noise

        # Random frame dropout (remove 5-10% of frames randomly)
        if np.random.random() < 0.3:
            keep_ratio = np.random.uniform(0.9, 0.95)
            mask = np.random.rand(landmarks.shape[0]) < keep_ratio
            if mask.sum() > 0:
                landmarks = landmarks[mask]

        return landmarks


# ==================== DECODING ====================

def decode_predictions_greedy(ctc_output, lengths, idx_to_gloss, blank_id=1):
    """
    Fast greedy CTC decoding (original approach).
    Use this during training for speed.
    """
    batch_size = ctc_output.size(0)
    predictions = torch.argmax(ctc_output, dim=-1)

    decoded = []
    for i in range(batch_size):
        pred = predictions[i, :lengths[i]]

        # Remove blanks and repeated tokens
        pred_seq = []
        prev = None
        for p in pred:
            p = p.item()
            if p != blank_id and p != prev:
                pred_seq.append(p)
            prev = p

        # Convert to glosses
        glosses = [idx_to_gloss.get(p, '<unk>') for p in pred_seq]
        decoded.append(glosses)

    return decoded


def decode_predictions_hybrid(ctc_output, lengths, idx_to_gloss, blank_id=1,
                              beam_width=30, confidence_threshold=0.4, use_length_penalty=True):
    """
    Hybrid decoding: Beam search + confidence filtering + length penalty.
    Best balance between accuracy and efficiency (use for final evaluation only).

    Args:
        ctc_output: [B, T, C] tensor of CTC logits
        lengths: [B] tensor of sequence lengths
        idx_to_gloss: dict mapping index to gloss string
        blank_id: ID of blank token (default 1)
        beam_width: number of beams to keep (default 30)
        confidence_threshold: minimum confidence to keep prediction (default 0.4)
        use_length_penalty: whether to penalize very short sequences (default True)
    """
    batch_size = ctc_output.size(0)
    log_probs = F.log_softmax(ctc_output, dim=-1)
    max_probs = torch.max(F.softmax(ctc_output, dim=-1), dim=-1)[0]

    decoded = []

    for batch_idx in range(batch_size):
        seq_len = lengths[batch_idx]
        probs = log_probs[batch_idx, :seq_len, :]
        conf = max_probs[batch_idx, :seq_len]

        # Beam: (prefix_tuple, score, length) - always use tuple for prefix
        beam = [(tuple(), 0.0, 0)]

        for t in range(len(probs)):
            new_beam = {}

            # Skip low-confidence frames
            if conf[t].item() < confidence_threshold:
                # Keep the prefix with small penalty
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

                        # Don't add consecutive duplicates
                        if len(prefix) > 0 and prefix[-1] == gloss:
                            new_prefix = prefix
                            new_length = length
                        else:
                            new_prefix = prefix + (gloss,)
                            new_length = length + 1

                    # Score components
                    ctc_score = probs[t, c_idx].item()
                    length_penalty = 0.0

                    if use_length_penalty:
                        # Penalize very short sequences
                        if new_length == 0:
                            length_penalty = -0.1

                    new_score = score + ctc_score + length_penalty

                    key = (new_prefix, new_length)
                    if key not in new_beam or new_beam[key] < new_score:
                        new_beam[key] = new_score

            # Keep top beams
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
    # FIX: Add handedness to unpacking
    for batch_idx, (landmarks, landmark_lengths, handedness, glosses, gloss_lengths) in enumerate(pbar):
        landmarks = landmarks.to(device)
        handedness = handedness.to(device)  # Add this line
        glosses = glosses.to(device)
        landmark_lengths = landmark_lengths.to(device)
        gloss_lengths = gloss_lengths.to(device)

        optimizer.zero_grad()

        # Forward pass - FIX: Pass handedness
        ctc_logits = model(landmarks, handedness, landmark_lengths)

        # CTC loss expects [T, B, C] format and log probabilities
        ctc_logits = ctc_logits.transpose(0, 1)
        log_probs = F.log_softmax(ctc_logits, dim=-1)

        # Calculate CTC loss
        loss = criterion(log_probs, glosses, landmark_lengths, gloss_lengths)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Log batch loss to MLflow
        if batch_idx % 50 == 0:
            step = epoch * len(train_loader) + batch_idx
            mlflow.log_metric("batch_loss", loss.item(), step=step)

    return total_loss / num_batches



def validate(model, val_loader, criterion, device, vocab, idx_to_gloss, use_beam_search=False):
    """
    Validate the model.
    use_beam_search=False for fast training validation (greedy)
    use_beam_search=True for final evaluation (hybrid beam search)
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        # FIX: Add handedness to unpacking
        for landmarks, landmark_lengths, handedness, glosses, gloss_lengths in pbar:
            landmarks = landmarks.to(device)
            handedness = handedness.to(device)  # Add this line
            glosses = glosses.to(device)
            landmark_lengths = landmark_lengths.to(device)
            gloss_lengths = gloss_lengths.to(device)

            # Forward pass - FIX: Pass handedness
            ctc_logits = model(landmarks, handedness, landmark_lengths)

            # Calculate loss
            ctc_logits_t = ctc_logits.transpose(0, 1)
            log_probs = F.log_softmax(ctc_logits_t, dim=-1)
            loss = criterion(log_probs, glosses, landmark_lengths, gloss_lengths)

            total_loss += loss.item()
            num_batches += 1

            # Choose decoder based on setting
            if use_beam_search:
                # Slow but more accurate (for final evaluation)
                predictions = decode_predictions_hybrid(
                    ctc_logits,
                    landmark_lengths,
                    idx_to_gloss,
                    blank_id=1,
                    beam_width=30,
                    confidence_threshold=0.4,
                    use_length_penalty=True
                )
            else:
                # Fast greedy decoder (for training validation)
                predictions = decode_predictions_greedy(
                    ctc_logits,
                    landmark_lengths,
                    idx_to_gloss,
                    blank_id=1
                )

            # Convert targets to glosses
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
    """Full training loop with fixed scheduler and fast greedy validation"""
    os.makedirs(save_dir, exist_ok=True)

    criterion = CTCLoss(blank=1, zero_infinity=True)

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

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, vocab, epoch, scheduler)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate with GREEDY decoder (fast) during training
        val_loss, val_wer = validate(model, val_loader, criterion, device, vocab, idx_to_gloss, use_beam_search=False)
        print(f"Val Loss: {val_loss:.4f}, Val WER: {val_wer:.4f} (greedy decoder - fast)")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Log metrics
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_wer": val_wer,
            "learning_rate": optimizer.param_groups[0]['lr']
        }, step=epoch)

        # Save best model based on greedy WER
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

        # Early stopping
        if early_stopping(val_wer, epoch):
            break

    return best_wer


def evaluate_with_beam_search(model, val_loader, criterion, device, vocab, idx_to_gloss):
    """
    Final evaluation with slow beam search decoder.
    Run this AFTER training to get accurate WER with hybrid decoder.
    """
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

    # Create dummy input - landmarks only (handedness added in model)
    dummy_input = torch.randn(batch_size, seq_length, input_dim).to(device)
    dummy_lengths = torch.tensor([seq_length] * batch_size).to(device)
    dummy_handedness = torch.randint(0, 3, (batch_size, seq_length, 2)).to(device)

    # Get torchinfo summary
    model_stats = summary(
        model,
        input_data=(dummy_input, dummy_handedness, dummy_lengths),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
        verbose=0
    )

    print(model_stats)

    # Additional statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 ** 2

    # FIX: Pass all 3 required arguments to model
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



# ==================== MAIN ====================

def main():
    # Configuration
    LANDMARKS_TRAIN = "./landmarks_train"
    LANDMARKS_DEV = "./landmarks_dev"
    VOCAB_FILE = "vocab.json"
    BATCH_SIZE = 16
    NUM_EPOCHS = 150
    # Handedness one-hot: 2 hands × 3 classes = 6 features per frame
    INPUT_DIM = 1659  # 126 hand + 1434 face + 99 pose (handedness is separate)

    # Smaller model with aggressive regularization
    D_MODEL = 384
    NHEAD = 8
    NUM_LAYERS = 6
    DROPOUT = 0.5

    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 2e-4

    # MLflow configuration
    EXPERIMENT_NAME = "SignNetWord"
    RUN_NAME = "transformer_ctc_fast_greedy_training"

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

        # GPU details
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
            "loss_function": "CTC"
        })

        mlflow.log_param("device", str(device))
        mlflow.log_param("cuda_available", torch.cuda.is_available())

        # Create datasets
        print("\n" + "=" * 50)
        print("Loading Datasets")
        print("=" * 50)

        train_dataset = LandmarkDataset(
            LANDMARKS_TRAIN,
            vocab_file=VOCAB_FILE,
            build_vocab=True,
            augment=True
        )

        val_dataset = LandmarkDataset(
            LANDMARKS_DEV,
            vocab_file=VOCAB_FILE,
            build_vocab=False,
            augment=False
        )

        mlflow.log_params({
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "vocab_size": len(train_dataset.gloss_vocab)
        })

        mlflow.log_artifact(VOCAB_FILE)

        use_pin_memory = torch.cuda.is_available()
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

        # Train model with FAST greedy decoder
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

        mlflow.log_metric("final_best_wer", best_wer)
        # Load best model
        checkpoint = torch.load(os.path.join('checkpoints', 'best_model.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded best model for final evaluation")

        # Final evaluation with SLOW hybrid beam search decoder
        val_loss_beam, val_wer_beam = evaluate_with_beam_search(
            model,
            val_loader,
            CTCLoss(blank=1, zero_infinity=True),
            device,
            train_dataset.gloss_vocab,
            train_dataset.idx_to_gloss
        )

        mlflow.log_metric("final_best_wer_greedy", best_wer_greedy)
        mlflow.log_metric("final_val_wer_beam_search", val_wer_beam)
        mlflow.log_metric("wer_improvement_percent", ((best_wer_greedy - val_wer_beam) / best_wer_greedy) * 100)

        mlflow.pytorch.log_model(model, "model")

        mlflow.set_tags({
            "model_type": "transformer",
            "task": "sign_language_translation",
            "dataset": "phoenix-2014",
            "decoder_training": "greedy",
            "decoder_final": "hybrid_beam_search",
            "status": "completed"
        })


if __name__ == "__main__":
    main()
