import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
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
                 augment=False, all_dirs_for_vocab=None, max_samples=None, 
                 random_subset=False, seed=42, compute_stats=False, stats_file=None):
        self.landmarks_dir = landmarks_dir
        self.samples = sorted(glob.glob(os.path.join(landmarks_dir, "*.npz")))
        self.augment = augment
        self.seed = seed

        # Initialize augmentation with stronger parameters
        if self.augment:
            self.temporal_aug = TemporalAugmentation(speed_range=(0.7, 1.3), prob=0.5)

        # Limit samples
        if max_samples is not None:
            if random_subset:
                random.seed(seed)
                self.samples = random.sample(self.samples, 
                                           min(max_samples, len(self.samples)))
                print(f"Random subset: {len(self.samples)} samples (seed={seed})")
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
        
        # Compute normalization statistics
        if compute_stats:
            self.mean, self.std = self._compute_normalization_stats()
            self._save_stats(stats_file)
        elif stats_file and os.path.exists(stats_file):
            self.mean, self.std = self._load_stats(stats_file)
        else:
            print("Warning: No normalization stats provided. Using zero mean and unit std.")
            self.mean = 0.0
            self.std = 1.0
        
    def _compute_normalization_stats(self):
        """Compute mean and std for landmark normalization"""
        print("Computing normalization statistics...")
        all_landmarks = []
        
        # Sample subset for efficiency (use 10% of data or max 1000 samples)
        sample_size = min(len(self.samples), max(100, len(self.samples) // 10))
        sampled_files = random.sample(self.samples, sample_size)
        
        for sample_path in tqdm(sampled_files, desc="Computing stats"):
            data = np.load(sample_path, allow_pickle=True)
            landmarks = data['landmarks'].astype(np.float32)
            
            # NEW: Apply relative/scale normalization before computing stats
            landmarks = self._normalize_landmarks(landmarks)
            
            all_landmarks.append(landmarks)
        
        # Concatenate all landmarks
        all_landmarks = np.concatenate(all_landmarks, axis=0)
        
        # Compute statistics
        mean = np.mean(all_landmarks, axis=0)
        std = np.std(all_landmarks, axis=0)
        
        # Avoid division by zero
        std = np.where(std < 1e-6, 1.0, std)
        
        print(f"Computed statistics from {sample_size} samples")
        print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
        
        return mean, std
    
    def _save_stats(self, stats_file):
        """Save normalization statistics"""
        if stats_file:
            np.savez(stats_file, mean=self.mean, std=self.std)
            print(f"Normalization stats saved to {stats_file}")
    
    def _load_stats(self, stats_file):
        """Load normalization statistics"""
        data = np.load(stats_file)
        mean = data['mean']
        std = data['std']
        print(f"Normalization stats loaded from {stats_file}")
        return mean, std
        
    def _build_vocab_from_multiple_dirs(self, directories):
        """Build vocabulary from multiple directories"""
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
        
    def _normalize_landmarks(self, landmarks):
        """
        Complete preprocessing pipeline for MediaPipe landmarks
        landmarks: [num_frames, 1659] flattened array
        
        Landmark structure (YOUR specific order):
        - Hands (both): 0-125 (126 features = 2 hands × 21 landmarks × 3)
        - Face: 126-1559 (1434 features = 478 landmarks × 3)
        - Pose: 1560-1658 (99 features = 33 landmarks × 3)
        """
        num_frames = landmarks.shape[0]
        feature_dim = landmarks.shape[1]
        
        # Reshape to [num_frames, num_landmarks, 3]
        num_landmarks = feature_dim // 3  # Should be 553 = (126 + 1434 + 99) / 3
        landmarks_3d = landmarks.reshape(num_frames, num_landmarks, 3)
        
        processed = []
        
        for frame_idx in range(num_frames):
            frame_landmarks = landmarks_3d[frame_idx].copy()
            
            # YOUR landmark order: Hands (both) → Face → Pose
            # Hands: indices 0-41 (42 landmarks = 2 hands × 21)
            hands_start, hands_end = 0, 42  # 126 features / 3 = 42 landmarks
            # Face: indices 42-519 (478 landmarks)
            face_start, face_end = 42, 520  # 42 + 478 = 520
            # Pose: indices 520-552 (33 landmarks)
            pose_start, pose_end = 520, 553  # 520 + 33 = 553
            
            # Step 1: Relative positioning
            
            # === HANDS (both stored together) ===
            # Left hand: landmarks 0-20, Right hand: landmarks 21-41
            hands = frame_landmarks[hands_start:hands_end].copy()
            
            # Check if hands are detected (not all zeros)
            if not np.all(np.isclose(hands, 0, atol=1e-6)):
                # Split into left and right hand
                left_hand = hands[0:21].copy()
                right_hand = hands[21:42].copy()
                
                # Normalize left hand relative to left wrist (index 0)
                if not np.all(np.isclose(left_hand, 0, atol=1e-6)):
                    left_hand = left_hand - left_hand[0]  # Relative to wrist
                
                # Normalize right hand relative to right wrist (index 0 within right hand)
                if not np.all(np.isclose(right_hand, 0, atol=1e-6)):
                    right_hand = right_hand - right_hand[0]  # Relative to wrist
                
                # Recombine
                hands = np.concatenate([left_hand, right_hand])
                frame_landmarks[hands_start:hands_end] = hands
            
            # === FACE ===
            face = frame_landmarks[face_start:face_end].copy()
            if not np.all(np.isclose(face, 0, atol=1e-6)):
                # Normalize relative to nose tip (landmark 1 in face mesh)
                # With refined landmarks, nose tip is still at index 1
                reference = face[1].copy()  # Nose tip
                face = face - reference
                frame_landmarks[face_start:face_end] = face
            
            # === POSE ===
            pose = frame_landmarks[pose_start:pose_end].copy()
            if not np.all(np.isclose(pose, 0, atol=1e-6)):
                # Normalize relative to mid-shoulder point
                # MediaPipe Pose: landmark 11 = left shoulder, 12 = right shoulder
                left_shoulder = pose[11].copy()
                right_shoulder = pose[12].copy()
                
                # Use midpoint of shoulders as reference
                reference = (left_shoulder + right_shoulder) / 2.0
                pose = pose - reference
                frame_landmarks[pose_start:pose_end] = pose
            
            # Step 2: Scale normalization for each part independently
            parts = [
                (hands_start, hands_end, "hands"),
                (face_start, face_end, "face"),
                (pose_start, pose_end, "pose")
            ]
            
            for start, end, name in parts:
                part = frame_landmarks[start:end].copy()
                
                # Use L2 norm (Euclidean distance) for scale normalization
                # Compute distance of each landmark from origin (after centering)
                distances = np.linalg.norm(part, axis=-1)
                max_dist = np.max(distances)
                
                if max_dist > 1e-6:
                    part = part / max_dist
                    frame_landmarks[start:end] = part
            
            processed.append(frame_landmarks)
        
        processed = np.array(processed)
        
        # Reshape back to [num_frames, 1659]
        processed = processed.reshape(num_frames, -1)
        
        return processed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx], allow_pickle=True)
        landmarks = data['landmarks'].astype(np.float32)

        # Apply relative + scale normalization FIRST
        landmarks = self._normalize_landmarks(landmarks)
    
        # Apply normalization
        landmarks = (landmarks - self.mean) / self.std

        # Apply augmentation during training
        if self.augment:
            # Temporal augmentation
            landmarks = self.temporal_aug(landmarks)

            # Stronger spatial noise
            if np.random.random() > 0.5:
                landmarks = add_spatial_noise(landmarks, noise_std=0.015)
            
            # Random masking (mask some frames)
            if np.random.random() > 0.7:
                landmarks = random_masking(landmarks, mask_prob=0.1)

        landmarks = torch.FloatTensor(landmarks)
        
        # Handle unknown glosses safely
        glosses = []
        for g in data['glosses']:
            g_str = str(g)
            glosses.append(self.gloss_vocab.get(g_str, self.gloss_vocab['<unk>']))
        
        return landmarks, torch.LongTensor(glosses)


def collate_fn(batch):
    """Collate function for variable-length sequences"""
    landmarks, glosses = zip(*batch)

    # Pad landmarks to max length in batch
    max_len = max(lm.shape[0] for lm in landmarks)
    feature_dim = landmarks[0].shape[1]
    padded_landmarks = torch.zeros(len(landmarks), max_len, feature_dim)
    landmark_lengths = []

    for i, lm in enumerate(landmarks):
        padded_landmarks[i, :lm.shape[0]] = lm
        landmark_lengths.append(lm.shape[0])

    # Pad glosses
    max_gloss_len = max(len(g) for g in glosses)
    padded_glosses = torch.zeros(len(glosses), max_gloss_len).long()
    gloss_lengths = []

    for i, g in enumerate(glosses):
        padded_glosses[i, :len(g)] = g
        gloss_lengths.append(len(g))

    return (padded_landmarks, torch.LongTensor(landmark_lengths),
            padded_glosses, torch.LongTensor(gloss_lengths))


# ==================== MODEL ====================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = math.sqrt(d_model)  # Scale to prevent positional encoding from dominating

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x * self.scale + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SignLanguageTranslator(nn.Module):
    """Improved Transformer-based Sign Language Translation Model"""
    def __init__(self, input_dim, num_glosses, d_model=512, nhead=8,
                 num_encoder_layers=8, dropout=0.1, dim_feedforward=2048):
        super().__init__()

        # Input projection with residual connection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # Enhanced temporal convolution with multiple kernel sizes
        # Each conv outputs d_model//3, but we need to ensure they sum to d_model
        conv_channels = [d_model // 3, d_model // 3, d_model - 2 * (d_model // 3)]  # Ensures exact sum
        self.temporal_conv = nn.ModuleList([
            nn.Conv1d(d_model, conv_channels[i], kernel_size=k, padding=k//2)
            for i, k in enumerate([3, 5, 7])
        ])
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv_dropout = nn.Dropout(dropout)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder with more layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',  # GELU instead of ReLU
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Enhanced CTC head
        self.ctc_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_glosses)
        )

        self.d_model = d_model
        self._init_weights()

    def _init_weights(self):
        """Initialize weights properly"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_lengths):
        # Input projection: [B, T, input_dim] -> [B, T, d_model]
        x = self.input_proj(src)

        # Multi-scale temporal convolution
        x_t = x.transpose(1, 2)  # [B, d_model, T]
        conv_outputs = [conv(x_t) for conv in self.temporal_conv]
        x_conv = torch.cat(conv_outputs, dim=1).transpose(1, 2)  # [B, T, d_model]
        
        # Residual connection with normalization
        x = x + self.conv_dropout(self.conv_norm(x_conv))

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

class TemporalAugmentation:
    """Enhanced temporal data augmentation"""
    def __init__(self, speed_range=(0.7, 1.3), prob=0.5):
        self.speed_range = speed_range
        self.prob = prob

    def __call__(self, landmarks):
        if np.random.random() > self.prob:
            return landmarks

        # Random temporal scaling
        speed = np.random.uniform(*self.speed_range)
        num_frames = landmarks.shape[0]
        new_num_frames = max(1, int(num_frames / speed))

        indices = np.linspace(0, num_frames - 1, new_num_frames)
        augmented = np.array([landmarks[min(int(i), num_frames-1)] for i in indices])

        return augmented


def add_spatial_noise(landmarks, noise_std=0.015):
    """Add stronger Gaussian noise to landmarks"""
    noise = np.random.normal(0, noise_std, landmarks.shape)
    return landmarks + noise


def random_masking(landmarks, mask_prob=0.1):
    """Randomly mask some frames"""
    num_frames = landmarks.shape[0]
    mask = np.random.random(num_frames) > mask_prob
    # Keep at least one frame
    if not mask.any():
        mask[0] = True
    return landmarks[mask]


# ==================== BEAM SEARCH DECODER ====================

class CTCBeamDecoder:
    """Simple beam search decoder for CTC"""
    def __init__(self, blank_id=1, beam_width=10):
        self.blank_id = blank_id
        self.beam_width = beam_width
    
    def decode(self, log_probs, lengths):
        """
        Decode using beam search
        Args:
            log_probs: [B, T, num_classes] log probabilities
            lengths: [B] sequence lengths
        Returns:
            List of decoded sequences
        """
        batch_size = log_probs.size(0)
        decoded = []
        
        for b in range(batch_size):
            seq_len = lengths[b].item()
            probs = log_probs[b, :seq_len].cpu().numpy()
            
            # Beam search
            beam = [([self.blank_id], 0.0)]  # (sequence, score)
            
            for t in range(seq_len):
                new_beam = []
                
                for seq, score in beam:
                    for c in range(probs.shape[1]):
                        new_score = score + probs[t, c]
                        
                        # Extend sequence
                        if c == self.blank_id:
                            new_seq = seq
                        elif len(seq) > 0 and c == seq[-1]:
                            # Repeated character
                            new_seq = seq
                        else:
                            new_seq = seq + [c]
                        
                        new_beam.append((new_seq, new_score))
                
                # Keep top beam_width
                new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:self.beam_width]
                beam = new_beam
            
            # Get best sequence
            best_seq = beam[0][0]
            # Remove blanks
            best_seq = [c for c in best_seq if c != self.blank_id]
            decoded.append(best_seq)
        
        return decoded


def decode_predictions_beam(ctc_output, lengths, vocab, blank_id=1, beam_width=10):
    """Decode CTC output using beam search"""
    log_probs = F.log_softmax(ctc_output, dim=-1)
    decoder = CTCBeamDecoder(blank_id=blank_id, beam_width=beam_width)
    
    predictions = decoder.decode(log_probs, lengths)
    
    # Convert to glosses
    decoded = []
    for pred_seq in predictions:
        glosses = [vocab.get(p, '<unk>') for p in pred_seq]
        decoded.append(glosses)
    
    return decoded


# ==================== TRAINING ====================

def decode_predictions(ctc_output, lengths, vocab, blank_id=1):
    """Decode CTC output using greedy decoding (fallback)"""
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
        glosses = [vocab.get(p, '<unk>') for p in pred_seq]
        decoded.append(glosses)

    return decoded


def compute_wer(predictions, targets):
    """Compute Word Error Rate"""
    errors = 0
    total_words = 0

    for pred, tgt in zip(predictions, targets):
        # Simple edit distance calculation
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


def train_epoch(model, train_loader, optimizer, criterion, device, vocab, epoch, scheduler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (landmarks, landmark_lengths, glosses, gloss_lengths) in enumerate(pbar):
        landmarks = landmarks.to(device)
        glosses = glosses.to(device)
        landmark_lengths = landmark_lengths.to(device)
        gloss_lengths = gloss_lengths.to(device)

        optimizer.zero_grad()

        # Forward pass
        ctc_logits = model(landmarks, landmark_lengths)

        # CTC loss expects [T, B, C] format and log probabilities
        ctc_logits = ctc_logits.transpose(0, 1)
        log_probs = F.log_softmax(ctc_logits, dim=-1)

        # Calculate CTC loss
        loss = criterion(log_probs, glosses, landmark_lengths, gloss_lengths)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Step scheduler if provided (for warmup)
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
        
        # Log batch loss to MLflow every 50 batches
        if batch_idx % 50 == 0:
            step = epoch * len(train_loader) + batch_idx
            mlflow.log_metric("batch_loss", loss.item(), step=step)

    return total_loss / num_batches


def validate(model, val_loader, criterion, device, vocab, idx_to_gloss, use_beam_search=False, beam_width=10):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for landmarks, landmark_lengths, glosses, gloss_lengths in pbar:
            landmarks = landmarks.to(device)
            glosses = glosses.to(device)
            landmark_lengths = landmark_lengths.to(device)
            gloss_lengths = gloss_lengths.to(device)

            # Forward pass
            ctc_logits = model(landmarks, landmark_lengths)

            # Calculate loss
            ctc_logits_t = ctc_logits.transpose(0, 1)
            log_probs = F.log_softmax(ctc_logits_t, dim=-1)
            loss = criterion(log_probs, glosses, landmark_lengths, gloss_lengths)

            total_loss += loss.item()
            num_batches += 1

            # Decode predictions with beam search
            if use_beam_search:
                predictions = decode_predictions_beam(ctc_logits, landmark_lengths, idx_to_gloss, 
                                                     blank_id=1, beam_width=beam_width)
            else:
                predictions = decode_predictions(ctc_logits, landmark_lengths, idx_to_gloss)

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


def train_model(model, train_loader, val_loader, num_epochs, device, vocab, idx_to_gloss, 
                save_dir='checkpoints', beam_width=10):
    """Full training loop with MLflow tracking"""
    os.makedirs(save_dir, exist_ok=True)

    criterion = CTCLoss(blank=1, zero_infinity=True)
    
    # Lower learning rate with AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.98), 
                                  eps=1e-9, weight_decay=0.01)
    
    # Warmup scheduler
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = num_training_steps // 10  # 10% warmup
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Plateau scheduler for after warmup
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    best_wer = float('inf')
    patience_counter = 0
    max_patience = 10

    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, vocab, epoch, warmup_scheduler)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate with beam search
        val_loss, val_wer = validate(model, val_loader, criterion, device, vocab, idx_to_gloss, 
                                     use_beam_search=True, beam_width=beam_width)
        print(f"Val Loss: {val_loss:.4f}, Val WER: {val_wer:.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_wer", val_wer, step=epoch)
        mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)

        # Plateau scheduling after warmup
        if epoch * len(train_loader) > num_warmup_steps:
            plateau_scheduler.step(val_loss)

        # Save best model
        if val_wer < best_wer:
            best_wer = val_wer
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_wer': val_wer,
            }
            checkpoint_path = os.path.join(save_dir, 'best_model.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved best model with WER: {best_wer:.4f}")
            mlflow.log_metric("best_wer", best_wer)
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_wer': val_wer,
            }
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(checkpoint, checkpoint_path)
            mlflow.log_artifact(checkpoint_path)

    return best_wer


def generate_model_summary(model, input_dim, device, batch_size=8, seq_length=100):
    """Generate comprehensive model summary"""
    print("\n" + "="*70)
    print("MODEL SUMMARY")
    print("="*70)
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, seq_length, input_dim).to(device)
    dummy_lengths = torch.tensor([seq_length] * batch_size).to(device)
    
    # Get torchinfo summary
    model_stats = summary(
        model,
        input_data=(dummy_input, dummy_lengths),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
        verbose=0
    )
    
    # Print summary
    print(model_stats)
    
    # Additional statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    
    print("\n" + "="*70)
    print("DETAILED STATISTICS")
    print("="*70)
    print(f"Total Parameters:           {total_params:>15,}")
    print(f"Trainable Parameters:       {trainable_params:>15,}")
    print(f"Non-trainable Parameters:   {non_trainable_params:>15,}")
    print(f"Model Size:                 {size_mb:>15.2f} MB")
    print(f"Input Shape:                {tuple(dummy_input.shape)}")
    print(f"Output Shape:               {model(dummy_input, dummy_lengths).shape}")
    print("="*70 + "\n")
    
    summary_str = str(model_stats)
    
    summary_dict = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "model_size_mb": size_mb,
        "input_shape": list(dummy_input.shape),
        "output_shape": list(model(dummy_input, dummy_lengths).shape)
    }
    
    return summary_str, summary_dict


# ==================== MAIN ====================

def main():
    # Configuration
    LANDMARKS_TRAIN = "./landmarks_train"
    LANDMARKS_DEV = "./landmarks_dev"
    VOCAB_FILE = "vocab.json"
    STATS_FILE = "normalization_stats.npz"
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    INPUT_DIM = 1659
    D_MODEL = 512
    NHEAD = 8
    NUM_LAYERS = 8  # Increased from 6
    DROPOUT = 0.1
    DIM_FEEDFORWARD = 2048
    BEAM_WIDTH = 10
    SEED = 42
    
    # Set random seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    # MLflow configuration
    EXPERIMENT_NAME = "SignNetAdvanced++"
    RUN_NAME = "transformer_ctc_improved_v2"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.environ['MLFLOW_TRACKING_USERNAME'] = 'roman'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'SignNet'
    mlflow.set_tracking_uri("https://mlflow.schlaepfer.me")
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Start MLflow run
    with mlflow.start_run(log_system_metrics=True, run_name=RUN_NAME):
        # System information
        mlflow.log_param("python_version", platform.python_version())
        mlflow.log_param("pytorch_version", torch.__version__)
        mlflow.log_param("os", platform.system())
        mlflow.log_param("cpu_count", os.cpu_count())
        mlflow.log_param("total_ram_gb", round(psutil.virtual_memory().total / (1024**3), 2))

        # GPU details
        if torch.cuda.is_available():
            mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
            mlflow.log_param("gpu_count", torch.cuda.device_count())
            mlflow.log_param("cuda_version", torch.version.cuda)
            mlflow.log_param("cudnn_version", torch.backends.cudnn.version())
            mlflow.log_param("gpu_memory_gb", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
            
        # Log hyperparameters
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "input_dim": INPUT_DIM,
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "dim_feedforward": DIM_FEEDFORWARD,
            "optimizer": "AdamW",
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "scheduler": "Warmup+Plateau",
            "warmup_ratio": 0.1,
            "augmentation": "temporal+spatial+masking",
            "loss_function": "CTC",
            "beam_width": BEAM_WIDTH,
            "normalization": "z-score",
            "seed": SEED,
            "activation": "gelu",
            "architecture": "pre-norm"
        })
        
        mlflow.log_param("device", str(device))
        mlflow.log_param("cuda_available", torch.cuda.is_available())

        # Create datasets
        print("\n" + "="*50)
        print("Loading Datasets")
        print("="*50)

        # Check if stats file exists
        compute_stats = not os.path.exists(STATS_FILE)

        train_dataset = LandmarkDataset(
            LANDMARKS_TRAIN,
            vocab_file=VOCAB_FILE,
            build_vocab=True,
            augment=True,
            compute_stats=compute_stats,
            stats_file=STATS_FILE,
            seed=SEED
        )

        val_dataset = LandmarkDataset(
            LANDMARKS_DEV,
            vocab_file=VOCAB_FILE,
            build_vocab=False,
            augment=False,
            compute_stats=False,
            stats_file=STATS_FILE,
            seed=SEED
        )
        
        # Log dataset info
        mlflow.log_params({
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "vocab_size": len(train_dataset.gloss_vocab)
        })
        
        # Log artifacts
        mlflow.log_artifact(VOCAB_FILE)
        if os.path.exists(STATS_FILE):
            mlflow.log_artifact(STATS_FILE)

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
        print("\n" + "="*50)
        print("Creating Model")
        print("="*50)

        num_glosses = len(train_dataset.gloss_vocab)
        model = SignLanguageTranslator(
            input_dim=INPUT_DIM,
            num_glosses=num_glosses,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_encoder_layers=NUM_LAYERS,
            dropout=DROPOUT,
            dim_feedforward=DIM_FEEDFORWARD
        ).to(device)

        # Generate and log model summary
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
            f.write("\n\n" + "="*70 + "\n")
            f.write("DETAILED STATISTICS\n")
            f.write("="*70 + "\n")
            for key, value in summary_dict.items():
                f.write(f"{key}: {value}\n")
        mlflow.log_artifact(model_summary_path)

        # Train model
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)

        best_wer = train_model(
            model,
            train_loader,
            val_loader,
            NUM_EPOCHS,
            device,
            train_dataset.gloss_vocab,
            train_dataset.idx_to_gloss,
            beam_width=BEAM_WIDTH
        )

        print(f"\n{'='*50}")
        print(f"Training Complete! Best WER: {best_wer:.4f}")
        print(f"{'='*50}")
        
        mlflow.log_metric("final_best_wer", best_wer)
        mlflow.pytorch.log_model(model, "model")
        
        mlflow.set_tags({
            "model_type": "transformer",
            "task": "sign_language_translation",
            "dataset": "phoenix-2014",
            "status": "completed",
            "improvements": "normalization+beam_search+warmup+gelu+prenorm"
        })


if __name__ == "__main__":
    main()
