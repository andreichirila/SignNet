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
import subprocess
from torch.amp import autocast, GradScaler

# ==================== DATASET ====================

class LandmarkDataset(Dataset):
    """Dataset loader for preprocessed landmarks with attention targets"""
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
        
        landmarks = np.clip(landmarks, -5.0, 5.0)  # Clip outliers to ±5 std deviations

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

# ==================== JOINT CTC/ATTENTION MODEL ====================

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

class TransformerDecoderLayer(nn.Module):
    """Custom decoder layer for sign language translation"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        # Ensure all masks are boolean type if provided
        if tgt_mask is not None and tgt_mask.dtype != torch.bool:
            tgt_mask = tgt_mask.bool()
        if tgt_key_padding_mask is not None and tgt_key_padding_mask.dtype != torch.bool:
            tgt_key_padding_mask = tgt_key_padding_mask.bool()
        if memory_key_padding_mask is not None and memory_key_padding_mask.dtype != torch.bool:
            memory_key_padding_mask = memory_key_padding_mask.bool()
        
        # Self-attention
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        
        # Cross-attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn(tgt2, memory, memory,
                              key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        
        # Feed-forward
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        
        return tgt

class SignLanguageTranslationModel(nn.Module):
    """Joint CTC/Attention Transformer for Sign Language Translation"""
    def __init__(self, input_dim, num_glosses, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dropout=0.1, 
                 dim_feedforward=2048):
        super().__init__()

        self.d_model = d_model
        self.num_glosses = num_glosses
        self.blank_id = 1  # CTC blank token

        # === ENCODER ===
        # Input projection with residual connection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # Enhanced temporal convolution with multiple kernel sizes
        # Each conv outputs d_model//3, but we need to ensure they sum to d_model
        conv_channels = [d_model // 3, d_model // 3, d_model - 2 * (d_model // 3)]
        self.temporal_conv = nn.ModuleList([
            nn.Conv1d(d_model, conv_channels[i], kernel_size=k, padding=k//2)
            for i, k in enumerate([3, 5, 7])
        ])
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv_dropout = nn.Dropout(dropout)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder with pre-norm
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

        # CTC head
        self.ctc_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_glosses)
        )

        # === DECODER ===
        self.decoder_embedding = nn.Embedding(num_glosses, d_model)
        self.decoder_pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection for attention-based prediction
        self.decoder_output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_glosses)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights properly"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_lengths):
        """Encoder forward pass"""
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

        return memory

    def decode(self, tgt, memory, tgt_lengths, src_lengths):
        """Decoder forward pass with teacher forcing"""
        # Embed target sequence [B, S] -> [B, S, d_model]
        tgt_embed = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        
        # Positional encoding for decoder
        tgt_embed = self.decoder_pos_encoder(tgt_embed)
        
        # Generate target mask (causal mask for autoregressive decoding)
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        tgt_key_padding_mask = self._generate_padding_mask(tgt_lengths, tgt.size(1)).to(tgt.device)
        src_key_padding_mask = self._generate_padding_mask(src_lengths, memory.size(1)).to(memory.device)
        
        # Pass through decoder layers
        dec_output = tgt_embed
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, memory, tgt_mask=tgt_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=src_key_padding_mask)
        
        # Generate output logits
        output_logits = self.decoder_output(dec_output)
        
        return output_logits

    def forward(self, src, src_lengths, tgt=None, tgt_lengths=None, training=True):
        """
        Forward pass with joint CTC/Attention
        Args:
            src: [B, T, input_dim] source landmarks
            src_lengths: [B] source sequence lengths
            tgt: [B, S] target gloss sequence (for training)
            tgt_lengths: [B] target sequence lengths
            training: whether in training mode
        """
        # Encode source
        memory = self.encode(src, src_lengths)
        
        # CTC branch (always computed)
        ctc_logits = self.ctc_head(memory)  # [B, T, num_glosses]
        
        # Attention branch (only during training or when target provided)
        attention_logits = None
        if training and tgt is not None:
            attention_logits = self.decode(tgt, memory, tgt_lengths, src_lengths)  # [B, S, num_glosses]
        
        return ctc_logits, attention_logits

    def _generate_padding_mask(self, lengths, max_len):
        """Generate padding mask for variable length sequences - BOOLEAN"""
        batch_size = len(lengths)
        mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths.unsqueeze(1)
        return mask  # Already returns boolean

    def _generate_square_subsequent_mask(self, sz):
        """Generate square mask for sequential decoding - RETURN BOOLEAN"""
        # Create upper triangular matrix of True values
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)
        return mask  # Return boolean mask, not float

# ==================== JOINT LOSS COMPUTATION ====================

class JointCTCLoss(nn.Module):
    """Combined CTC and Attention loss for joint training"""
    def __init__(self, ctc_weight=0.3, attention_weight=0.7, blank_id=1):
        super().__init__()
        self.ctc_criterion = CTCLoss(blank=blank_id, zero_infinity=True)
        self.attention_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.ctc_weight = ctc_weight
        self.attention_weight = attention_weight
        self.blank_id = blank_id

    def forward(self, ctc_logits, attention_logits, targets_ctc, targets_att,
                src_lengths, tgt_lengths):
        """Compute joint loss with float32 for CTC"""
        total_loss = 0.0
        
        # CTC Loss - MUST use float32
        # Convert to float32 if in float16
        ctc_logits_fp32 = ctc_logits.float()
        ctc_logits_t = ctc_logits_fp32.transpose(0, 1)  # [T, B, num_classes]
        ctc_log_probs = F.log_softmax(ctc_logits_t, dim=-1)
        ctc_loss = self.ctc_criterion(ctc_log_probs, targets_ctc, src_lengths, tgt_lengths)
        total_loss += self.ctc_weight * ctc_loss
        
        # Attention Loss (works fine with float16/32)
        if attention_logits is not None:
            batch_size, seq_len, num_classes = attention_logits.shape
            attention_logits_flat = attention_logits.reshape(batch_size * seq_len, num_classes)
            targets_att_flat = targets_att.reshape(batch_size * seq_len)
            valid_mask = targets_att_flat != 0
            
            if valid_mask.sum() > 0:
                att_loss = self.attention_criterion(
                    attention_logits_flat[valid_mask],
                    targets_att_flat[valid_mask]
                )
                total_loss += self.attention_weight * att_loss
        
        return total_loss


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

# ==================== BEAM SEARCH DECODER (CTC) ====================

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

# ==================== ATTENTION DECODER (Inference) ====================

def greedy_decode_attention(model, memory, memory_lengths, max_len=100, 
                           sos_id=2, eos_id=3, pad_id=0, device='cuda'):
    """Greedy decoding for attention decoder during inference"""
    model.eval()
    batch_size = memory.size(0)
    
    # Initialize with SOS token
    ys = torch.ones(batch_size, 1).fill_(sos_id).long().to(device)
    ys_lengths = torch.ones(batch_size).fill_(1).long().to(device)
    
    for i in range(max_len - 1):
        with torch.no_grad():
            # Decode one step
            output = model.decode(ys, memory, ys_lengths, memory_lengths)
            output = output[:, -1, :]  # Take last token prediction [B, num_classes]
            
            # Get next token
            next_token = output.argmax(-1)  # [B]
            
            # Stop if EOS or PAD
            finished = (next_token == eos_id) | (next_token == pad_id)
            next_token[finished] = pad_id
            
            # Append to sequence
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            ys_lengths += 1
            
            # Stop early if all sequences finished
            if finished.all():
                break
    
    return ys

# ==================== EVALUATION ====================

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

def compute_bleu(predictions, targets):
    """Compute BLEU score with smoothing for short sequences"""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Use smoothing function to handle zero n-gram overlaps
    smoothing = SmoothingFunction()
    
    bleu_scores = []
    for pred, tgt in zip(predictions, targets):
        pred_str = ' '.join(pred if isinstance(pred, list) else pred.split())
        tgt_str = ' '.join(tgt if isinstance(tgt, list) else tgt.split())
        
        pred_tokens = pred_str.split()
        tgt_tokens = [tgt_str.split()]  # Wrap in list for sentence_bleu
        
        if len(pred_tokens) > 0 and len(tgt_tokens[0]) > 0:
            # Use method1 smoothing (epsilon smoothing) for low-quality outputs
            # Use lower weights for sign language (BLEU-2 instead of BLEU-4)
            bleu = sentence_bleu(
                tgt_tokens, 
                pred_tokens, 
                weights=(0.5, 0.5, 0, 0),  # BLEU-2 for short sequences
                smoothing_function=smoothing.method1  # Add-epsilon smoothing
            )
            bleu_scores.append(bleu)
    
    return np.mean(bleu_scores) if bleu_scores else 0.0


# ==================== TRAINING ====================
class WarmupScheduler:
    """Learning rate scheduler with linear warmup and cosine decay"""
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.current_step = 0
    
    def step(self):
        """Update learning rate and return current LR"""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup from 0 to base_lr
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay after warmup
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        
        # Update all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self):
        """Get current learning rate"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
        
def train_epoch(model, train_loader, optimizer, criterion, device, vocab, epoch, scheduler=None):
    """Train for one epoch with mixed precision and joint loss"""
    model.train()
    total_loss = 0
    num_batches = 0
    ctc_losses = []
    att_losses = []
    
    scaler = GradScaler(device.type)

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (landmarks, landmark_lengths, glosses, gloss_lengths) in enumerate(pbar):
        landmarks = landmarks.to(device)
        glosses = glosses.to(device)
        landmark_lengths = landmark_lengths.to(device)
        gloss_lengths = gloss_lengths.to(device)

        # Forward pass with mixed precision
        with autocast(device.type):
            # Input to decoder: remove last token (exclude <eos>)
            decoder_input = glosses[:, :-1]
            # Target for decoder: remove first token (exclude <sos>)
            decoder_target = glosses[:, 1:]
            # Adjusted target lengths
            adjusted_tgt_lengths = gloss_lengths - 1
            
            ctc_logits, attention_logits = model(landmarks, landmark_lengths, 
                                               decoder_input,
                                               adjusted_tgt_lengths, 
                                               training=True)
            
            # CTC targets use full glosses
            ctc_targets = glosses
            
            # Compute joint loss with shifted targets for attention
            loss = criterion(ctc_logits, attention_logits, 
                           ctc_targets, decoder_target,  # Use shifted target here
                           landmark_lengths, gloss_lengths)

        # Backward pass
        optimizer.zero_grad()  # Move zero_grad here
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            current_lr = scheduler.step()
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        # Call scheduler AFTER optimizer.step() AND scaler.update()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        
        # Track individual losses
        with torch.no_grad():
            # CTC loss tracking - use float32
            ctc_loss_only = criterion.ctc_criterion(
                F.log_softmax(ctc_logits.float().transpose(0, 1), dim=-1),  # Add .float()
                ctc_targets, landmark_lengths, gloss_lengths
            )
            ctc_losses.append(ctc_loss_only.item())
            
            if attention_logits is not None:
                att_loss_only = criterion.attention_criterion(
                    attention_logits.reshape(-1, attention_logits.size(-1)),
                    decoder_target.reshape(-1)
                )
                att_losses.append(att_loss_only.item())

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ctc': f'{np.mean(ctc_losses[-10:]):.4f}' if ctc_losses else '0.0000',
            'att': f'{np.mean(att_losses[-10:]):.4f}' if att_losses else '0.0000',
            'lr': f'{current_lr:.2e}'
        })
        
        if batch_idx % 20 == 0:
            step = epoch * len(train_loader) + batch_idx
            mlflow.log_metric("batch_loss", loss.item(), step=step)
            mlflow.log_metric("batch_ctc_loss", ctc_loss_only.item(), step=step)
            if attention_logits is not None:
                mlflow.log_metric("batch_att_loss", att_loss_only.item(), step=step)
            log_gpu_stats(step)

    return total_loss / num_batches



def validate(model, val_loader, criterion, device, vocab, idx_to_gloss,
             use_beam_search=False, beam_width=10, joint_eval=True, 
             debug=True, debug_samples=5):
    """
    Validate the model with detailed debugging output

    Args:
        debug: Enable detailed per-sample output
        debug_samples: Number of samples to show detailed output for
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    all_ctc_predictions = []
    all_att_predictions = []
    all_targets = []

    # Track prediction statistics
    debug_info = {
        'correct_ctc': 0,
        'correct_att': 0,
        'partially_correct_ctc': 0,
        'partially_correct_att': 0,
        'total_samples': 0,
        'examples': []
    }

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")

        for batch_idx, (landmarks, landmark_lengths, glosses, gloss_lengths) in enumerate(pbar):
            landmarks = landmarks.to(device)
            glosses = glosses.to(device)
            landmark_lengths = landmark_lengths.to(device)
            gloss_lengths = gloss_lengths.to(device)

            batch_size = landmarks.size(0)

            with autocast(device.type):
                # Prepare decoder inputs
                decoder_input = glosses[:, :-1]
                decoder_target = glosses[:, 1:]
                adjusted_tgt_lengths = gloss_lengths - 1

                # Forward pass
                ctc_logits, attention_logits = model(
                    landmarks, landmark_lengths, 
                    decoder_input, adjusted_tgt_lengths,
                    training=True
                )

                # Compute loss
                ctc_targets = glosses  # CTC uses full glosses
                loss = criterion(ctc_logits, attention_logits, ctc_targets, 
                               decoder_target, landmark_lengths, gloss_lengths)

            total_loss += loss.item()
            num_batches += 1

            # Get predictions
            if use_beam_search:
                ctc_preds = decode_predictions_beam(
                    ctc_logits, landmark_lengths, idx_to_gloss,
                    blank_id=1, beam_width=beam_width
                )
            else:
                ctc_preds = decode_predictions(ctc_logits, landmark_lengths, idx_to_gloss)

            all_ctc_predictions.extend(ctc_preds)

            # Attention predictions (greedy decode)
            if joint_eval and attention_logits is not None:
                memory = model.encode(landmarks, landmark_lengths)
                att_decoded = greedy_decode_attention(
                    model, memory, landmark_lengths,
                    max_len=max(gloss_lengths) + 10,
                    device=device
                )

                att_preds = []
                for i in range(att_decoded.size(0)):
                    seq = att_decoded[i, :gloss_lengths[i]].cpu().numpy()
                    gloss_seq = [idx_to_gloss.get(int(t), '<unk>') for t in seq if t != 0]
                    att_preds.append(gloss_seq)

                all_att_predictions.extend(att_preds)
            else:
                all_att_predictions.extend(ctc_preds)  # Fallback

            # Convert targets
            for i in range(glosses.size(0)):
                target = glosses[i, :gloss_lengths[i]].cpu().numpy()
                target_glosses = [idx_to_gloss.get(int(t), '<unk>') for t in target]
                all_targets.append(target_glosses)

            # ============================================================
            # DEBUGGING OUTPUT - Detailed per-sample analysis
            # ============================================================
            if debug and len(debug_info['examples']) < debug_samples:
                for i in range(min(batch_size, debug_samples - len(debug_info['examples']))):
                    sample_idx = batch_idx * val_loader.batch_size + i

                    target = all_targets[-(batch_size - i)]
                    ctc_pred = ctc_preds[i]
                    att_pred = att_preds[i] if joint_eval else ctc_pred

                    # Check correctness
                    ctc_correct = (ctc_pred == target)
                    att_correct = (att_pred == target)

                    # Check partial correctness (at least 50% matching)
                    ctc_matches = sum(1 for a, b in zip(ctc_pred, target) if a == b)
                    att_matches = sum(1 for a, b in zip(att_pred, target) if a == b)

                    ctc_partial = (ctc_matches / max(len(target), 1)) >= 0.5
                    att_partial = (att_matches / max(len(target), 1)) >= 0.5

                    debug_info['examples'].append({
                        'sample_idx': sample_idx,
                        'target': target,
                        'ctc_pred': ctc_pred,
                        'att_pred': att_pred,
                        'ctc_correct': ctc_correct,
                        'att_correct': att_correct,
                        'ctc_match_rate': ctc_matches / max(len(target), 1),
                        'att_match_rate': att_matches / max(len(target), 1),
                        'frames': landmark_lengths[i].item(),
                        'target_len': gloss_lengths[i].item()
                    })

                    if ctc_correct:
                        debug_info['correct_ctc'] += 1
                    elif ctc_partial:
                        debug_info['partially_correct_ctc'] += 1

                    if att_correct:
                        debug_info['correct_att'] += 1
                    elif att_partial:
                        debug_info['partially_correct_att'] += 1

            debug_info['total_samples'] = len(all_targets)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / num_batches

    # Compute metrics
    ctc_wer = compute_wer(all_ctc_predictions, all_targets)
    att_wer = compute_wer(all_att_predictions, all_targets) if joint_eval else ctc_wer
    bleu_score = compute_bleu(all_att_predictions if joint_eval else all_ctc_predictions, 
                              all_targets)

    # ============================================================
    # PRINT DETAILED DEBUG INFORMATION
    # ============================================================
    if debug:
        print("\n" + "="*80)
        print("VALIDATION DEBUGGING OUTPUT")
        print("="*80)

        print(f"\n📊 Overall Metrics:")
        print(f"   Validation Loss: {avg_loss:.4f}")
        print(f"   CTC WER: {ctc_wer:.4f} ({100*(1-ctc_wer):.2f}% accuracy)")
        print(f"   Attention WER: {att_wer:.4f} ({100*(1-att_wer):.2f}% accuracy)")
        print(f"   BLEU Score: {bleu_score:.4f}")

        print(f"\n✅ Correct Predictions (from {debug_samples} samples):")
        print(f"   CTC: {debug_info['correct_ctc']}/{debug_samples} " +
              f"({100*debug_info['correct_ctc']/debug_samples:.1f}%)")
        print(f"   Attention: {debug_info['correct_att']}/{debug_samples} " +
              f"({100*debug_info['correct_att']/debug_samples:.1f}%)")

        print(f"\n⚠️  Partially Correct (≥50% match):")
        print(f"   CTC: {debug_info['partially_correct_ctc']}/{debug_samples}")
        print(f"   Attention: {debug_info['partially_correct_att']}/{debug_samples}")

        print(f"\n" + "-"*80)
        print("DETAILED SAMPLE ANALYSIS")
        print("-"*80)

        for idx, example in enumerate(debug_info['examples'], 1):
            status_ctc = "✅" if example['ctc_correct'] else ("⚠️" if example['ctc_match_rate'] >= 0.5 else "❌")
            status_att = "✅" if example['att_correct'] else ("⚠️" if example['att_match_rate'] >= 0.5 else "❌")

            print(f"\nSample {idx} (Index: {example['sample_idx']}):")
            print(f"  Frames: {example['frames']}, Target Length: {example['target_len']}")
            print(f"  Frames/Gloss Ratio: {example['frames']/example['target_len']:.2f}")

            print(f"\n  Target:     {' '.join(example['target'])}")
            print(f"  CTC Pred:   {' '.join(example['ctc_pred'])} {status_ctc}")
            print(f"  Att Pred:   {' '.join(example['att_pred'])} {status_att}")

            print(f"\n  CTC Match: {example['ctc_match_rate']*100:.1f}%")
            print(f"  Att Match: {example['att_match_rate']*100:.1f}%")

            # Highlight differences
            if not example['ctc_correct']:
                print("\n  CTC Differences:")
                for i, (pred, tgt) in enumerate(zip(example['ctc_pred'], example['target'])):
                    if pred != tgt:
                        print(f"    Position {i}: predicted '{pred}' but expected '{tgt}'")

            if not example['att_correct'] and joint_eval:
                print("\n  Attention Differences:")
                for i, (pred, tgt) in enumerate(zip(example['att_pred'], example['target'])):
                    if pred != tgt:
                        print(f"    Position {i}: predicted '{pred}' but expected '{tgt}'")

        print("\n" + "="*80)

        # Common error analysis
        print("\n🔍 COMMON ERROR PATTERNS:")

        # Find most common mispredictions
        ctc_errors = {}
        att_errors = {}

        for example in debug_info['examples']:
            # CTC errors
            for pred, tgt in zip(example['ctc_pred'], example['target']):
                if pred != tgt:
                    key = f"{tgt} → {pred}"
                    ctc_errors[key] = ctc_errors.get(key, 0) + 1

            # Attention errors
            if joint_eval:
                for pred, tgt in zip(example['att_pred'], example['target']):
                    if pred != tgt:
                        key = f"{tgt} → {pred}"
                        att_errors[key] = att_errors.get(key, 0) + 1

        if ctc_errors:
            print("\n  Top CTC Mispredictions:")
            for error, count in sorted(ctc_errors.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {error}: {count} times")

        if att_errors and joint_eval:
            print("\n  Top Attention Mispredictions:")
            for error, count in sorted(att_errors.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {error}: {count} times")

        print("\n" + "="*80)

    # Log to MLflow
    mlflow.log_metric("val_ctc_wer", ctc_wer)
    mlflow.log_metric("val_att_wer", att_wer)
    mlflow.log_metric("val_bleu", bleu_score)

    if debug:
        mlflow.log_metric("val_ctc_exact_match", debug_info['correct_ctc'] / debug_samples)
        mlflow.log_metric("val_att_exact_match", debug_info['correct_att'] / debug_samples)

    return avg_loss, ctc_wer, att_wer, bleu_score

def train_model(model, train_loader, val_loader, num_epochs, device, vocab, idx_to_gloss, 
                save_dir='checkpoints', beam_width=10, ctc_weight=0.3):
    """Full training loop with MLflow tracking and joint loss"""
    os.makedirs(save_dir, exist_ok=True)

    # Joint CTC/Attention loss with weights
    criterion = JointCTCLoss(ctc_weight=0.3, attention_weight=0.7, blank_id=1)
    
    # ==================== NEW: HIGHER LEARNING RATE ====================
    base_lr = 1e-4
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=base_lr,  # Use base_lr directly
        betas=(0.9, 0.98), 
        eps=1e-9, 
        weight_decay=0.01
    )
    
    # ==================== NEW: WARMUP SCHEDULER ====================
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
    warmup_scheduler = WarmupScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        base_lr=base_lr
    )
    
    print(f"\n{'='*50}")
    print(f"LEARNING RATE CONFIGURATION")
    print(f"{'='*50}")
    print(f"Base LR: {base_lr:.2e}")
    print(f"Total training steps: {total_steps:,}")
    print(f"Warmup steps: {warmup_steps:,} ({warmup_steps/total_steps*100:.1f}%)")
    print(f"{'='*50}\n")
    
    # Log LR configuration to MLflow
    mlflow.log_param("base_learning_rate", base_lr)
    mlflow.log_param("warmup_steps", warmup_steps)
    mlflow.log_param("total_steps", total_steps)
    mlflow.log_param("warmup_ratio", 0.1)

    best_wer = float('inf')
    best_bleu = 0.0
    patience_counter = 0
    max_patience = 10

    # Log loss weights
    mlflow.log_param("ctc_loss_weight", ctc_weight)
    mlflow.log_param("attention_loss_weight", 1-ctc_weight)

    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")

        # Train - PASS warmup_scheduler instead of old scheduler
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, 
            device, vocab, epoch, warmup_scheduler  # CHANGED: pass warmup_scheduler
        )
        print(f"Train Loss: {train_loss:.4f}")

        # Validate with joint evaluation
        val_loss, ctc_wer, att_wer, bleu = validate(
            model, val_loader, criterion, 
            device, vocab, idx_to_gloss, 
            use_beam_search=False, 
            beam_width=beam_width,
            joint_eval=True
        )
        print(f"Val Loss: {val_loss:.4f}")
        print(f"CTC WER: {ctc_wer:.4f}, Attention WER: {att_wer:.4f}")
        print(f"BLEU: {bleu:.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_ctc_wer", ctc_wer, step=epoch)
        mlflow.log_metric("val_att_wer", att_wer, step=epoch)
        mlflow.log_metric("val_bleu", bleu, step=epoch)
        mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)

        # Save best model (based on attention WER + BLEU)
        combined_metric = att_wer - bleu  # Lower WER + higher BLEU is better
        if combined_metric < best_wer:
            best_wer = combined_metric
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'ctc_wer': ctc_wer,
                'att_wer': att_wer,
                'bleu': bleu,
                'ctc_weight': ctc_weight
            }
            checkpoint_path = os.path.join(save_dir, 'best_model_joint.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved best joint model: Att WER={att_wer:.4f}, BLEU={bleu:.4f}")
            mlflow.log_metric("best_att_wer", att_wer)
            mlflow.log_metric("best_bleu", bleu)
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
                'ctc_wer': ctc_wer,
                'att_wer': att_wer,
                'bleu': bleu,
            }
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}_joint.pt')
            torch.save(checkpoint, checkpoint_path)

    return best_wer, bleu


def generate_model_summary(model, input_dim, device, batch_size=8, seq_length=100):
    """Generate comprehensive model summary"""
    print("\n" + "="*70)
    print("MODEL SUMMARY - JOINT CTC/ATTENTION")
    print("="*70)
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, seq_length, input_dim).to(device)
    dummy_lengths = torch.tensor([seq_length] * batch_size).to(device)
    dummy_tgt = torch.randint(0, 100, (batch_size, seq_length//2)).to(device)  # Dummy target
    dummy_tgt_lengths = torch.tensor([seq_length//2] * batch_size).to(device)
    
    # Get torchinfo summary (encoder + decoder)
    model_stats = summary(
        model,
        input_data=(dummy_input, dummy_lengths, dummy_tgt, dummy_tgt_lengths),
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
    print(f"CTC Output Shape:           {model.encode(dummy_input, dummy_lengths).shape}")
    print(f"Attention Output Shape:     {model.decode(dummy_tgt, model.encode(dummy_input, dummy_lengths), dummy_tgt_lengths, dummy_lengths).shape}")
    print("="*70 + "\n")
    
    summary_str = str(model_stats)
    
    summary_dict = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "model_size_mb": size_mb,
        "input_shape": list(dummy_input.shape),
        "ctc_output_shape": list(model.encode(dummy_input, dummy_lengths).shape),
        "attention_output_shape": list(model.decode(dummy_tgt, model.encode(dummy_input, dummy_lengths), dummy_tgt_lengths, dummy_lengths).shape),
        "architecture": "joint_ctc_attention_transformer"
    }
    
    return summary_str, summary_dict

def log_gpu_stats(step):
    """Log detailed GPU stats from nvidia-smi"""
    if not torch.cuda.is_available():
        return
    
    try:
        # Get GPU utilization
        result = subprocess.check_output([
            'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], encoding='utf-8')
        
        gpu_util, mem_used, mem_total, temp = result.strip().split(',')
        
        mlflow.log_metric("gpu_utilization_%", float(gpu_util), step=step)
        mlflow.log_metric("gpu_memory_used_mb", float(mem_used), step=step)
        mlflow.log_metric("gpu_memory_total_mb", float(mem_total), step=step)
        mlflow.log_metric("gpu_temperature_c", float(temp), step=step)
    except:
        pass  # Silently fail if nvidia-smi not available

def verify_data_normalization(train_loader, vocab_size, num_samples=5):
    """Verify that data normalization is correct"""
    print("\n" + "="*50)
    print("DATA NORMALIZATION VERIFICATION")
    print("="*50)
    
    for i, (landmarks, landmark_lengths, glosses, gloss_lengths) in enumerate(train_loader):
        if i >= num_samples:
            break
        
        # Check statistics
        print(f"\nSample {i+1}:")
        print(f"  Landmarks shape: {landmarks.shape}")
        print(f"  Landmarks range: [{landmarks.min():.3f}, {landmarks.max():.3f}]")
        print(f"  Landmarks mean: {landmarks.mean():.3f}, std: {landmarks.std():.3f}")
        print(f"  Sequence length: {landmark_lengths[0].item()}")
        print(f"  Glosses length: {gloss_lengths[0].item()}")
        print(f"  Glosses (first 10): {glosses[0][:10].tolist()}")
        
        # Check for anomalies
        if landmarks.isnan().any():
            print("  ⚠️  WARNING: NaN values detected!")
        if landmarks.isinf().any():
            print("  ⚠️  WARNING: Inf values detected!")
        if landmark_lengths[0] < 10:
            print("  ⚠️  WARNING: Very short sequence!")
        
        # Check if landmarks are properly normalized
        if landmarks.min() < -10 or landmarks.max() > 10:
            print("  ⚠️  WARNING: Landmarks may not be normalized (extreme values)")
        
        # ✅ FIXED: Check vocabulary range with passed vocab_size
        max_gloss = glosses.max().item()
        if max_gloss >= vocab_size:
            print(f"  ⚠️  CRITICAL: Gloss ID ({max_gloss}) >= vocab size ({vocab_size})!")
        elif max_gloss > vocab_size * 0.95:
            print(f"  ℹ️  INFO: High gloss ID detected: {max_gloss}/{vocab_size}")
        
        if glosses.min() < 0:
            print("  ⚠️  WARNING: Negative gloss ID detected!")
    
    print("="*50 + "\n")

    
# ==================== MAIN ====================

def main():
    # Configuration
    LANDMARKS_TRAIN = "./landmarks_train"
    LANDMARKS_DEV = "./landmarks_dev"
    VOCAB_FILE = "vocab.json"
    STATS_FILE = "normalization_stats.npz"
    BATCH_SIZE = 16  # Reduced batch size for joint model
    NUM_EPOCHS = 100
    INPUT_DIM = 1659
    D_MODEL = 512
    NHEAD = 8
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    DROPOUT = 0.1
    DIM_FEEDFORWARD = 2048
    BEAM_WIDTH = 10
    CTC_WEIGHT = 0.3  # Balance between CTC (0.3) and Attention (0.7)
    SEED = 42
    
    # Set random seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # MLflow configuration
    EXPERIMENT_NAME = "SignNetAdvanced++"
    RUN_NAME = "joint_ctc_attention_transformer_v1"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Enable cuDNN optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        # Enable TF32 for faster matmul on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # MLflow setup
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
            "num_encoder_layers": NUM_ENCODER_LAYERS,
            "num_decoder_layers": NUM_DECODER_LAYERS,
            "dropout": DROPOUT,
            "dim_feedforward": DIM_FEEDFORWARD,
            "optimizer": "AdamW",
            "learning_rate": 5e-4,  # CHANGED from 5e-5 to 5e-4
            "weight_decay": 0.01,
            "scheduler": "Warmup+CosineDecay",  # CHANGED from "Warmup+Plateau"
            "warmup_ratio": 0.1,
            "augmentation": "temporal+spatial+masking",
            "loss_function": "joint_ctc_attention",
            "ctc_weight": CTC_WEIGHT,
            "attention_weight": 1-CTC_WEIGHT,
            "beam_width": BEAM_WIDTH,
            "normalization": "z-score",
            "seed": SEED,
            "activation": "gelu",
            "architecture": "joint_ctc_attention_transformer"
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
            augment=False,
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
            pin_memory=use_pin_memory,
            prefetch_factor=4,  # Add prefetching
            persistent_workers=True  # Keep workers alive between epochs
        )
        
        verify_data_normalization(train_loader, vocab_size=len(train_dataset.gloss_vocab), num_samples=5)

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=use_pin_memory
        )

        # Create joint model
        print("\n" + "="*50)
        print("Creating Joint CTC/Attention Model")
        print("="*50)

        vocab_size = len(train_dataset.gloss_vocab)
        model = SignLanguageTranslationModel(
            input_dim=1659,
            num_glosses=vocab_size,
            d_model=1024,           # CHANGED: 512 -> 1024 (2x)
            nhead=16,               # CHANGED: 8 -> 16 (keep nhead = d_model/64)
            num_encoder_layers=8,   # CHANGED: 6 -> 8 (deeper)
            num_decoder_layers=8,   # CHANGED: 6 -> 8 (deeper)
            dropout=0.1,
            dim_feedforward=4096    # CHANGED: 2048 -> 4096 (2x)
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
        
        model_summary_path = "model_summary_joint.txt"
        with open(model_summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_str)
            f.write("\n\n" + "="*70 + "\n")
            f.write("JOINT CTC/ATTENTION ARCHITECTURE\n")
            f.write("="*70 + "\n")
            for key, value in summary_dict.items():
                f.write(f"{key}: {value}\n")
        mlflow.log_artifact(model_summary_path)

        # Train model
        print("\n" + "="*50)
        print("Starting Joint Training")
        print("="*50)

        best_wer, best_bleu = train_model(
            model,
            train_loader,
            val_loader,
            NUM_EPOCHS,
            device,
            train_dataset.gloss_vocab,
            train_dataset.idx_to_gloss,
            beam_width=BEAM_WIDTH,
            ctc_weight=CTC_WEIGHT
        )

        print(f"\n{'='*50}")
        print(f"Joint Training Complete!")
        print(f"Best Attention WER: {best_wer:.4f}, Best BLEU: {best_bleu:.4f}")
        print(f"{'='*50}")
        
        mlflow.log_metric("final_best_att_wer", best_wer)
        mlflow.log_metric("final_best_bleu", best_bleu)
        mlflow.pytorch.log_model(model, "joint_model")
        
        mlflow.set_tags({
            "model_type": "joint_transformer",
            "task": "sign_language_translation",
            "dataset": "phoenix-2014",
            "status": "completed",
            "improvements": "joint_ctc_attention+teacher_forcing+bleu_eval",
            "ctc_weight": str(CTC_WEIGHT)
        })

if __name__ == "__main__":
    main()
