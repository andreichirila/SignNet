#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
True-Skeleton Bidirectional Multi-Stream GCN for Isolated Sign Recognition
Enhanced version with:
- STC (Spatial-Temporal-Channel) Attention modules
- Start/End frame detection with Bi-LSTM
- Paper-exact training hyperparameters (SGD+Nesterov, step LR decay)
- Decoupled GCN with DropGraph
"""

import os
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch
import platform
import psutil
from telegram import Bot
import asyncio
import signal

# ===========================
# Utils: Reproducibility
# ===========================
TELEGRAM_BOT_TOKEN = '8327173184:AAGLA5pcLiAz-vMSVBq4tVJCHo7TPH3Zu8g'
CHAT_ID = '8541359800'

bot = Bot(token=TELEGRAM_BOT_TOKEN)

async def send_message(text, chat_id):
    async with bot:
        await bot.send_message(text=text, chat_id=chat_id)

async def run_bot(messages, chat_id):
    text = '\n'.join(messages)
    await send_message(text, chat_id)


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_topk_vocabulary(npz_files, K=50, debug=True):
    """
    Scan all .npz files and return:
    - top_k_words: set of K most frequent glosses
    - counts: dict gloss -> count
    """
    from collections import Counter
    counts = Counter()
    skipped = 0
    for f in npz_files:
        try:
            d = np.load(f, allow_pickle=True)
            g = d['glosses'][0]
            counts[g] += 1
        except Exception:
            skipped += 1
    if debug and skipped:
        print(f" [INFO] TopK builder skipped {skipped} unreadable files.")
    most_common = counts.most_common(K)
    top_k_words = {w for (w, _) in most_common}
    return top_k_words, dict(counts)


# ===========================
# Dataset splitting utilities
# ===========================
def split_dataset_stratified(dataset,
                             train_ratio=0.7,
                             val_ratio=0.15,
                             test_ratio=0.15,
                             random_state=42,
                             min_samples_per_class=3,
                             debug=True):
    """
    Robust stratified split with safeguards on rare classes.
    """
    if hasattr(dataset, 'npz_files'):
        file_list = dataset.npz_files
    elif hasattr(dataset, 'files'):
        file_list = dataset.files
    else:
        raise AttributeError("Dataset must have either 'npz_files' or 'files'.")

    print("\n[DATASET SPLITTING]")
    print(f" Total samples: {len(file_list)}")
    print(f" Split ratios: Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")

    labels = []
    indices = []
    skipped = []
    for idx, fpath in enumerate(file_list):
        try:
            data = np.load(fpath, allow_pickle=True)
            gloss = data['glosses'][0]
            labels.append(dataset.word_to_idx[gloss])
            indices.append(idx)
        except Exception as e:
            skipped.append((idx, str(fpath), str(e)))
    if skipped and debug:
        print(f" [INFO] Skipped {len(skipped)} files with read/gloss errors.")

    labels = np.array(labels)
    indices = np.array(indices)

    min_required = max(min_samples_per_class, 4)
    counts = Counter(labels)
    rare_classes = [c for c, n in counts.items() if n < min_required]
    if rare_classes:
        if debug:
            print(f" [INFO] Filtering {len(rare_classes)} classes with < {min_required} samples")
            for c in rare_classes[:10]:
                cname = dataset.idx_to_word.get(c, str(c))
                print(f"  - {cname} (count={counts[c]})")
        mask = ~np.isin(labels, rare_classes)
        labels = labels[mask]
        indices = indices[mask]
        print(f" Remaining samples: {len(indices)} \n Remaining classes: {len(set(labels))}")

    try:
        tr_idx, tmp_idx, tr_y, tmp_y = train_test_split(
            indices, labels,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            stratify=labels
        )
    except ValueError as e:
        print(f" [WARN] First stratified split failed: {e}")
        tr_idx, tmp_idx, tr_y, tmp_y = train_test_split(
            indices, labels,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            stratify=None
        )

    tmp_counts = Counter(tmp_y)
    too_small_tmp = [c for c, n in tmp_counts.items() if n < 2]
    if too_small_tmp:
        if debug:
            print(f" [INFO] Adjusting classes with <2 samples in temp for second split: {len(too_small_tmp)} classes")
        tr_by_c = defaultdict(list)
        tmp_by_c = defaultdict(list)
        for i, c in zip(tr_idx, tr_y):
            tr_by_c[c].append(i)
        for i, c in zip(tmp_idx, tmp_y):
            tmp_by_c[c].append(i)
        moved = 0
        for c in too_small_tmp:
            need = 2 - len(tmp_by_c[c])
            give = min(need, len(tr_by_c[c]))
            if give > 0:
                move_indices = tr_by_c[c][:give]
                tr_by_c[c] = tr_by_c[c][give:]
                tmp_by_c[c].extend(move_indices)
                moved += give
        if debug and moved > 0:
            print(f" [INFO] Moved {moved} samples from train to temp to stabilize stratification.")
        tr_idx = np.array([i for ilist in tr_by_c.values() for i in ilist])
        tmp_idx = np.array([i for ilist in tmp_by_c.values() for i in ilist])
        tr_y = np.array([labels[np.where(indices==i)[0][0]] for i in tr_idx])
        tmp_y = np.array([labels[np.where(indices==i)[0][0]] for i in tmp_idx])

    tmp_counts = Counter(tmp_y)
    stratify_tmp = None if any(n < 2 for n in tmp_counts.values()) else tmp_y

    try:
        val_size = test_ratio / (val_ratio + test_ratio)
        va_idx, te_idx, va_y, te_y = train_test_split(
            tmp_idx, tmp_y,
            test_size=val_size,
            random_state=random_state,
            stratify=stratify_tmp
        )
    except ValueError as e:
        print(f" [WARN] Second stratified split failed: {e}")
        val_size = test_ratio / (val_ratio + test_ratio)
        va_idx, te_idx, va_y, te_y = train_test_split(
            tmp_idx, tmp_y,
            test_size=val_size,
            random_state=random_state,
            stratify=None
        )

    total = len(tr_idx) + len(va_idx) + len(te_idx)
    print("\n[SPLIT RESULTS]")
    print(f" Train: {len(tr_idx)} ({100*len(tr_idx)/total:.1f}%)")
    print(f" Val:   {len(va_idx)} ({100*len(va_idx)/total:.1f}%)")
    print(f" Test:  {len(te_idx)} ({100*len(te_idx)/total:.1f}%)")
    print("\n[CLASS DISTRIBUTION CHECK]")
    print(f" Classes in train: {len(set([labels[np.where(indices==i)[0][0]] for i in tr_idx]))}")
    print(f" Classes in val:   {len(set([labels[np.where(indices==i)[0][0]] for i in va_idx]))}")
    print(f" Classes in test:  {len(set([labels[np.where(indices==i)[0][0]] for i in te_idx]))}")

    return tr_idx.tolist(), va_idx.tolist(), te_idx.tolist()


class SubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        return self.dataset[self.indices[i]]


# ===========================
# NEW: Start/End Frame Detection with Bi-LSTM
# ===========================
class StartEndFrameDetector(nn.Module):
    """
    Bi-LSTM model for detecting start/end frames of signs.
    Uses bounding box features, velocity, and acceleration.
    Paper: Section 3.2.1
    """
    def __init__(self, input_dim=8, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)  # Output probability
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (B, T, input_dim) - features including bbox coords, velocity, acceleration
        Returns: (B, T, 1) - probability of boundary at each frame
        """
        lstm_out, _ = self.lstm(x)  # (B, T, hidden*2)
        probs = self.sigmoid(self.fc(lstm_out))  # (B, T, 1)
        return probs

    def detect_boundaries(self, features, confidence_threshold=0.5):
        """
        Detect start/end frames from a sequence.
        Returns: (start_frame, end_frame) indices
        """
        with torch.no_grad():
            probs = self.forward(features).squeeze(-1)  # (B, T)

            # For start detection: first frame above threshold
            start_candidates = (probs > confidence_threshold).nonzero(as_tuple=True)
            if len(start_candidates[1]) > 0:
                start_frame = start_candidates[1][0].item()
            else:
                start_frame = 0

            # For end detection: reverse and find first (use reversed input for end detector)
            return start_frame, None  # End detector would be trained separately


def extract_bbox_features(landmarks_seq):
    """
    Extract bounding box features for hand detection (simplified).
    In practice, use YOLOv3 as in paper.
    Returns: (T, 8) features: [left_hand_bbox(4), right_hand_bbox(4)]
    """
    T = landmarks_seq.shape[0]
    # Placeholder: extract hand regions (indices 9-18 for left, 19-27 for right from 27-node skeleton)
    # In real implementation, use proper hand detection
    features = np.zeros((T, 8))
    return features


class SkeletonAugmentation:
    """
    Data augmentation for skeleton sequences as described in the paper.
    Implements: random sampling, mirroring, rotation, scaling, and shifting.
    """
    def __init__(self,
                 mirror_prob=0.5,
                 rotation_range=(-10, 10),  # degrees
                 scale_range=(0.9, 1.1),
                 shift_range=(-0.1, 0.1),
                 apply_prob=0.8):
        """
        Args:
            mirror_prob: Probability of applying horizontal flip
            rotation_range: Range of rotation angles in degrees (min, max)
            scale_range: Range of scaling factors (min, max)
            shift_range: Range of translation in normalized coordinates
            apply_prob: Overall probability of applying augmentation
        """
        self.mirror_prob = mirror_prob
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.apply_prob = apply_prob

    def mirror(self, nodes_seq):
        """
        Horizontal flip (mirroring) with left-right hand swap.
        nodes_seq: (T, V=29, 3)
        """
        nodes = nodes_seq.copy()

        # Flip x-coordinates
        nodes[:, :, 0] = -nodes[:, :, 0]

        # Swap left-right body parts
        nodes[:, [1, 2]] = nodes[:, [2, 1]]  # Eyes
        nodes[:, [3, 4]] = nodes[:, [4, 3]]  # Shoulders
        nodes[:, [5, 6]] = nodes[:, [6, 5]]  # Elbows
        nodes[:, [7, 8]] = nodes[:, [8, 7]]  # Wrists

        # Swap hands - NOW SYMMETRIC (both 10 nodes)
        left_hand = nodes[:, 9:19].copy()    # Nodes 9-18 (10 nodes)
        right_hand = nodes[:, 19:29].copy()  # Nodes 19-28 (10 nodes)
        nodes[:, 9:19] = right_hand          # Right → Left
        nodes[:, 19:29] = left_hand          # Left → Right

        return nodes

    def rotate(self, nodes_seq, angle):
        """
        Rotate skeleton around the nose (center point).
        angle: rotation angle in degrees
        """
        nodes = nodes_seq.copy()
        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Rotation matrix
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])

        # Get center (nose is node 0)
        center = nodes[:, 0:1, :2]  # (T, 1, 2)

        # Rotate around center
        coords = nodes[:, :, :2] - center
        T, V = coords.shape[:2]
        coords_flat = coords.reshape(-1, 2)
        rotated = (rotation_matrix @ coords_flat.T).T
        nodes[:, :, :2] = rotated.reshape(T, V, 2) + center

        return nodes

    def scale(self, nodes_seq, scale_factor):
        """
        Scale skeleton size around the nose (center point).
        """
        nodes = nodes_seq.copy()
        center = nodes[:, 0:1, :2]  # (T, 1, 2)

        # Scale around center
        coords = nodes[:, :, :2] - center
        coords = coords * scale_factor
        nodes[:, :, :2] = coords + center

        # Clip to valid range
        nodes[:, :, 0] = np.clip(nodes[:, :, 0], -1, 1)
        nodes[:, :, 1] = np.clip(nodes[:, :, 1], -1, 1)

        return nodes

    def shift(self, nodes_seq, shift_x, shift_y):
        """
        Translate skeleton in x and y directions.
        """
        nodes = nodes_seq.copy()
        nodes[:, :, 0] += shift_x
        nodes[:, :, 1] += shift_y

        # Clip to valid range
        nodes[:, :, 0] = np.clip(nodes[:, :, 0], -1, 1)
        nodes[:, :, 1] = np.clip(nodes[:, :, 1], -1, 1)

        return nodes

    def __call__(self, nodes_seq):
        """
        Apply random augmentations to skeleton sequence.
        nodes_seq: (T, V, 3) numpy array
        Returns: augmented (T, V, 3) array
        """
        # Skip augmentation with probability (1 - apply_prob)
        if np.random.rand() > self.apply_prob:
            return nodes_seq

        nodes = nodes_seq.copy()

        T_orig = nodes.shape[0]
        if T_orig > 32 and np.random.rand() < 0.3:  # 30% chance to shorten
            target_T = np.random.randint(24, T_orig)
            indices = np.linspace(0, T_orig-1, target_T, dtype=int)
            nodes = nodes[indices]

        # Mirroring
        if np.random.rand() < self.mirror_prob:
            nodes = self.mirror(nodes)

        # Rotation
        angle = np.random.uniform(*self.rotation_range)
        nodes = self.rotate(nodes, angle)

        # Scaling
        scale_factor = np.random.uniform(*self.scale_range)
        nodes = self.scale(nodes, scale_factor)

        # Shifting
        shift_x = np.random.uniform(*self.shift_range)
        shift_y = np.random.uniform(*self.shift_range)
        nodes = self.shift(nodes, shift_x, shift_y)

        return nodes

# ===========================
# 27-node skeleton extractor
# ===========================
class Skeleton27FeatureExtractor:
    """29-node skeleton extractor (updated from 27 for symmetric hands)"""
    def __init__(self, conf_valid_thresh: float = 0.5, augmentation=None):
        self.num_nodes = 29  # CHANGED: 27 → 29
        self.conf_valid_thresh = conf_valid_thresh
        self.augmentation = augmentation
        self.edges = [
            # Body connections (0-8)
            (0,1),(0,2),(1,3),(2,4),(3,5),(4,6),(5,7),(6,8),
            # Left hand (9-18): 10 nodes
            (7,9),(9,10),(7,11),(11,12),(7,13),(13,14),(7,15),(15,16),(7,17),(17,18),
            # Right hand (19-28): 10 nodes - SYMMETRIC NOW
            (8,19),(19,20),(8,21),(21,22),(8,23),(23,24),(8,25),(25,26),(8,27),(27,28)
        ]

    def map_combined_to_27(self, combined_frame):
        """Map to 29-node skeleton with symmetric 10-node hands."""
        nodes = np.zeros((self.num_nodes, 3), dtype=np.float32)

        HAND_L = combined_frame[0:63].reshape(21, 3)
        HAND_R = combined_frame[63:126].reshape(21, 3)
        FACE   = combined_frame[126:126+1434].reshape(478, 3)
        POSE   = combined_frame[126+1434:126+1434+99].reshape(33, 3)

        # Upper body (nodes 0-8)
        nodes[0] = FACE[1]    # Nose
        nodes[1] = FACE[33]   # Left eye
        nodes[2] = FACE[263]  # Right eye
        nodes[3] = POSE[12]   # Left shoulder
        nodes[4] = POSE[11]   # Right shoulder
        nodes[5] = POSE[14]   # Left elbow
        nodes[6] = POSE[13]   # Right elbow
        nodes[7] = POSE[16]   # Left wrist
        nodes[8] = POSE[15]   # Right wrist

        # Left hand (nodes 9-18): 10 nodes - base + tip of 5 fingers
        if np.any(HAND_L != 0):
            nodes[9]  = HAND_L[1]   # Thumb base
            nodes[10] = HAND_L[4]   # Thumb tip
            nodes[11] = HAND_L[5]   # Index base
            nodes[12] = HAND_L[8]   # Index tip
            nodes[13] = HAND_L[9]   # Middle base
            nodes[14] = HAND_L[12]  # Middle tip
            nodes[15] = HAND_L[13]  # Ring base
            nodes[16] = HAND_L[16]  # Ring tip
            nodes[17] = HAND_L[17]  # Pinky base
            nodes[18] = HAND_L[20]  # Pinky tip
        else:
            nodes[9:19] = 0
            nodes[9:19, 2] = 0.1

        # Right hand (nodes 19-28): 10 nodes - SYMMETRIC to left hand
        if np.any(HAND_R != 0):
            nodes[19] = HAND_R[1]   # Thumb base
            nodes[20] = HAND_R[4]   # Thumb tip
            nodes[21] = HAND_R[5]   # Index base
            nodes[22] = HAND_R[8]   # Index tip
            nodes[23] = HAND_R[9]   # Middle base
            nodes[24] = HAND_R[12]  # Middle tip
            nodes[25] = HAND_R[13]  # Ring base
            nodes[26] = HAND_R[16]  # Ring tip
            nodes[27] = HAND_R[17]  # Pinky base
            nodes[28] = HAND_R[20]  # Pinky tip
        else:
            nodes[19:29] = 0
            nodes[19:29, 2] = 0.1

        # Normalization
        nodes[:, 0:2] = np.clip(nodes[:, 0:2], -1, 1)
        nodes[:, 2]   = np.clip(nodes[:, 2], 0, 1)

        if np.all(nodes[:, :2] == 0):
            nodes[:, 2] = 0.0

        return nodes

    def compute_streams(self, nodes_seq):
        """Compute all feature streams with AGGRESSIVE normalization."""
        T, V = nodes_seq.shape[0], nodes_seq.shape[1]
        K = nodes_seq.copy().astype(np.float32)
        valid = ((np.abs(K[..., 0]) + np.abs(K[..., 1])) > 1e-6).astype(np.float32)

        B = np.zeros_like(K)
        D = np.zeros_like(K)
        b_count = np.zeros((T, V, 1), dtype=np.float32)

        for (i, j) in self.edges:
            diff = K[:, j, :2] - K[:, i, :2]
            dist = np.linalg.norm(diff, axis=-1, keepdims=True)
            conf_ij = 0.5 * (K[:, j, 2:3] + K[:, i, 2:3])

            B[:, j, :2] += diff
            B[:, j,  2:3] += conf_ij
            b_count[:, j, :] += 1.0
            D[:, j, 2:3] = np.maximum(D[:, j, 2:3], dist)

        mask_b = (b_count > 0).astype(np.float32)
        B[:, :, :2] = np.where(mask_b.astype(bool), B[:, :, :2] / np.maximum(b_count, 1e-6), 0.0)
        B[:, :,  2] = np.where(mask_b[...,0].astype(bool), B[:, :, 2] / np.maximum(b_count[...,0], 1e-6), 0.0)
        D[:, :, 0:2] = 0.0

        # CRITICAL: Aggressive clipping for bone/distance features
        B[:, :, :2] = np.clip(B[:, :, :2], -1, 1)   # Bone vectors
        D[:, :, 2] = np.clip(D[:, :, 2], 0, 2)       # Distances

        # Normalize bone vectors by their magnitude
        bone_mag = np.linalg.norm(B[:, :, :2], axis=-1, keepdims=True)
        B[:, :, :2] = np.where(bone_mag > 1e-6, B[:, :, :2] / (bone_mag + 1e-6), B[:, :, :2])

        # Normalize distances to [0, 1] range
        D[:, :, 2] = D[:, :, 2] / (D[:, :, 2].max() + 1e-6)

        KV = np.zeros_like(K); BV = np.zeros_like(B)
        KA = np.zeros_like(K); BA = np.zeros_like(B)
        valid_b = (B[..., 2] > 0).astype(np.float32)

        for t in range(1, T):
            pair_mask_k = (valid[t] * valid[t-1])[..., None]
            KV[t] = (K[t] - K[t-1]) * pair_mask_k
            pair_mask_b = (valid_b[t] * valid_b[t-1])[..., None]
            BV[t] = (B[t] - B[t-1]) * pair_mask_b

        for t in range(2, T):
            tri_mask_k = (valid[t] * valid[t-1] * valid[t-2])[..., None]
            KA[t] = (KV[t] - KV[t-1]) * tri_mask_k
            tri_mask_b = (valid_b[t] * valid_b[t-1] * valid_b[t-2])[..., None]
            BA[t] = (BV[t] - BV[t-1]) * tri_mask_b

        # CRITICAL: Clip ALL velocity and acceleration features
        KV = np.clip(KV, -0.5, 0.5)   # Keypoint velocity
        BV = np.clip(BV, -0.5, 0.5)   # Bone velocity
        KA = np.clip(KA, -0.2, 0.2)   # Keypoint acceleration
        BA = np.clip(BA, -0.2, 0.2)   # Bone acceleration

        return {
            'keypoint_coords': K,
            'edge_distance':   D,
            'bone_vectors':    B,
            'keypoint_velocity': KV,
            'bone_velocity':     BV,
            'keypoint_accel':    KA,
            'bone_accel':        BA,
        }


    def extract(self, combined_sequence, apply_augmentation=True):
        """
        Extract features with optional augmentation.
        apply_augmentation: bool, whether to apply augmentation (set False for val/test)
        """
        T = combined_sequence.shape[0]
        nodes_seq = np.stack([self.map_combined_to_27(combined_sequence[t]) for t in range(T)], axis=0)

        # NEW: Apply augmentation if enabled
        if apply_augmentation and self.augmentation is not None:
            nodes_seq = self.augmentation(nodes_seq)

        return self.compute_streams(nodes_seq)


# ===========================
# Dataset for true skeleton
# ===========================
class TrueSkeletonDataset(Dataset):
    def __init__(self, npz_dir, feature_extractor: Skeleton27FeatureExtractor,
                 word_to_idx=None, debug=True, is_training=True):
        self.dir = Path(npz_dir)
        self.files = sorted(self.dir.glob('*.npz'))
        self.fe = feature_extractor
        self.debug = debug
        self.is_training = is_training  # NEW: Track if training mode

        if word_to_idx is None:
            self.word_to_idx = {}
            for f in self.files:
                try:
                    d = np.load(f, allow_pickle=True)
                    g = d['glosses'][0]
                    if g not in self.word_to_idx:
                        self.word_to_idx[g] = len(self.word_to_idx)
                except Exception as e:
                    if debug:
                        print('[WARN] building vocab:', f, e)
        else:
            self.word_to_idx = word_to_idx
        self.idx_to_word = {v:k for k,v in self.word_to_idx.items()}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        d = np.load(f, allow_pickle=True)
        X = d['landmarks'].astype(np.float32)
        # NEW: Pass is_training flag to enable/disable augmentation
        streams = self.fe.extract(X, apply_augmentation=self.is_training)
        g = d['glosses'][0] if len(d['glosses'])>0 else 'UNKNOWN'
        y = self.word_to_idx.get(g, 0)
        return streams, torch.tensor(y, dtype=torch.long), f.stem


class TopKTrueSkeletonDataset(Dataset):
    def __init__(self, base_dataset: 'TrueSkeletonDataset', top_k_words: set,
                 feature_extractor=None, debug=True, is_training=True):
        self.base = base_dataset
        self.debug = debug
        self.is_training = is_training  # NEW: Track training mode
        self.fe = feature_extractor if feature_extractor is not None else self.base.fe

        kept = []
        for f in self.base.files:
            try:
                d = np.load(f, allow_pickle=True)
                g = d['glosses'][0]
                if g in top_k_words:
                    kept.append(f)
            except Exception as e:
                if debug:
                    print('[WARN] filtering top-K:', f, e)
        self.files = kept

        words = []
        for f in self.files:
            d = np.load(f, allow_pickle=True)
            words.append(d['glosses'][0])
        uniq = sorted(set(words))
        if debug:
            print(f" [TOP-K] Kept {len(self.files)} files across {len(uniq)} words")
        self.word_to_idx = {w:i for i,w in enumerate(uniq)}
        self.idx_to_word = {i:w for w,i in self.word_to_idx.items()}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        d = np.load(f, allow_pickle=True)
        X = d['landmarks'].astype(np.float32)
        # NEW: Pass training flag for augmentation
        streams = self.fe.extract(X, apply_augmentation=self.is_training)
        g = d['glosses'][0]
        y = self.word_to_idx[g]
        return streams, torch.tensor(y, dtype=torch.long), f.stem


# ===========================
# Collator with node mask
# ===========================
class TrueSkeletonCollator:
    def __init__(self, conf_valid_thresh: float = 0.5):
        self.conf_valid_thresh = conf_valid_thresh

    def __call__(self, batch):
        max_T = max(next(iter(b[0].values())).shape[0] for b in batch)
        stream_names = list(batch[0][0].keys())

        fwd = {s: [] for s in stream_names}
        bwd = {s: [] for s in stream_names}
        lengths = []
        node_masks = []

        labels, ids = [], []

        for streams, y, sid in batch:
            T = next(iter(streams.values())).shape[0]
            lengths.append(T)
            labels.append(y)
            ids.append(sid)

            kc = streams['keypoint_coords']
            mask = ((np.abs(kc[..., 0]) + np.abs(kc[..., 1])) > 1e-6).astype(np.float32)

            if kc.shape[0] < max_T:
                pad = max_T - kc.shape[0]
                mask = np.pad(mask, ((0,pad),(0,0)), mode='constant')
            node_masks.append(mask)

            for s in stream_names:
                arr = streams[s]
                if arr.shape[0] < max_T:
                    pad = max_T - arr.shape[0]
                    arr = np.pad(arr, ((0,pad),(0,0),(0,0)), mode='constant')
                fwd[s].append(arr)
                bwd[s].append(arr[::-1].copy())

        for s in stream_names:
            fwd[s] = torch.tensor(np.stack(fwd[s], axis=0), dtype=torch.float32)
            bwd[s] = torch.tensor(np.stack(bwd[s], axis=0), dtype=torch.float32)

        lengths = torch.tensor(lengths, dtype=torch.long)
        node_masks = torch.tensor(np.stack(node_masks, axis=0), dtype=torch.float32)

        return {
            'features_forward': fwd,
            'features_backward': bwd,
            'lengths': lengths,
            'node_mask': node_masks,
            'labels': torch.stack(labels),
            'ids': ids
        }


# ===========================
# NEW: STC Attention Module (Paper Section 3.2.2, Figure 5)
# ===========================
class SpatialAttention(nn.Module):
    """Spatial attention sub-module."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, V, T) or similar spatial feature map
        # Compute attention over spatial dimension (nodes)
        att = self.sigmoid(self.conv(x))  # (B, 1, V, T)
        return x * att + x  # Residual connection


class TemporalAttention(nn.Module):
    """Temporal attention sub-module."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, V, T)
        # Attention over temporal dimension
        att = self.sigmoid(self.conv(x))  # (B, 1, V, T)
        return x * att + x


class ChannelAttention(nn.Module):
    """Channel attention sub-module."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, V, T)
        # Global pooling over spatial and temporal dims
        B, C, V, T = x.shape
        y = x.view(B, C, -1).mean(dim=2)  # (B, C)
        att = self.fc(y).view(B, C, 1, 1)  # (B, C, 1, 1)
        return x * att + x


class STCAttentionSimple(nn.Module):
    """Simplified STC Attention: Spatial-Temporal-Channel attention for GCN blocks."""
    def __init__(self, channels, reduction=8):  # FIXED: Added channels param (after self)
        super().__init__()
        # Spatial attention: Conv on (B, C, T, V) -> (B, 1, T, V)
        self.spatial_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)

        # Temporal attention: Similar, but could avg over V if needed
        self.temporal_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)

        # Channel attention: Global avg pool then MLP
        self.channel_conv = nn.Conv1d(channels, channels // reduction, kernel_size=1, bias=False)
        self.fc = nn.Linear(channels // reduction, channels, bias=False)

        # Sigmoid for gating
        self.sigmoid = nn.Sigmoid()

        # Init weights (minimal scaling)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='sigmoid')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, lengths=None):
        B, C, T, V = x.shape

        # Apply temporal masking if lengths provided (zero pads)
        if lengths is not None:
            max_t = T  # Assume x padded to fixed max_T (e.g., 75)
            t_idx = torch.arange(max_t, device=x.device).unsqueeze(0).expand(B, -1)  # (B, T), long
            mask = t_idx < lengths.unsqueeze(1)  # Bool mask (B, T)
            mask = mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, T, 1) broadcasts to (B, C, T, V)
            x = x.masked_fill(~mask, 0.0)

        # Spatial attention
        spatial_att = self.sigmoid(self.spatial_conv(x))  # (B, 1, T, V)

        # Temporal attention
        temporal_att = self.sigmoid(self.temporal_conv(x))  # (B, 1, T, V)

        # Channel attention: Global avg pool for gating
        x_channel_in = x.mean(dim=[2, 3])  # (B, C)
        x_channel = torch.relu(self.channel_conv(x_channel_in.unsqueeze(-1)).squeeze(-1))  # (B, C//r)
        x_channel = self.sigmoid(self.fc(x_channel))  # (B, C)
        x_channel = x_channel.view(B, C, 1, 1)  # (B, C, 1, 1)

        # FIXED: Full STC: Modulate x with all attentions (broadcast)
        out = x * spatial_att * temporal_att * x_channel  # (B, C, T, V) - element-wise

        # FIXED: Return full tensor for block-level addition (no early pooling)
        return out  # Shape: (B, C, T, V)





# ===========================
# NEW: Decoupled GCN with DropGraph (Paper Section 3.2.2)
# ===========================
class DecoupledGraphConvolution(nn.Module):
    """
    Decoupled spatial GCN with group parameter.
    Paper: "decoupling GCN layer... features organized into g groups"

    FIX: Automatically adjust groups for input layers where in_features < groups
    """
    def __init__(self, in_features, out_features, groups=8):
        super().__init__()

        # Auto-adjust groups if in_features is too small
        # Use greatest common divisor approach
        effective_groups = groups
        while in_features % effective_groups != 0 or out_features % effective_groups != 0:
            effective_groups -= 1
            if effective_groups == 1:
                break

        self.groups = effective_groups
        self.in_per_group = in_features // self.groups
        self.out_per_group = out_features // self.groups

        # Each group has its own weight matrix
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(self.in_per_group, self.out_per_group))
            for _ in range(self.groups)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(self.out_per_group))
            for _ in range(self.groups)
        ])

        for w in self.weights:
            nn.init.xavier_uniform_(w)

    def forward(self, x, A):
        # x: (B, V, Fin)
        B, V, F = x.shape
        x_groups = x.chunk(self.groups, dim=-1)  # List of (B, V, Fin/g)

        outputs = []
        for i, x_g in enumerate(x_groups):
            support = x_g @ self.weights[i]  # (B, V, Fout/g)
            out = A.unsqueeze(0) @ support + self.biases[i]  # (B, V, Fout/g)
            outputs.append(out)

        return torch.cat(outputs, dim=-1)  # (B, V, Fout)


class DropGraph(nn.Module):
    """
    DropGraph layer for regularization.
    Paper: Cheng et al., 2020 - cited in Section 3.2.2
    """
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        # Randomly drop nodes with probability p
        mask = torch.rand(x.size(0), x.size(1), 1, device=x.device) > self.p
        return x * mask.float()


class StableGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_kernel=3, dropout=0.2):
        super().__init__()
        self.graph_conv = nn.Linear(in_channels, out_channels, bias=False)
        self.ln_graph = nn.LayerNorm(out_channels)
        self.temporal_conv = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(temporal_kernel, 1),
            padding=((temporal_kernel - 1) // 2, 0), bias=False)
        self.bn_temporal = nn.BatchNorm2d(out_channels)
        self.ln_temporal = nn.LayerNorm([out_channels])  # FIXED: Explicit shape for LN on channels
        self.attention = STCAttentionSimple(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.residual = nn.Linear(in_channels, out_channels, bias=False) if in_channels != out_channels else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        # Unchanged from previous fix
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, A, lengths=None):  # FIXED: Added lengths=None param
        B, T, V, C_in = x.shape
        x_flat = x.view(B * T * V, C_in)
        x_feat = self.graph_conv(x_flat)
        x_feat = x_feat.view(B * T, V, -1)  # (B*T, V, C_out)
        x_gcn = torch.matmul(A, x_feat)  # (B*T, V, C_out)
        x_gcn = self.ln_graph(x_gcn)
        x_gcn = self.relu(x_gcn)
        x_gcn = self.dropout(x_gcn)
        x_gcn = x_gcn.view(B, T, V, -1)  # (B, T, V, C_out)

        # Temporal conv on (B, C, V, T)
        x_temp = x_gcn.permute(0, 3, 2, 1)  # (B, C_out, V, T)
        x_temp = self.temporal_conv(x_temp)  # (B, C_out, V, T)
        x_temp = self.bn_temporal(x_temp)

        # Permute back to (B, T, V, C_out) before LN
        x_temp = x_temp.permute(0, 3, 2, 1).contiguous()  # (B, T, V, C_out)
        assert x_temp.shape == (B, T, V, self.ln_temporal.normalized_shape[0]), f"LN shape mismatch: {x_temp.shape}"
        x_temp = self.ln_temporal(x_temp)  # Norm over C_out dim
        x_temp = self.relu(x_temp)

        # Permute to (B, C_out, T, V) for attention
        x_att_input = x_temp.permute(0, 3, 1, 2)  # (B, C_out, T, V)
        assert x_att_input.shape[1] == self.attention.spatial_conv.in_channels, f"Attention input channels: {x_att_input.shape[1]} expected {self.attention.spatial_conv.in_channels}"
        x_att = self.attention(x_att_input, lengths=lengths)  # FIXED: Pass lengths to attention

        # Permute attention output back (B, C, T, V) -> (B, T, V, C)
        if x_att.dim() == 4 and x_att.shape[1] == x_temp.shape[-1]:
            x_att = x_att.permute(0, 2, 3, 1).contiguous()
        x_att = self.dropout(x_att)

        # Residual (unchanged)
        if isinstance(self.residual, nn.Identity):
            x_res = x
            if x.shape[-1] != x_att.shape[-1]:
                x_res = self.residual(x.view(-1, C_in)).view(B, T, V, -1)
        else:
            x_res = self.residual(x.view(-1, C_in)).view(B, T, V, -1)
        out = 0.98 * x_res + 0.02 * x_att
        return out




class StableStreamProcessor(nn.Module):
    def __init__(self, in_feat=3, hidden=64, num_blocks=3, temporal_kernel=3, dropout=0.2):
        super().__init__()
        self.blocks = nn.ModuleList([StableGCNBlock(in_feat if i == 0 else hidden, hidden, temporal_kernel, dropout) for i in range(num_blocks)])

    def forward(self, seq, A, node_mask, lengths):
        x = seq
        for block in self.blocks:
            x = block(x, A, lengths=lengths)  # NEW: Pass lengths
        # Masked mean pooling
        w = node_mask.unsqueeze(-1)
        x_masked = (x * w).sum(dim=2) / (w.sum(dim=2) + 1e-6)
        return x_masked.mean(dim=1)




# ===========================
# NEW: Enhanced Bidirectional GCN Model
# ===========================
class StableBidirectionalGCN(nn.Module):
    def __init__(self, num_classes, hidden=256, num_blocks=3, temporal_kernel=3, dropout=0.3):
        super().__init__()

        # All 7 streams for full paper implementation
        streams = [
            'keypoint_coords',
            'edge_distance',
            'bone_vectors',
            'keypoint_velocity',
            'bone_velocity',
            'keypoint_accel',
            'bone_accel'
        ]
        self.streams = streams

        self.forward_streams = nn.ModuleDict({
            s: StableStreamProcessor(3, hidden, num_blocks, temporal_kernel, dropout)
            for s in streams
        })
        self.backward_streams = nn.ModuleDict({
            s: StableStreamProcessor(3, hidden, num_blocks, temporal_kernel, dropout)
            for s in streams
        })

        # NEW: LayerNorm for fused outputs
        self.ln_fused = nn.LayerNorm(hidden)

        # Fusion weights
        self.w_fwd = nn.Parameter(torch.ones(len(streams)) / len(streams))
        self.w_bwd = nn.Parameter(torch.ones(len(streams)) / len(streams))
        self.w_dir = nn.Parameter(torch.tensor([0.5, 0.5]))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes)
        )

        self.register_buffer('A', self._build_adj())

    def _build_adj(self):
        """Build adjacency matrix for 29 nodes."""
        V = 29  # CHANGED: 27 → 29
        A = torch.zeros(V, V)
        edges = [
            # Body (0-8)
            (0,1),(0,2),(1,3),(2,4),(3,5),(4,6),(5,7),(6,8),
            # Left hand (9-18): 10 nodes
            (7,9),(9,10),(7,11),(11,12),(7,13),(13,14),(7,15),(15,16),(7,17),(17,18),
            # Right hand (19-28): 10 nodes
            (8,19),(19,20),(8,21),(21,22),(8,23),(23,24),(8,25),(25,26),(8,27),(27,28)
        ]
        for i, j in edges:
            A[i, j] = 1
            A[j, i] = 1
        A = A + torch.eye(V)
        d = A.sum(dim=1)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0
        return d_inv_sqrt.unsqueeze(0) * A * d_inv_sqrt.unsqueeze(1)

    def forward(self, fwd, bwd, node_mask, lengths):
        # Forward pass
        outs_f = []
        for s in self.streams:
            out_s = self.forward_streams[s](fwd[s], self.A, node_mask, lengths)
            outs_f.append(out_s)

        Fstk = torch.stack(outs_f, dim=1)  # (B, S, H)
        wf = F.softmax(self.w_fwd, dim=0)
        Ffused = self.ln_fused((Fstk * wf.view(1, -1, 1)).sum(dim=1))  # NEW: Norm after fusion

        # Backward pass (similar, with node_mask_b)
        node_mask_b = node_mask.flip(dims=[1])
        outs_b = []
        for s in self.streams:
            out_s = self.backward_streams[s](bwd[s], self.A, node_mask_b, lengths)
            outs_b.append(out_s)

        Bstk = torch.stack(outs_b, dim=1)  # (B, S, H)
        wb = F.softmax(self.w_bwd, dim=0)
        Bfused = self.ln_fused((Bstk * wb.view(1, -1, 1)).sum(dim=1))  # NEW: Norm

        # Directional fusion
        wd = F.softmax(self.w_dir, dim=0)
        fused = wd[0] * Ffused + wd[1] * Bfused

        return self.classifier(fused)





# ===========================
# NEW: Trainer with Paper-Exact Hyperparameters
# ===========================
class StableTrainer:
    """Ultra-conservative trainer for gradient stability."""
    def __init__(self, model, shutdown_handler = None, device='cuda', lr=5e-5, wd=1e-4, epochs=100):  # FIXED: LR up to 5e-5, wd down
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.shutdown_handler = shutdown_handler

        # FIXED: Use AdamW for decoupled decay
        self.optim = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Cosine annealing unchanged
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, T_max=epochs, eta_min=1e-6
        )

        self.crit = nn.CrossEntropyLoss()
        self.scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
        self.hist = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_top5': [], 'val_top5': []
        }

    def step(self, batch):
        fwd = {k: v.to(self.device) for k, v in batch['features_forward'].items()}
        bwd = {k: v.to(self.device) for k, v in batch['features_backward'].items()}
        lengths = batch['lengths'].to(self.device)
        node_mask = batch['node_mask'].to(self.device)
        y = batch['labels'].to(self.device)

        # Input validation
        for k, v in fwd.items():
            v_max = v.abs().max().item()
            if v_max > 10.0:
                print(f'[EXTREME INPUT] {k} has max value {v_max:.2f}')
                fwd[k] = torch.clamp(v, -10, 10)
                bwd[k] = torch.clamp(bwd[k], -10, 10)
            if torch.isnan(v).any() or torch.isinf(v).any():
                print(f'[NaN/Inf in input {k}], skipping batch')
                return 0.0, 0, 0, y.size(0)  # Return zeros for metrics

        # Forward pass
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            logits = self.model(fwd, bwd, node_mask, lengths)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print('[NaN/Inf in model output], skipping batch')
                return 0.0, 0, 0, y.size(0)

            loss = self.crit(logits, y)

            if torch.isnan(loss) or torch.isinf(loss):
                print('[NaN/Inf loss], skipping batch')
                return 0.0, 0, 0, y.size(0)

        # Backward pass
        self.optim.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()

        # Gradient checks
        total_norm = 0
        max_grad = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                max_grad = max(max_grad, p.grad.data.abs().max().item())
        total_norm = total_norm ** 0.5

        if total_norm > 500.0:
            print(f'[Large gradients] norm={total_norm:.2f}, max={max_grad:.2f}')
            print(f'[Clipping] LR={self.optim.param_groups[0]["lr"]:.2e}')

        # Gradient clipping
        self.scaler.unscale_(self.optim)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)

        # Check for NaN/Inf in parameters after clipping
        has_invalid = False
        for p in self.model.parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                has_invalid = True
                break

        if has_invalid:
            print('[NaN/Inf in parameters after clipping], skipping step')
            self.optim.zero_grad(set_to_none=True)
            return 0.0, 0, 0, y.size(0)

        # Optimizer step
        self.scaler.step(self.optim)
        self.scaler.update()

        # Calculate metrics
        top1_correct = (logits.argmax(1) == y).sum().item()

        # Top-5 calculation (FIXED - was missing)
        num_classes = logits.size(1)
        k = min(5, num_classes)
        _, topk_preds = logits.topk(k, dim=1)
        top5_correct = (topk_preds == y.unsqueeze(1)).any(dim=1).sum().item()

        return loss.item(), top1_correct, top5_correct, y.size(0)


    @torch.no_grad()
    def eval_loop(self, loader):
        self.model.eval()
        tot_loss = 0
        top1_correct = 0
        top5_correct = 0
        total = 0

        for batch in loader:
            fwd = {k: v.to(self.device) for k, v in batch['features_forward'].items()}
            bwd = {k: v.to(self.device) for k, v in batch['features_backward'].items()}
            lengths = batch['lengths'].to(self.device)
            node_mask = batch['node_mask'].to(self.device)
            y = batch['labels'].to(self.device)

            logits = self.model(fwd, bwd, node_mask, lengths)
            loss = self.crit(logits, y).item()
            tot_loss += loss * y.size(0)  # Weight by batch size

            # Compute top-1 and top-5
            num_classes = logits.size(1)
            k = min(5, num_classes)

            top1_correct += (logits.argmax(1) == y).sum().item()

            _, topk_preds = logits.topk(k, dim=1)
            top5_correct += (topk_preds == y.unsqueeze(1)).any(dim=1).sum().item()

            total += y.size(0)

            if self.shutdown_handler.is_interrupted():
                break

        avg_loss = tot_loss / max(total, 1)
        top1_acc = top1_correct / max(total, 1)
        top5_acc = top5_correct / max(total, 1)

        return avg_loss, top1_acc, top5_acc  # FIXED: Return all three metrics

    def train(self, train_loader, val_loader, mlflow_log=True):
        best_val = 0
        patience = 20
        patience_counter = 0

        for ep in range(self.epochs):
            # Training phase
            self.model.train()
            tot_loss = 0
            top1_correct = 0
            top5_correct = 0
            total = 0

            for batch in tqdm(train_loader, desc=f'Epoch {ep+1}/{self.epochs}'):
                loss, c1, c5, n = self.step(batch)
                tot_loss += loss * n  # Weight by batch size
                top1_correct += c1
                top5_correct += c5
                total += n
                if self.shutdown_handler.is_interrupted():
                    break

            if self.shutdown_handler.is_interrupted():
                break

            tr_loss = tot_loss / max(total, 1)
            tr_top1 = top1_correct / max(total, 1)
            tr_top5 = top5_correct / max(total, 1)

            # Validation phase - FIXED: Add actual evaluation
            va_loss, va_top1, va_top5 = self.eval_loop(val_loader)

            # Early stopping
            if va_top1 > best_val:  # FIXED: Use va_top1 instead of undefined va_acc
                best_val = va_top1
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_stable_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {ep+1}")
                    break

            # History logging
            self.hist['train_loss'].append(tr_loss)
            self.hist['val_loss'].append(va_loss)
            self.hist['train_acc'].append(tr_top1)
            self.hist['val_acc'].append(va_top1)
            self.hist['train_top5'].append(tr_top5)
            self.hist['val_top5'].append(va_top5)

            print(f"Epoch {ep+1}: Train Loss={tr_loss:.4f} Top1={tr_top1:.2%} Top5={tr_top5:.2%} | "
                f"Val Loss={va_loss:.4f} Top1={va_top1:.2%} Top5={va_top5:.2%} | "
                f"LR={self.optim.param_groups[0]['lr']:.6f}")

            self.sched.step()

            if mlflow_log:
                mlflow.log_metrics({
                    'train_loss': tr_loss,
                    'val_loss': va_loss,
                    'train_top1_accuracy': tr_top1,
                    'val_top1_accuracy': va_top1,
                    'train_top5_accuracy': tr_top5,
                    'val_top5_accuracy': va_top5,
                    'learning_rate': self.optim.param_groups[0]['lr']
                }, step=ep)

        return best_val

class GracefulShutdown:
    """
    Handles graceful shutdown on keyboard interrupt (Ctrl+C).
    Saves model, metrics, and completes MLflow logging before exiting.
    """
    def __init__(self):
        self.interrupted = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, sig, frame):
        print("\n" + "="*80)
        print("[GRACEFUL SHUTDOWN] Keyboard interrupt detected (Ctrl+C)")
        print("="*80)
        self.interrupted = True

    def is_interrupted(self):
        """Check if shutdown has been requested."""
        return self.interrupted

# ===========================
# Main
# ===========================
def main():
    set_seed(42)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA_DIR = './word_landmarks_extracted'

    # REVISED HYPERPARAMETERS for 150 classes with stability
    TOP_K = 150
    NUM_BLOCKS = 5
    HIDDEN = 256
    TEMPORAL_KERNEL = 5
    DROPOUT = 0.3
    EPOCHS = 200
    LR = 5e-5  # FIXED: Increased from 1e-5
    BATCH = 4
    WEIGHT_DECAY = 1e-4  # FIXED: Decreased from 2e-4

    # System
    PREFETCH_FACTOR = 4
    NUM_WORKERS = 6

    RUN_NAME = f'Top{TOP_K} ALL-STREAMS: 7 Streams × 5 Blocks × 256 Hidden'

    print('=' * 80)
    print(RUN_NAME)
    print('=' * 80)

    # MLflow
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME', 'roman')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD', 'SignNet')
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'https://mlflow.schlaepfer.me'))
    mlflow.set_experiment('SignNetWord')
    run = mlflow.start_run(log_system_metrics=True, run_name=RUN_NAME)

    shutdown_handler = GracefulShutdown()

    try:
        mlflow.log_params({
            'epochs': EPOCHS, 'batch_size': BATCH, 'hidden': HIDDEN,
            'num_blocks': NUM_BLOCKS, 'temporal_kernel': TEMPORAL_KERNEL,
            'dropout': DROPOUT, 'lr': LR, 'weight_decay': WEIGHT_DECAY,
            'lr_schedule': 'MultiStep_150_200', 'device': DEVICE
        })
        mlflow.log_params({
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'os': platform.system(),
            'cpu_count': os.cpu_count(),
            'total_ram_gb': round(psutil.virtual_memory().total / (1024**3), 2)
        })
        if torch.cuda.is_available():
            mlflow.log_params({
                'gpu_name': torch.cuda.get_device_name(0),
                'cuda_version': torch.version.cuda,
                'gpu_mem_gb': round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            })

        train_augmentation = SkeletonAugmentation(
            mirror_prob=0.5,
            rotation_range=(-10, 10),
            scale_range=(0.9, 1.1),
            shift_range=(-0.1, 0.1),
            apply_prob=0.8  # Apply augmentation to 80% of training samples
        )

        # Build dataset with augmentation for training
        fe_train = Skeleton27FeatureExtractor(conf_valid_thresh=0.5, augmentation=train_augmentation)
        fe_val = Skeleton27FeatureExtractor(conf_valid_thresh=0.5, augmentation=None)  # No augmentation for val/test

        fe_clean = Skeleton27FeatureExtractor(conf_valid_thresh=0.5, augmentation=None)
        full_base = TrueSkeletonDataset(DATA_DIR, fe_clean, debug=True, is_training=False)  # Clean for vocab
        top_k_words, _ = build_topk_vocabulary(full_base.files, K=TOP_K, debug=True)

        # Create datasets with appropriate augmentation settings
        full_train = TopKTrueSkeletonDataset(full_base, top_k_words, feature_extractor=fe_train,
                                   debug=True, is_training=True)

        tr_idx, va_idx, te_idx = split_dataset_stratified(full_train, 0.7, 0.15, 0.15, random_state=42)

        # Training set: uses augmentation
        train_ds = SubsetDataset(full_train, tr_idx)

        # Val/Test sets: no augmentation
        full_val = TopKTrueSkeletonDataset(full_base, top_k_words, feature_extractor=fe_val, debug=True, is_training=False)
        val_ds = SubsetDataset(full_val, va_idx)
        test_ds = SubsetDataset(full_val, te_idx)

        collate = TrueSkeletonCollator(conf_valid_thresh=full_train.fe.conf_valid_thresh)

        train_loader = DataLoader(
            train_ds, batch_size=BATCH, shuffle=True,
            num_workers=4, pin_memory=True,
            collate_fn=collate, prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH, shuffle=False,
            num_workers=4, pin_memory=True,
            collate_fn=collate, prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=True
        )
        test_loader = DataLoader(
            test_ds, batch_size=BATCH, shuffle=False,
            num_workers=4, pin_memory=True,
            collate_fn=collate, prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=True
        )

        # Model
        num_classes = len(full_train.word_to_idx)
        print(f"\n[INFO] num_classes = {num_classes}")

        model = StableBidirectionalGCN(
            num_classes=num_classes,
            hidden=HIDDEN,
            num_blocks=NUM_BLOCKS,
            temporal_kernel=TEMPORAL_KERNEL,
            dropout=DROPOUT
        )

        mlflow.log_param('top_k', TOP_K)
        with open('topk_words.txt', 'w') as f:
            for w in sorted(full_train.word_to_idx.keys()):
                f.write(f"{w}\n")
        mlflow.log_artifact('topk_words.txt')

        # Train
        trainer = StableTrainer(
            model, shutdown_handler, device=DEVICE, lr=LR, wd=WEIGHT_DECAY,
            epochs=EPOCHS
        )


        # DIAGNOSTIC: Check first batch
        print("\n[DIAGNOSTIC] Checking first batch...")
        for batch in train_loader:
            print("Batch shapes:")
            for k, v in batch['features_forward'].items():
                print(f"  {k}: {v.shape}, range=[{v.min():.4f}, {v.max():.4f}], mean={v.mean():.4f}")

            # Check for extreme values
            all_ok = True
            for k, v in batch['features_forward'].items():
                if v.abs().max() > 5.0:
                    print(f"  ⚠️  WARNING: {k} has extreme values!")
                    all_ok = False
                if torch.isnan(v).any():
                    print(f"  ⚠️  WARNING: {k} contains NaN!")
                    all_ok = False

            if all_ok:
                print("  ✓ All inputs look reasonable")
            break
        print()



        best_val = trainer.train(train_loader, val_loader)
        print(f"Best Val Acc: {best_val:.2%}")

        # Test
        test_loss, test_top1, test_top5 = trainer.eval_loop(test_loader)
        print(f"Test: Loss={test_loss:.4f} Top1={test_top1:.2%} Top5={test_top5:.2%}")
        mlflow.log_metrics({
            'test_loss': test_loss,
            'test_top1_accuracy': test_top1,
            'test_top5_accuracy': test_top5
        })

        asyncio.run(send_message(
            f"Training complete!\n"
            f"Best Val Acc: {best_val:.2%}\n"
            f"Test Top1: {test_top1:.2%} Top5: {test_top5:.2%}\n"
            f"Run: {RUN_NAME}",
            CHAT_ID
        ))

        # Save split
        with open('dataset_split_enhanced.json', 'w') as f:
            json.dump({
                'train': tr_idx, 'val': va_idx, 'test': te_idx,
                'word_to_idx': full_train.word_to_idx
            }, f, indent=2)
        mlflow.log_artifact('dataset_split_enhanced.json')

    finally:
        mlflow.end_run()


if __name__ == '__main__':
    main()
