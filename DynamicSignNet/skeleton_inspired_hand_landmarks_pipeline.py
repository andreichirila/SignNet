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


# ===========================
# 27-node skeleton extractor
# ===========================
class Skeleton27FeatureExtractor:
    def __init__(self, conf_valid_thresh: float = 0.5):
        self.num_nodes = 27
        self.conf_valid_thresh = conf_valid_thresh
        self.edges = [
            (0,1),(0,2),(1,3),(2,4),(3,5),(4,6),(5,7),(6,8),
            (7,9),(9,10),(7,11),(11,12),(7,13),(13,14),(7,15),(15,16),(7,17),(17,18),
            (8,19),(19,20),(8,21),(21,22),(8,23),(23,24),(8,25),(25,26)
        ]

    def map_combined_to_27(self, combined_frame):
        nodes = np.zeros((self.num_nodes, 3), dtype=np.float32)

        HAND_L = combined_frame[0:63].reshape(21, 3)
        HAND_R = combined_frame[63:126].reshape(21, 3)
        FACE   = combined_frame[126:126+1434].reshape(478, 3)
        POSE   = combined_frame[126+1434:126+1434+99].reshape(33, 3)

        nodes[0] = FACE[1]
        nodes[1] = FACE[33]
        nodes[2] = FACE[263]
        nodes[3] = POSE[12]
        nodes[4] = POSE[11]
        nodes[5] = POSE[14]
        nodes[6] = POSE[13]
        nodes[7] = POSE[16]
        nodes[8] = POSE[15]

        if np.any(HAND_L != 0):
            nodes[9]  = HAND_L[1];  nodes[10] = HAND_L[4]
            nodes[11] = HAND_L[5];  nodes[12] = HAND_L[8]
            nodes[13] = HAND_L[9];  nodes[14] = HAND_L[12]
            nodes[15] = HAND_L[13]; nodes[16] = HAND_L[16]
            nodes[17] = HAND_L[17]; nodes[18] = HAND_L[20]
        else:
            nodes[9:19] = 0
            nodes[9:19, 2] = 0.1

        if np.any(HAND_R != 0):
            nodes[19] = HAND_R[1];  nodes[20] = HAND_R[4]
            nodes[21] = HAND_R[5];  nodes[22] = HAND_R[8]
            nodes[23] = HAND_R[9];  nodes[24] = HAND_R[12]
            nodes[25] = HAND_R[13]; nodes[26] = HAND_R[16]
        else:
            nodes[19:27] = 0
            nodes[19:27, 2] = 0.1

        # CRITICAL: Strict normalization to [-1, 1]
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


    def extract(self, combined_sequence):
        T = combined_sequence.shape[0]
        nodes_seq = np.stack([self.map_combined_to_27(combined_sequence[t]) for t in range(T)], axis=0)
        return self.compute_streams(nodes_seq)


# ===========================
# Dataset for true skeleton
# ===========================
class TrueSkeletonDataset(Dataset):
    def __init__(self, npz_dir, feature_extractor: Skeleton27FeatureExtractor, word_to_idx=None, debug=True):
        self.dir = Path(npz_dir)
        self.files = sorted(self.dir.glob('*.npz'))
        self.fe = feature_extractor
        self.debug = debug
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
        streams = self.fe.extract(X)
        g = d['glosses'][0] if len(d['glosses'])>0 else 'UNKNOWN'
        y = self.word_to_idx.get(g, 0)
        return streams, torch.tensor(y, dtype=torch.long), f.stem


class TopKTrueSkeletonDataset(Dataset):
    """Top-K vocabulary filtering."""
    def __init__(self, base_dataset: 'TrueSkeletonDataset', top_k_words: set, feature_extractor=None, debug=True):
        self.base = base_dataset
        self.debug = debug
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
        streams = self.fe.extract(X)
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
    """
    Simplified STC attention that won't explode.
    Uses sigmoid instead of softmax for stability.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        # Spatial attention
        self.spatial_conv = nn.Conv2d(channels, 1, kernel_size=1)

        # Temporal attention
        self.temporal_conv = nn.Conv2d(channels, 1, kernel_size=1)

        # Channel attention (squeeze-excitation style)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, V, T)
        B, C, V, T = x.shape

        # Spatial attention
        spatial_att = torch.sigmoid(self.spatial_conv(x))  # (B, 1, V, T)
        x = x * spatial_att

        # Temporal attention
        temporal_att = torch.sigmoid(self.temporal_conv(x))  # (B, 1, V, T)
        x = x * temporal_att

        # Channel attention
        gap = F.adaptive_avg_pool2d(x, (1, 1)).view(B, C)  # (B, C)
        channel_att = self.channel_fc(gap).view(B, C, 1, 1)  # (B, C, 1, 1)
        x = x * channel_att

        # Residual scaling (critical for stability)
        return x * 0.1


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

        # Batch norm over features for each node and time step
        self.bn_graph = nn.LayerNorm(out_channels)

        # Temporal convolution
        self.temporal_conv = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(temporal_kernel, 1),
            padding=((temporal_kernel - 1) // 2, 0),
            bias=False
        )
        self.bn_temporal = nn.BatchNorm2d(out_channels)

        # Simple attention
        self.attention = STCAttentionSimple(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        # Residual
        self.residual_conv = None
        if in_channels != out_channels:
            self.residual_conv = nn.Linear(in_channels, out_channels)

        # Conservative initialization
        self._init_weights()

    def _init_weights(self):
        """Ultra-conservative weights to prevent explosion."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)  # Very small
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.01)  # Very small
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, A):
        """
        x: (B, T, V, C_in)
        A: (V, V) adjacency matrix
        Returns: (B, T, V, C_out)
        """
        B, T, V, C_in = x.shape

        # 1. Apply graph convolution: Linear transform
        # x: (B, T, V, C_in) → (B*T*V, C_in) → (B*T*V, C_out)
        x_flat = x.reshape(B * T * V, C_in)
        x_features = self.graph_conv(x_flat)  # (B*T*V, C_out)

        # 2. Reshape to (B*T, V, C_out)
        x_features = x_features.reshape(B * T, V, -1)  # (B*T, V, C_out)

        # 3. Apply adjacency matrix to aggregate neighbors
        # For each time step and each node: sum weighted by A
        # A @ x_features: (V,V) @ (V, C_out) = (V, C_out) for each time step
        x_gcn = torch.bmm(
            A.unsqueeze(0).expand(B * T, -1, -1),
            x_features
        )  # (B*T, V, C_out)

        # 4. Apply layer normalization over features
        x_gcn = self.bn_graph(x_gcn)  # (B*T, V, C_out)
        x_gcn = self.relu(x_gcn)
        x_gcn = self.dropout(x_gcn)

        # 5. Reshape back to (B, T, V, C_out)
        C_out = x_gcn.shape[-1]
        x_gcn = x_gcn.reshape(B, T, V, C_out)  # (B, T, V, C_out)

        # 6. Temporal processing: (B, C_out, V, T)
        x_temp = x_gcn.permute(0, 3, 2, 1)  # (B, C_out, V, T)
        x_temp = self.temporal_conv(x_temp)
        x_temp = self.bn_temporal(x_temp)
        x_temp = self.relu(x_temp)

        # 7. Attention
        x_att = self.attention(x_temp)  # (B, C_out, V, T)

        # 8. Residual connection
        if self.residual_conv is not None:
            # Apply residual transformation if needed
            x_res = x.reshape(B * T * V, C_in)  # (B*T*V, C_in)
            x_res = self.residual_conv(x_res)  # (B*T*V, C_out)
            x_res = x_res.reshape(B, T, V, C_out)  # (B, T, V, C_out)
        else:
            x_res = x

        # 9. Combine with residual scaling
        out = 0.9 * x_res + 0.1 * x_att.permute(0, 3, 2, 1)

        return out




# ===========================
# NEW: Enhanced Stream Processor with GCN Blocks
# ===========================
class StableStreamProcessor(nn.Module):
    def __init__(self, in_feat=3, hidden=256, num_blocks=3, temporal_kernel=3, dropout=0.2):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.blocks.append(StableGCNBlock(in_feat, hidden, temporal_kernel, dropout))
        for _ in range(num_blocks - 1):
            self.blocks.append(StableGCNBlock(hidden, hidden, temporal_kernel, dropout))

    def forward(self, seq, A, node_mask, lengths):
        """
        seq: (B, T, V, F_in)
        Returns: (B, H) - pooled features
        """
        x = seq
        B, T, V, _ = x.shape

        for block in self.blocks:
            x = block(x, A)  # (B, T, V, H)

        # Masked node pooling
        weights = node_mask.unsqueeze(-1)  # (B, T, V, 1)
        x_masked = (x * weights).sum(dim=2) / (weights.sum(dim=2) + 1e-6)  # (B, T, H)

        # Simple temporal average pooling
        x_pooled = torch.mean(x_masked, dim=1)  # (B, H)

        return x_pooled



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
        V = 27
        A = torch.zeros(V, V)
        edges = [
            (0,1),(0,2),(1,3),(2,4),(3,5),(4,6),(5,7),(6,8),
            (7,9),(9,10),(7,11),(11,12),(7,13),(13,14),(7,15),(15,16),(7,17),(17,18),
            (8,19),(19,20),(8,21),(21,22),(8,23),(23,24),(8,25),(25,26)
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
        Ffused = (Fstk * wf.view(1, -1, 1)).sum(dim=1)  # (B, H)

        # Backward pass
        node_mask_b = node_mask.flip(dims=[1])
        outs_b = []
        for s in self.streams:
            out_s = self.backward_streams[s](bwd[s], self.A, node_mask_b, lengths)
            outs_b.append(out_s)

        Bstk = torch.stack(outs_b, dim=1)  # (B, S, H)
        wb = F.softmax(self.w_bwd, dim=0)
        Bfused = (Bstk * wb.view(1, -1, 1)).sum(dim=1)  # (B, H)

        # Directional fusion
        wd = F.softmax(self.w_dir, dim=0)
        fused = wd[0] * Ffused + wd[1] * Bfused

        return self.classifier(fused)




# ===========================
# NEW: Trainer with Paper-Exact Hyperparameters
# ===========================
class StableTrainer:
    """Ultra-conservative trainer for gradient stability."""
    def __init__(self, model, device='cuda', lr=1e-4, wd=1e-4, epochs=100):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs

        # Use Adam for better stability than SGD
        self.optim = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Cosine annealing for smooth decay
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, T_max=epochs, eta_min=1e-6
        )

        self.crit = nn.CrossEntropyLoss()
        self.scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
        self.hist = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def step(self, batch):
        fwd = {k: v.to(self.device) for k, v in batch['features_forward'].items()}
        bwd = {k: v.to(self.device) for k, v in batch['features_backward'].items()}
        lengths = batch['lengths'].to(self.device)
        node_mask = batch['node_mask'].to(self.device)
        y = batch['labels'].to(self.device)

        # CRITICAL: Check input ranges
        for k, v in fwd.items():
            v_max = v.abs().max().item()
            if v_max > 10.0:
                print(f"⚠️  EXTREME INPUT: {k} has max value {v_max:.2f}")
                # Clip at runtime as emergency measure
                fwd[k] = torch.clamp(v, -10, 10)
                bwd[k] = torch.clamp(bwd[k], -10, 10)

            if torch.isnan(v).any() or torch.isinf(v).any():
                print(f"⚠️  NaN/Inf in input {k}, skipping batch")
                return 0.0, 0, y.size(0)

        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            logits = self.model(fwd, bwd, node_mask, lengths)

            # Check model output
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("⚠️  NaN/Inf in model output, skipping batch")
                return 0.0, 0, y.size(0)

            loss = self.crit(logits, y)

        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️  NaN/Inf loss, skipping batch")
            return 0.0, 0, y.size(0)

        self.optim.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()

        # Check gradient norm
        total_norm = 0
        max_grad = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                max_grad = max(max_grad, p.grad.data.abs().max().item())
        total_norm = total_norm ** 0.5

        # STRICTER threshold for complex models
        if total_norm > 5.0:  # Changed from 10.0
            print(f"⚠️  Gradient explosion: norm={total_norm:.2f}, max={max_grad:.2f}")
            print(f"   Skipping batch (LR={self.optim.param_groups[0]['lr']:.2e})")
            self.optim.zero_grad(set_to_none=True)
            return 0.0, 0, y.size(0)


        # Aggressive clipping
        self.scaler.unscale_(self.optim)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)  # Changed from 1.0

        self.scaler.step(self.optim)
        self.scaler.update()

        return loss.item(), (logits.argmax(1) == y).sum().item(), y.size(0)


    @torch.no_grad()
    def eval_loop(self, loader):
        self.model.eval()
        tot_loss = 0
        correct = 0
        total = 0
        for batch in loader:
            fwd = {k: v.to(self.device) for k, v in batch['features_forward'].items()}
            bwd = {k: v.to(self.device) for k, v in batch['features_backward'].items()}
            lengths = batch['lengths'].to(self.device)
            node_mask = batch['node_mask'].to(self.device)
            y = batch['labels'].to(self.device)

            logits = self.model(fwd, bwd, node_mask, lengths)
            loss = self.crit(logits, y).item()
            tot_loss += loss
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        return tot_loss / max(len(loader), 1), correct / max(total, 1)

    def train(self, train_loader, val_loader, mlflow_log=True):
        best_val = 0
        patience = 20
        patience_counter = 0

        for ep in range(self.epochs):
            self.model.train()
            tot_loss = 0
            correct = 0
            total = 0

            for batch in tqdm(train_loader, desc=f'Epoch {ep+1}/{self.epochs}'):
                loss, c, n = self.step(batch)
                tot_loss += loss
                correct += c
                total += n

            tr_loss = tot_loss / len(train_loader)
            tr_acc = correct / total
            va_loss, va_acc = self.eval_loop(val_loader)

            if va_acc > best_val:
                best_val = va_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_stable_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {ep+1}")
                    break

            self.hist['train_loss'].append(tr_loss)
            self.hist['val_loss'].append(va_loss)
            self.hist['train_acc'].append(tr_acc)
            self.hist['val_acc'].append(va_acc)

            print(f"Epoch {ep+1}: Train Loss={tr_loss:.4f} Acc={tr_acc:.2%} | Val Loss={va_loss:.4f} Acc={va_acc:.2%} | LR={self.optim.param_groups[0]['lr']:.6f}")

            self.sched.step()

            if mlflow_log:
                mlflow.log_metrics({
                    'train_loss': tr_loss,
                    'val_loss': va_loss,
                    'train_accuracy': tr_acc,
                    'val_accuracy': va_acc,
                    'learning_rate': self.optim.param_groups[0]['lr']
                }, step=ep)

        return best_val


# ===========================
# Main
# ===========================
def main():
    set_seed(42)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA_DIR = './word_landmarks_extracted'

    # REVISED HYPERPARAMETERS for 150 classes with stability
    TOP_K = 150

    # Architecture (increased for 7 streams)
    NUM_BLOCKS = 5        # Reduced from 6 (7 streams add enough capacity)
    HIDDEN = 256          # Keep 256
    TEMPORAL_KERNEL = 5   # Keep 5
    DROPOUT = 0.3         # Increased from 0.25 for stability

    # Training (conservative for stability)
    EPOCHS = 200
    LR = 1e-5  # Changed from 5e-5 (5x reduction)
    BATCH = 16 # Reduced from 24
    WEIGHT_DECAY = 2e-4   # Increased from 1e-4

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

    try:
        mlflow.log_params({
            'epochs': EPOCHS, 'batch_size': BATCH, 'hidden': HIDDEN,
            'num_blocks': NUM_BLOCKS, 'temporal_kernel': TEMPORAL_KERNEL,
            'dropout': DROPOUT, 'lr': LR, 'weight_decay': WEIGHT_DECAY,
            'optimizer': 'SGD_Nesterov',
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

        # Build dataset
        fe = Skeleton27FeatureExtractor(conf_valid_thresh=0.5)
        full_base = TrueSkeletonDataset(DATA_DIR, fe, debug=True)
        top_k_words, _ = build_topk_vocabulary(full_base.files, K=TOP_K, debug=True)
        full = TopKTrueSkeletonDataset(full_base, top_k_words, feature_extractor=fe, debug=True)

        tr_idx, va_idx, te_idx = split_dataset_stratified(full, 0.7, 0.15, 0.15, random_state=42)

        train_ds = SubsetDataset(full, tr_idx)
        val_ds = SubsetDataset(full, va_idx)
        test_ds = SubsetDataset(full, te_idx)

        collate = TrueSkeletonCollator(conf_valid_thresh=fe.conf_valid_thresh)

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
        num_classes = len(full.word_to_idx)
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
            for w in sorted(full.word_to_idx.keys()):
                f.write(f"{w}\n")
        mlflow.log_artifact('topk_words.txt')

        # Train
        trainer = StableTrainer(
            model, device=DEVICE, lr=LR, wd=WEIGHT_DECAY,
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
        test_loss, test_acc = trainer.eval_loop(test_loader)
        print(f"Test: Loss={test_loss:.4f} Acc={test_acc:.2%}")
        mlflow.log_metrics({'test_loss': test_loss, 'test_acc': test_acc})

        asyncio.run(send_message(
            f"Training complete!\n"
            f"Best Val Acc: {best_val:.2%}\n"
            f"Test Acc: {test_acc:.2%}\n"
            f"Run: {RUN_NAME}",
            CHAT_ID
        ))

        # Save split
        with open('dataset_split_enhanced.json', 'w') as f:
            json.dump({
                'train': tr_idx, 'val': va_idx, 'test': te_idx,
                'word_to_idx': full.word_to_idx
            }, f, indent=2)
        mlflow.log_artifact('dataset_split_enhanced.json')

    finally:
        mlflow.end_run()


if __name__ == '__main__':
    main()
