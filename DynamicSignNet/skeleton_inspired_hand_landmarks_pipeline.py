#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
True-Skeleton Bidirectional Multi-Stream GCN for Isolated Sign Recognition
Patched version:
- PACKED LSTM with true sequence lengths
- Mask-aware weighted node pooling using confidence-derived node mask
- Fixed edge_distance (magnitudes) vs bone_vectors (signed vectors)
- Mask-aware velocities/accelerations
- Backward stream uses flipped node mask
- Extra sanity logging and class distribution printouts
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

# ===========================
# Utils: Reproducibility
# ===========================
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
    Works with datasets having .files and a word_to_idx mapping.
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

    # Build labels
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

    # Enforce min samples per class (>=4 safer for 2-level stratification)
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

    # First split
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

    # Ensure tmp has >=2 per class for second stratify
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

    # Second split (val vs test)
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
# 27-node skeleton extractor
# ===========================
class Skeleton27FeatureExtractor:
    def __init__(self, conf_valid_thresh: float = 0.5):
        self.num_nodes = 27
        self.conf_valid_thresh = conf_valid_thresh
        # edges per paper (undirected)
        self.edges = [
            (0,1),(0,2),  # nose-eyes
            (1,3),(2,4),  # eyes-shoulders
            (3,5),(4,6),  # shoulders-elbows
            (5,7),(6,8),  # elbows-wrists
            # left hand: base->tip per finger
            (7,9),(9,10),(7,11),(11,12),(7,13),(13,14),(7,15),(15,16),(7,17),(17,18),
            # right hand: base->tip per finger
            (8,19),(19,20),(8,21),(21,22),(8,23),(23,24),(8,25),(25,26)
        ]

    def map_combined_to_27(self, combined_frame):
        """
        Map combined = [hand_lms(126), face_lms(1434), pose_lms(99)] to 27 nodes.
        Output: (27, 3) with (x,y,confidence). Confidence in [0,1].
        """
        nodes = np.zeros((self.num_nodes, 3), dtype=np.float32)

        HAND_L = combined_frame[0:63].reshape(21, 3)
        HAND_R = combined_frame[63:126].reshape(21, 3)
        FACE   = combined_frame[126:126+1434].reshape(478, 3)
        POSE   = combined_frame[126+1434:126+1434+99].reshape(33, 3)

        # Face keypoints
        nodes[0] = FACE[1]   # nose
        nodes[1] = FACE[33]  # left eye
        nodes[2] = FACE[263] # right eye

        # Pose keypoints
        nodes[3] = POSE[12]  # left shoulder
        nodes[4] = POSE[11]  # right shoulder
        nodes[5] = POSE[14]  # left elbow
        nodes[6] = POSE[13]  # right elbow
        nodes[7] = POSE[16]  # left wrist
        nodes[8] = POSE[15]  # right wrist

        # LEFT HAND: treat "any nonzero" as data present
        if np.any(HAND_L != 0):
            nodes[9]  = HAND_L[1];  nodes[10] = HAND_L[4]   # thumb
            nodes[11] = HAND_L[5];  nodes[12] = HAND_L[8]   # index
            nodes[13] = HAND_L[9];  nodes[14] = HAND_L[12]  # middle
            nodes[15] = HAND_L[13]; nodes[16] = HAND_L[16]  # ring
            nodes[17] = HAND_L[17]; nodes[18] = HAND_L[20]  # pinky
        else:
            nodes[9:19] = 0
            # very low confidence to mark invalid for mask thresholding
            nodes[9:19, 2] = 0.1

        # RIGHT HAND
        if np.any(HAND_R != 0):
            nodes[19] = HAND_R[1];  nodes[20] = HAND_R[4]
            nodes[21] = HAND_R[5];  nodes[22] = HAND_R[8]
            nodes[23] = HAND_R[9];  nodes[24] = HAND_R[12]
            nodes[25] = HAND_R[13]; nodes[26] = HAND_R[16]
        else:
            nodes[19:27] = 0
            nodes[19:27, 2] = 0.1

        # Clamp and sanitize ranges
        nodes[:, 0:2] = np.clip(nodes[:, 0:2], -3, 3)
        nodes[:, 2]   = np.clip(nodes[:, 2], 0, 1)

        # fallback if everything zero coords
        if np.all(nodes[:, :2] == 0):
            nodes[:, 2] = 0.0  # keep invalid; mask will ignore

        return nodes

    # ==== PATCH: mask-aware streams; fixed edge_distance magnitudes ====
    def compute_streams(self, nodes_seq):
        """
        nodes_seq: (T,V,3) with (x, y, conf) in [0,1]
        Returns dict of streams each (T,V,3)
        - keypoint_coords: (x,y,conf)
        - edge_distance:   (0,0,dist)   where dist is max magnitude to incident neighbor
        - bone_vectors:    (dx,dy,conf_agg) average vectors over incident edges
        - *_velocity, *_accel: mask-aware temporal deltas
        """
        T, V = nodes_seq.shape[0], nodes_seq.shape[1]
        K = nodes_seq.copy().astype(np.float32)  # (T,V,3)
        conf = K[..., 2]                         # (T,V)
        valid = (conf >= self.conf_valid_thresh).astype(np.float32)  # (T,V)

        # Edge-based features
        # Accumulate per node j over incident edges
        B = np.zeros_like(K)   # (dx, dy, conf_agg)
        D = np.zeros_like(K)   # (0, 0, dist_agg)
        b_count = np.zeros((T, V, 1), dtype=np.float32)
        d_count = np.zeros((T, V, 1), dtype=np.float32)

        for (i, j) in self.edges:
            diff = K[:, j, :2] - K[:, i, :2]             # (T,2)
            dist = np.linalg.norm(diff, axis=-1, keepdims=True)  # (T,1)
            conf_ij = 0.5 * (K[:, j, 2:3] + K[:, i, 2:3])        # (T,1)

            # bone vectors: average vector, aggregate confidence
            B[:, j, :2] += diff
            B[:, j,  2:3] += conf_ij
            b_count[:, j, :] += 1.0

            # edge distance: aggregate by max of magnitude (store in z)
            # We'll keep the maximum distance among incident edges
            D[:, j, 2:3] = np.maximum(D[:, j, 2:3], dist)
            d_count[:, j, :] = np.maximum(d_count[:, j, :], 1.0)

        # Avoid divide-by-zero; average bone vectors and conf
        mask_b = (b_count > 0).astype(np.float32)
        B[:, :, :2] = np.where(mask_b.astype(bool), B[:, :, :2] / np.maximum(b_count, 1e-6), 0.0)
        B[:, :,  2] = np.where(mask_b[...,0].astype(bool), B[:, :, 2] / np.maximum(b_count[...,0], 1e-6), 0.0)

        # D: put zeros in x,y; z already holds max distance
        D[:, :, 0:2] = 0.0
        # Optional normalization for D.z (keep raw for now)

        # Temporal deltas with validity
        KV = np.zeros_like(K); BV = np.zeros_like(B)
        KA = np.zeros_like(K); BA = np.zeros_like(B)

        # Validity for derived streams (use node validity)
        valid_b = (B[..., 2] > 0).astype(np.float32)  # derived from aggregated confidence
        for t in range(1, T):
            pair_mask_k = (valid[t] * valid[t-1])[..., None]  # (T,V,1)
            KV[t] = (K[t] - K[t-1]) * pair_mask_k

            pair_mask_b = (valid_b[t] * valid_b[t-1])[..., None]
            BV[t] = (B[t] - B[t-1]) * pair_mask_b

        for t in range(2, T):
            tri_mask_k = (valid[t] * valid[t-1] * valid[t-2])[..., None]
            KA[t] = (KV[t] - KV[t-1]) * tri_mask_k

            tri_mask_b = (valid_b[t] * valid_b[t-1] * valid_b[t-2])[..., None]
            BA[t] = (BV[t] - BV[t-1]) * tri_mask_b

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
        X = d['landmarks'].astype(np.float32)  # (T, D)
        streams = self.fe.extract(X)  # dict of (T,27,3)
        g = d['glosses'][0] if len(d['glosses'])>0 else 'UNKNOWN'
        y = self.word_to_idx.get(g, 0)
        return streams, torch.tensor(y, dtype=torch.long), f.stem


class TopKTrueSkeletonDataset(Dataset):
    """
    Wraps TrueSkeletonDataset but keeps only samples whose gloss is in top_k_words.
    Builds a compact word_to_idx over that subset (0..K-1).
    """
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


# ==== PATCH: collator with lengths + node mask (confidence > 0.5) ====
class TrueSkeletonCollator:
    def __init__(self, conf_valid_thresh: float = 0.5):
        self.conf_valid_thresh = conf_valid_thresh

    def __call__(self, batch):
        # batch: list of (streams_dict, y, id)
        max_T = max(next(iter(b[0].values())).shape[0] for b in batch)
        stream_names = list(batch[0][0].keys())

        fwd = {s: [] for s in stream_names}
        bwd = {s: [] for s in stream_names}
        lengths = []
        node_masks = []  # (B, T, V)

        labels, ids = [], []

        for streams, y, sid in batch:
            T = next(iter(streams.values())).shape[0]
            lengths.append(T)
            labels.append(y)
            ids.append(sid)

            # Node validity mask from keypoint_coords confidence
            kc = streams['keypoint_coords']  # (T,V,3)
            mask = (kc[..., 2] >= self.conf_valid_thresh).astype(np.float32)  # (T,V)
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
            fwd[s] = torch.tensor(np.stack(fwd[s], axis=0), dtype=torch.float32)  # (B,T,V,F)
            bwd[s] = torch.tensor(np.stack(bwd[s], axis=0), dtype=torch.float32)  # (B,T,V,F)

        lengths = torch.tensor(lengths, dtype=torch.long)  # (B,)
        node_masks = torch.tensor(np.stack(node_masks, axis=0), dtype=torch.float32)  # (B,T,V)

        return {
            'features_forward': fwd,
            'features_backward': bwd,
            'lengths': lengths,
            'node_mask': node_masks,
            'labels': torch.stack(labels),
            'ids': ids
        }


# ===========================
# GCN backbone
# ===========================
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        self.b = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.W)
    def forward(self, x, A):
        # x: (B,V,Fin), A: (V,V)
        support = x @ self.W
        out = A.unsqueeze(0) @ support
        return out + self.b


class GCNLayer(nn.Module):
    def __init__(self, in_f, out_f, p=0.2):
        super().__init__()
        self.gc = GraphConvolution(in_f, out_f)
        self.do = nn.Dropout(p)
    def forward(self, x, A):
        x = self.gc(x, A)
        x = F.relu(x)
        return self.do(x)


# ==== PATCH: stream processor with weighted node pooling and packed LSTM ====
class SkeletonGCNStreamOptimized(nn.Module):
    """Vectorized temporal processing with node-masked pooling and packed LSTM."""
    def __init__(self, in_feat=3, hidden=64, layers=3, p=0.2):
        super().__init__()
        self.layers = nn.ModuleList(
            [GCNLayer(in_feat, hidden, p)] +
            [GCNLayer(hidden, hidden, p) for _ in range(layers-1)]
        )
        self.temporal = nn.LSTM(
            input_size=hidden, hidden_size=hidden, num_layers=2,
            batch_first=True, dropout=p, bidirectional=False
        )

    def forward(self, seq, A, node_mask, lengths):
        """
        seq: (B, T, V, Fin)
        node_mask: (B, T, V) in {0,1}
        lengths: (B,)
        """
        B, T, V, F = seq.shape
        # GCN over nodes (vectorized over B*T)
        seq_flat = seq.reshape(B*T, V, F)
        x = seq_flat
        for layer in self.layers:
            x = layer(x, A)  # (B*T, V, H)
        H = x.shape[-1]
        x = x.reshape(B, T, V, H)

        # Weighted pooling over nodes using mask
        eps = 1e-6
        w = node_mask.unsqueeze(-1)  # (B,T,V,1)
        x = (x * w).sum(dim=2) / (w.sum(dim=2) + eps)  # -> (B,T,H)

        # Pack sequences to ignore padded timesteps
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.temporal(packed)
        return h[-1]  # (B,H)


class BidirectionalSkeletonGCN(nn.Module):
    def __init__(self, num_classes, hidden=128, gcn_layers=3, p=0.2, streams=None):
        super().__init__()
        if streams is None:
            streams = ['keypoint_coords','edge_distance','bone_vectors',
                       'keypoint_velocity','bone_velocity','keypoint_accel','bone_accel']
        self.streams = streams
        self.forward_streams = nn.ModuleDict({s: SkeletonGCNStreamOptimized(3, hidden, gcn_layers, p) for s in streams})
        self.backward_streams = nn.ModuleDict({s: SkeletonGCNStreamOptimized(3, hidden, gcn_layers, p) for s in streams})
        self.w_fwd = nn.Parameter(torch.ones(len(streams))/len(streams))
        self.w_bwd = nn.Parameter(torch.ones(len(streams))/len(streams))
        self.w_dir = nn.Parameter(torch.tensor([0.5,0.5]))
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden//2, num_classes)
        )
        self.register_buffer('A', self._build_adj())

    def _build_adj(self):
        V = 27
        A = torch.zeros(V,V)
        edges = [
            (0,1),(0,2),(1,3),(2,4),(3,5),(4,6),(5,7),(6,8),
            (7,9),(9,10),(7,11),(11,12),(7,13),(13,14),(7,15),(15,16),(7,17),(17,18),
            (8,19),(19,20),(8,21),(21,22),(8,23),(23,24),(8,25),(25,26)
        ]
        for i,j in edges:
            A[i,j]=1; A[j,i]=1
        A.fill_diagonal_(1)
        d = A.sum(dim=1)
        dinv = torch.pow(d, -0.5)
        dinv[torch.isinf(dinv)] = 0
        return dinv.unsqueeze(1)*A*dinv.unsqueeze(0)

    # ==== PATCH: forward takes node_mask and lengths; flips mask for backward ====
    def forward(self, fwd, bwd, node_mask, lengths):
        outs_f = []
        for s in self.streams:
            outs_f.append(self.forward_streams[s](fwd[s], self.A, node_mask(outs_f, dim=1))  # (B,S,H)
        wf = F.softmax(self.w_fwd, dim=0)
        Ffused = (Fstk * wf.view(1,-1,1)).sum(dim=1)  # (B,H)

        # backward uses reversed mask along time
        node_mask_b = node_mask.flip(dims=[1])
        outs_b = []
        for s in self.streams:
            outs_b.append(self.backward_streamss)
        Bstk = torch.stack(outs_b, dim=1)
        wb = F.softmax(self.w_bwd, dim=0)
        Bfused = (Bstk * wb.view(1,-1,1)).sum(dim=1)

        wd = F.softmax(self.w_dir, dim=0)
        fused = wd[0]*Ffused + wd[1]*Bfused
        return self.classifier(fused)


# ===========================
# Trainer
# ===========================
class Trainer:
    def __init__(self, model, device='cuda', lr=1e-3, wd=5e-4, epochs=100):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        self.crit = nn.CrossEntropyLoss()
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=epochs, eta_min=1e-5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        self.hist = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}

    def step(self, batch):
        fwd = {k: v.to(self.device) for k,v in batch['features_forward'].items()}
        bwd = {k: v.to(self.device) for k,v in batch['features_backward'].items()}
        lengths = batch['lengths'].to(self.device)
        node_mask = batch['node_mask'].to(self.device)
        y = batch['labels'].to(self.device)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = self.model(fwd, bwd, node_mask, lengths)
            loss = self.crit(logits, y)
        self.optim.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optim)
        self.scaler.update()
        return loss.item(), (logits.argmax(1)==y).sum().item(), y.size(0)

    @torch.no_grad()
    def eval_loop(self, loader):
        self.model.eval()
        tot_loss=0; correct=0; total=0
        for batch in loader:
            fwd = {k: v.to(self.device) for k,v in batch['features_forward'].items()}
            bwd = {k: v.to(self.device) for k,v in batch['features_backward'].items()}
            lengths = batch['lengths'].to(self.device)
            node_mask = batch['node_mask'].to(self.device)
            y = batch['labels'].to(self.device)
            logits = self.model(fwd, bwd, node_mask, lengths)
            loss = self.crit(logits, y).item()
            tot_loss += loss
            correct += (logits.argmax(1)==y).sum().item()
            total += y.size(0)
        return tot_loss/len(loader), correct/total

    def train(self, train_loader, val_loader, mlflow_log=True):
        best_val = 0
        for ep in range(self.epochs):
            self.model.train()
            tot_loss=0; correct=0; total=0
            for batch in tqdm(train_loader, desc=f'Epoch {ep+1}/{self.epochs}'):
                loss, c, n = self.step(batch)
                tot_loss += loss
                correct += c
                total += n

            tr_loss = tot_loss/len(train_loader)
            tr_acc = correct/total
            va_loss, va_acc = self.eval_loop(val_loader)

            if ep == 0:
                # One-time sanity prints on first batch
                for batch in train_loader:
                    fwd = batch['features_forward']
                    first_stream = next(iter(fwd.values()))  # (B, T, V, F)
                    print(f"Batch shape: {first_stream.shape}")
                    print(f"Stream min/max: {first_stream.min():.4f} to {first_stream.max():.4f}")
                    zeros = (first_stream == 0).sum().item()
                    print(f"Zeros: {zeros} / {first_stream.numel()}")
                    print(f"Lengths (first 8): {batch['lengths'][:8].tolist()}")
                    node_mask = batch['node_mask']
                    print(f"Node-mask coverage: {node_mask.mean().item():.4f}")
                    break

            self.hist['train_loss'].append(tr_loss)
            self.hist['val_loss'].append(va_loss)
            self.hist['train_acc'].append(tr_acc)
            self.hist['val_acc'].append(va_acc)
            print(f"Train: Loss={tr_loss:.4f} Acc={tr_acc:.2f} | Val: Loss={va_loss:.4f} Acc={va_acc:.2f}")

            self.sched.step()

            if mlflow_log:
                mlflow.log_metrics({
                    'train_loss': tr_loss,
                    'val_loss': va_loss,
                    'train_accuracy': tr_acc,
                    'val_accuracy': va_acc,
                    'learning_rate': self.optim.param_groups[0]['lr']
                }, step=ep)

            if va_acc > best_val:
                best_val = va_acc
                torch.save(self.model.state_dict(), 'best_true_skeleton_gcn.pt')

        return best_val


# ===========================
# Main
# ===========================
def main():
    set_seed(42)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    RUN_NAME = 'Top 10: True-Skeleton Bidirectional Multi-Stream GCN (Patched)'

    # Config
    EPOCHS=100; BATCH=64; HIDDEN=128; GCN_LAYERS=3; DROPOUT=0.2; LR=1e-3
    PREFETCH_FACTOR = 4
    DATA_DIR='./word_landmarks_extracted'

    print('='*80)
    print(RUN_NAME)
    print('='*80)

    # MLflow
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME','roman')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD','SignNet')
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI','https://mlflow.schlaepfer.me'))
    mlflow.set_experiment('SignNetWord')
    run = mlflow.start_run(log_system_metrics=True, run_name=RUN_NAME)
    try:
        mlflow.log_params({
            'epochs':EPOCHS,'batch_size':BATCH,'hidden':HIDDEN,'gcn_layers':GCN_LAYERS,
            'dropout':DROPOUT,'lr':LR,'device':DEVICE
        })
        mlflow.log_params({
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'os': platform.system(), 'cpu_count': os.cpu_count(),
            'total_ram_gb': round(psutil.virtual_memory().total/(1024**3),2)
        })
        if torch.cuda.is_available():
            mlflow.log_params({
                'gpu_name': torch.cuda.get_device_name(0),
                'cuda_version': torch.version.cuda,
                'gpu_mem_gb': round(torch.cuda.get_device_properties(0).total_memory/(1024**3),2)
            })

        # Build dataset
        fe = Skeleton27FeatureExtractor(conf_valid_thresh=0.5)
        full_base = TrueSkeletonDataset(DATA_DIR, fe, debug=True)

        # Choose top K words
        TOP_K = 2
        top_k_words, _ = build_topk_vocabulary(full_base.files, K=TOP_K, debug=True)

        # Create top-K dataset wrapper (new compact vocab)
        full = TopKTrueSkeletonDataset(full_base, top_k_words, feature_extractor=fe, debug=True)

        # Split
        tr_idx, va_idx, te_idx = split_dataset_stratified(full, 0.7, 0.15, 0.15, random_state=42)

        # Small per-class counts (train only)
        train_words = []
        for i in tr_idx:
            d = np.load(full.files[i], allow_pickle=True)
            train_words.append(d['glosses'][0])
        train_counts = Counter(train_words)
        print("\n[TRAIN CLASS COUNTS] (top 10)")
        for w, c in train_counts.most_common(10):
            print(f"  {w}: {c}")

        train_ds = SubsetDataset(full, tr_idx)
        val_ds   = SubsetDataset(full, va_idx)
        test_ds  = SubsetDataset(full, te_idx)

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
        model = BidirectionalSkeletonGCN(
            num_classes=num_classes, hidden=HIDDEN,
            gcn_layers=GCN_LAYERS, p=DROPOUT, streams=['keypoint_coords']
        )
        mlflow.log_param('top_k', TOP_K)
        with open('topk_words.txt', 'w') as f:
            for w in sorted(full.word_to_idx.keys()):
                f.write(f"{w}\n")
        mlflow.log_artifact('topk_words.txt')

        # Train
        trainer = Trainer(model, device=DEVICE, lr=LR, wd=5e-4, epochs=EPOCHS)
        best_val = trainer.train(train_loader, val_loader)
        print(f"Best Val Acc: {best_val:.2f}")

        # Test
        test_loss, test_acc = trainer.eval_loop(test_loader)
        print(f"Test: Loss={test_loss:.4f} Acc={test_acc:.2f}")
        mlflow.log_metrics({'test_loss': test_loss, 'test_acc': test_acc})

        # Save split
        with open('dataset_split_true_skeleton.json','w') as f:
            json.dump({'train':tr_idx,'val':va_idx,'test':te_idx,'word_to_idx':full.word_to_idx}, f, indent=2)
        mlflow.log_artifact('dataset_split_true_skeleton.json')

    finally:
        mlflow.end_run()


if __name__ == '__main__':
    main()
