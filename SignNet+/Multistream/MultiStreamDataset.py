"""
Multi-Stream Dataset for Sign Language Recognition

Computes and provides all 4 streams:
- Joint (original landmarks)
- Bone (vectors between connected landmarks)
- Joint Motion (temporal difference)
- Bone Motion (temporal difference of bones)

Author: Andrei Chirila, Roman Schläpfer
Date: 2025-12-01
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# ============================================================================
# 📋 BONE DEFINITIONS
# ============================================================================

HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

POSE_BONES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
]


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    max_seq_length: int = 214
    normalize: bool = True
    augment: bool = True
    rotation_range: float = 15.0
    scale_range: Tuple[float, float] = (0.9, 1.1)
    temporal_dropout_prob: float = 0.1


class MultiStreamDataset(Dataset):
    """Multi-stream dataset for sign language recognition."""

    def __init__(
            self,
            data_list: List[Dict],
            vocab: Any,
            config: DatasetConfig = None,
            is_train: bool = True
    ):
        self.data_list = data_list
        self.vocab = vocab
        self.config = config or DatasetConfig()
        self.is_train = is_train

        self._build_bones()
        print(f"📁 MultiStreamDataset: {len(data_list)} samples, train={is_train}")

    def _build_bones(self):
        """Build bone connections array."""
        bones = []

        # Left hand (0-20)
        for p, c in HAND_BONES:
            bones.append((p, c))

        # Right hand (21-41)
        for p, c in HAND_BONES:
            bones.append((21 + p, 21 + c))

        # Pose (42-74)
        for p, c in POSE_BONES:
            bones.append((42 + p, 42 + c))

        # Hand to wrist
        bones.append((0, 42 + 15))
        bones.append((21, 42 + 16))

        self.bone_connections = np.array(bones)
        self.num_bones = len(bones)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data_list[idx]

        # Load landmarks
        data = np.load(item['features'])
        landmarks = data['landmarks'].astype(np.float32)

        # Ensure shape (T, 543, 2)
        if landmarks.ndim == 2:
            landmarks = landmarks.reshape(landmarks.shape[0], -1, 2)

        # Augment
        if self.is_train and self.config.augment:
            landmarks = self._augment(landmarks)

        # Normalize to [-1, 1]
        if self.config.normalize:
            landmarks = (landmarks - 0.5) * 2

        # Compute streams
        joint = landmarks
        bone = self._compute_bones(landmarks)
        joint_motion = self._compute_motion(landmarks)
        bone_motion = self._compute_motion(bone)

        # Labels
        glosses = item.get('glosses', item.get('annotation', []))
        if isinstance(glosses, str):
            glosses = glosses.split()
        labels = self._encode_labels(glosses)

        return {
            'joint': torch.FloatTensor(joint),
            'bone': torch.FloatTensor(bone),
            'joint_motion': torch.FloatTensor(joint_motion),
            'bone_motion': torch.FloatTensor(bone_motion),
            'labels': torch.LongTensor(labels),
            'length': len(landmarks),
            'label_length': len(labels)
        }

    def _compute_bones(self, landmarks: np.ndarray) -> np.ndarray:
        """Compute normalized bone vectors."""
        T = landmarks.shape[0]
        bones = np.zeros((T, self.num_bones, 2), dtype=np.float32)

        for i, (p, c) in enumerate(self.bone_connections):
            if p < landmarks.shape[1] and c < landmarks.shape[1]:
                bones[:, i, :] = landmarks[:, c, :] - landmarks[:, p, :]

        # Normalize
        lengths = np.linalg.norm(bones, axis=-1, keepdims=True)
        lengths = np.maximum(lengths, 1e-8)
        bones = bones / lengths

        return bones

    def _compute_motion(self, features: np.ndarray) -> np.ndarray:
        """Compute temporal difference."""
        motion = np.zeros_like(features)
        if len(features) > 1:
            motion[1:] = features[1:] - features[:-1]
        return motion

    def _augment(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply augmentations."""
        cfg = self.config

        # Rotation
        if cfg.rotation_range > 0:
            angle = np.random.uniform(-cfg.rotation_range, cfg.rotation_range)
            rad = np.radians(angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

            center = np.array([0.5, 0.5])
            landmarks = (landmarks - center) @ rot.T + center

        # Scale
        scale = np.random.uniform(*cfg.scale_range)
        center = np.array([0.5, 0.5])
        landmarks = (landmarks - center) * scale + center

        # Temporal dropout
        if cfg.temporal_dropout_prob > 0 and len(landmarks) > 20:
            keep = np.random.random(len(landmarks)) > cfg.temporal_dropout_prob
            if keep.sum() >= 20:
                landmarks = landmarks[keep]

        return landmarks.astype(np.float32)

    def _encode_labels(self, glosses: List[str]) -> List[int]:
        """Encode glosses to indices."""
        if hasattr(self.vocab, 'gloss2idx'):
            unk_idx = self.vocab.gloss2idx.get('<UNK>', 2)
            return [self.vocab.gloss2idx.get(g, unk_idx) for g in glosses]
        else:
            return [self.vocab.get(g, 2) for g in glosses]


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch with padding."""
    max_seq = max(item['length'] for item in batch)
    max_label = max(item['label_length'] for item in batch)

    B = len(batch)
    num_bones = batch[0]['bone'].shape[1]

    joint = torch.zeros(B, max_seq, 543, 2)
    bone = torch.zeros(B, max_seq, num_bones, 2)
    joint_motion = torch.zeros(B, max_seq, 543, 2)
    bone_motion = torch.zeros(B, max_seq, num_bones, 2)
    labels = torch.zeros(B, max_label, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)
    label_lengths = torch.zeros(B, dtype=torch.long)

    for i, item in enumerate(batch):
        L = item['length']
        joint[i, :L] = item['joint']
        bone[i, :L] = item['bone']
        joint_motion[i, :L] = item['joint_motion']
        bone_motion[i, :L] = item['bone_motion']

        ll = item['label_length']
        labels[i, :ll] = item['labels']

        lengths[i] = L
        label_lengths[i] = ll

    return {
        'joint': joint,
        'bone': bone,
        'joint_motion': joint_motion,
        'bone_motion': bone_motion,
        'labels': labels,
        'lengths': lengths,
        'label_lengths': label_lengths
    }


if __name__ == "__main__":
    # Test
    import tempfile
    import os

    tmpdir = tempfile.mkdtemp()
    data_list = []

    for i in range(5):
        T = np.random.randint(50, 100)
        landmarks = np.random.rand(T, 543, 2).astype(np.float32)
        path = os.path.join(tmpdir, f"sample_{i}.npz")
        np.savez(path, landmarks=landmarks)
        data_list.append({'features': path, 'glosses': ['A', 'B', 'C']})

    vocab = {'<PAD>': 0, '<BLANK>': 1, '<UNK>': 2, 'A': 3, 'B': 4, 'C': 5}

    dataset = MultiStreamDataset(data_list, vocab, is_train=True)
    sample = dataset[0]

    print(f"joint: {sample['joint'].shape}")
    print(f"bone: {sample['bone'].shape}")
    print(f"joint_motion: {sample['joint_motion'].shape}")
    print(f"bone_motion: {sample['bone_motion'].shape}")

    import shutil

    shutil.rmtree(tmpdir)