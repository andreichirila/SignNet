import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import seaborn as sns
import mlflow
import mlflow.pytorch
import json
import platform
import psutil
import matplotlib.pyplot as plt
from datetime import datetime
from telegram import Bot
import asyncio
import signal


# ==================== FEATURE ENGINEERING MODULE (NEW) ====================
class EnhancedLandmarkFeatures:
    """
    Extract velocity, acceleration, and spatial features from MediaPipe landmarks
    State-of-the-art feature engineering for sign language recognition
    """

    @staticmethod
    def reshape_landmarks(landmarks_flat):
        """Reshape flat landmarks (T, 1659) to (T, num_landmarks, 3)"""
        T = landmarks_flat.shape[0]
        if landmarks_flat.shape[1] % 3 != 0:
            pad_size = 3 - (landmarks_flat.shape[1] % 3)
            landmarks_flat = np.pad(landmarks_flat, ((0, 0), (0, pad_size)), mode='constant')

        num_landmarks = landmarks_flat.shape[1] // 3
        landmarks_3d = landmarks_flat.reshape(T, num_landmarks, 3)
        return landmarks_3d

    @staticmethod
    def compute_velocity(landmarks, fps=25):
        """Compute first-order velocity"""
        velocity = np.zeros_like(landmarks)
        if landmarks.shape[0] > 1:
            velocity[1:] = (landmarks[1:] - landmarks[:-1]) * fps
            velocity[0] = velocity[1]
        return velocity

    @staticmethod
    def compute_acceleration(velocity, fps=25):
        """Compute second-order acceleration"""
        acceleration = np.zeros_like(velocity)
        if velocity.shape[0] > 1:
            acceleration[1:] = (velocity[1:] - velocity[:-1]) * fps
            acceleration[0] = acceleration[1]
        return acceleration

    @staticmethod
    def compute_hand_distances(landmarks_3d):
        """Compute inter-hand distance aligned with _default_edges mapping."""
        T, L, _ = landmarks_3d.shape
        EXPECTED_L = 553
        if L != EXPECTED_L:
            return np.zeros((T, 1))  # fallback-safe
        LH_START, LH_END = 0, 21
        RH_START, RH_END = 21, 42
        left_center = landmarks_3d[:, LH_START:LH_END, :].mean(axis=1)
        right_center = landmarks_3d[:, RH_START:RH_END, :].mean(axis=1)
        distances = np.linalg.norm(left_center - right_center, axis=1, keepdims=True)
        return distances

    @staticmethod
    def compute_hand_to_face_distances(landmarks_3d):
        """Distance from each hand center to face center aligned with _default_edges mapping."""
        T, L, _ = landmarks_3d.shape
        EXPECTED_L = 553
        if L != EXPECTED_L:
            return np.zeros((T, 1)), np.zeros((T, 1))  # fallback-safe
        FACE_START, FACE_END = 42, 520
        LH_START, LH_END = 0, 21
        RH_START, RH_END = 21, 42
        face_center = landmarks_3d[:, FACE_START:FACE_END, :].mean(axis=1)
        left_center = landmarks_3d[:, LH_START:LH_END, :].mean(axis=1)
        right_center = landmarks_3d[:, RH_START:RH_END, :].mean(axis=1)
        left_to_face = np.linalg.norm(left_center - face_center, axis=1, keepdims=True)
        right_to_face = np.linalg.norm(right_center - face_center, axis=1, keepdims=True)
        return left_to_face, right_to_face

    @staticmethod
    def compute_velocity_magnitude(velocity):
        """Compute speed"""
        speed = np.linalg.norm(velocity, axis=2)
        return speed

    @staticmethod
    def _default_edges(num_landmarks: int):
        """
        Default bone edges for your NPZ layout:
        - Left hand: 0..20  (MediaPipe Hands indices)
        - Right hand: 21..41 (MediaPipe Hands indices)
        - Face: 42..519 (MediaPipe Face Mesh)  -> no edges by default (too dense)
        - Pose: 520..552 (MediaPipe Pose, 33 pts) -> compact upper-body graph
        Falls back to consecutive edges if layout is unexpected.
        """
        # Expected layout: (2*21) + 478 + 33 = 553 landmarks
        EXPECTED_L = 553
        if num_landmarks != EXPECTED_L:
            # Fallback: connect consecutive landmarks to stay safe
            print(f"\n[WARNING ] num_landmarks != EXPECTED_L {num_landmarks}")
            return [(i, i + 1) for i in range(max(0, num_landmarks - 1))]

        LH_START = 0
        RH_START = 21
        FACE_START, FACE_END = 42, 520   # 42..519 inclusive (478 points)
        POSE_START = 520                 # 520..552 inclusive (33 points)

        edges = []

        # MediaPipe Hands chains (relative indices inside a single hand block of 21 points)
        # 0: wrist
        hand_chains = [
            [0, 1, 2, 3, 4],        # thumb: 0-1-2-3-4
            [0, 5, 6, 7, 8],        # index: 0-5-6-7-8
            [0, 9, 10, 11, 12],     # middle: 0-9-10-11-12
            [0, 13, 14, 15, 16],    # ring: 0-13-14-15-16
            [0, 17, 18, 19, 20],    # pinky: 0-17-18-19-20
        ]

        def add_hand_edges(base):
            for chain in hand_chains:
                for a, b in zip(chain[:-1], chain[1:]):
                    edges.append((base + a, base + b))

        # Left and Right hands
        add_hand_edges(LH_START)
        add_hand_edges(RH_START)

        # Compact upper-body pose graph (MediaPipe Pose indices, relative to POSE_START):
        # Key indices: 11 L_shoulder, 12 R_shoulder, 13 L_elbow, 14 R_elbow,
        #              15 L_wrist,   16 R_wrist,   23 L_hip,   24 R_hip, 0 nose.
        def P(i):
            return POSE_START + i

        pose_edges = [
            (P(11), P(13)), (P(13), P(15)),     # left shoulder-elbow-wrist
            (P(12), P(14)), (P(14), P(16)),     # right shoulder-elbow-wrist
            (P(11), P(12)),                     # clavicle
            (P(11), P(23)), (P(12), P(24)),     # shoulders to hips
            (P(23), P(24)),                     # hip line
            (P(0),  P(11)), (P(0),  P(12)),     # nose to shoulders (helps head-arm relation)
        ]
        edges.extend(pose_edges)

        # Face mesh omitted by default (too dense); add sparse anchors only if needed.
        # Example (optional): nose to a couple of face anchors if you map them.
        # edges.append((P(0), FACE_START + some_face_idx))

        return edges

    @staticmethod
    def compute_bones(landmarks_3d):
        """
        Compute bone vectors for each frame given edges: v = joint_child - joint_parent.
        Returns normalized bones: unit direction + z-scored across time.
        """
        T, num_landmarks, _ = landmarks_3d.shape
        edges = EnhancedLandmarkFeatures._default_edges(num_landmarks)

        if not edges:
            # Fallback if no edges defined
            return np.zeros((T, 0, 3))

        parent = np.array([p for p, c in edges], dtype=np.int64)
        child = np.array([c for p, c in edges], dtype=np.int64)

        # Raw bone vectors: child - parent (T, E, 3)
        bones = landmarks_3d[:, child, :] - landmarks_3d[:, parent, :]

        # FIX 1: Unit normalize each bone vector (direction only, ||v||=1)
        lengths = np.linalg.norm(bones, axis=2, keepdims=True)  # (T, E, 1)
        lengths = np.maximum(lengths, 1e-6)  # Avoid div-by-zero for zero-length bones
        unit_bones = bones / lengths  # Now scale-invariant directions

        # FIX 1: Z-score across time per bone (temporal normalization, mean=0/std=1)
        # This stabilizes varying sign lengths/poses, matching joint/velocity scale
        bone_mean = unit_bones.mean(axis=0, keepdims=True)  # (1, E, 3)
        bone_std = unit_bones.std(axis=0, keepdims=True) + 1e-6  # Avoid div-by-zero
        normalized_bones = (unit_bones - bone_mean) / bone_std

        return normalized_bones  # (T, num_edges, 3) - ready for flattening

    @staticmethod
    def extract_all_features(landmarks_flat, fps=25, is_train=False, include_accel=True,
                            include_bones=True, include_bone_velocity=False):
        """Extract all engineered features with optional bone scaling for training."""
        landmarks_3d = EnhancedLandmarkFeatures.reshape_landmarks(landmarks_flat)
        T, num_landmarks, _ = landmarks_3d.shape

        original = landmarks_flat
        velocity = EnhancedLandmarkFeatures.compute_velocity(landmarks_3d, fps)
        velocity_flat = velocity.reshape(T, -1)

        if include_accel:
            acceleration = EnhancedLandmarkFeatures.compute_acceleration(velocity, fps)
            accel_flat = acceleration.reshape(T, -1)

        hand_dist = EnhancedLandmarkFeatures.compute_hand_distances(landmarks_3d)
        left_to_face, right_to_face = EnhancedLandmarkFeatures.compute_hand_to_face_distances(landmarks_3d)
        velocity_magnitude = EnhancedLandmarkFeatures.compute_velocity_magnitude(velocity)

        features_list = [
            original,
            velocity_flat,
            hand_dist,
            left_to_face,
            right_to_face,
            velocity_magnitude
        ]

        if include_accel:
            accel_lengths = np.linalg.norm(acceleration, axis=2, keepdims=True)
            accel_lengths = np.maximum(accel_lengths, 1e-6)
            unit_accel = acceleration / accel_lengths
            accel_mean = unit_accel.mean(axis=0, keepdims=True)
            accel_std = unit_accel.std(axis=0, keepdims=True) + 1e-6
            normalized_accel = (unit_accel - accel_mean) / accel_std
            features_list.insert(2, normalized_accel.reshape(T, -1))

        if include_bones:
            bones = EnhancedLandmarkFeatures.compute_bones(landmarks_3d)

            # Add bone scaling for training data
            if is_train and np.random.random() < 0.3:
                # Randomly scale bone lengths to simulate signer variation
                bone_lengths = np.linalg.norm(bones, axis=2, keepdims=True) + 1e-6
                bone_dirs = bones / bone_lengths
                scale_factor = np.random.uniform(0.9, 1.1, size=(1, bones.shape[1], 1))
                bones = bone_dirs * bone_lengths * scale_factor

            bones_flat = bones.reshape(T, -1)
            features_list.append(bones_flat)

            if include_bone_velocity:
                bone_vel = EnhancedLandmarkFeatures.compute_velocity(bones, fps)
                # Apply same scaling to bone velocities if bones were scaled
                if is_train and np.random.random() < 0.3:
                    vel_lengths = np.linalg.norm(bone_vel, axis=2, keepdims=True) + 1e-6
                    vel_dirs = bone_vel / vel_lengths
                    vel_scale = np.random.uniform(0.9, 1.1, size=(1, bone_vel.shape[1], 1))
                    bone_vel = vel_dirs * vel_lengths * vel_scale

                vel_lengths = np.linalg.norm(bone_vel, axis=2, keepdims=True)
                vel_lengths = np.maximum(vel_lengths, 1e-6)
                unit_vel = bone_vel / vel_lengths
                vel_mean = unit_vel.mean(axis=0, keepdims=True)
                vel_std = unit_vel.std(axis=0, keepdims=True) + 1e-6
                normalized_vel = (unit_vel - vel_mean) / vel_std
                features_list.append(normalized_vel.reshape(T, -1))

        enhanced_features = np.concatenate(features_list, axis=1)
        return enhanced_features.astype(np.float32)


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Monitors validation loss/accuracy and stops training if no improvement.
    """
    def __init__(self, patience=10, min_delta=0.001, metric="val_loss", mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric  # e.g., "val_loss" or "val_accuracy"
        self.mode = mode      # "min" for loss, "max" for accuracy
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False

        # Initialize best score
        if mode == "min":
            self.best_score = float('inf')
        else:
            self.best_score = -float('inf')

    def __call__(self, current_score, epoch):
        """
        Check if training should stop.
        Args:
            current_score: Current validation metric value
            epoch: Current epoch number
        Returns:
            True if early stopping triggered, False otherwise
        """
        improved = False

        # Check if score improved
        if self.mode == "min":
            if current_score < (self.best_score - self.min_delta):
                improved = True
                self.best_score = current_score
                self.best_epoch = epoch
                self.counter = 0
        else:  # mode == "max"
            if current_score > (self.best_score + self.min_delta):
                improved = True
                self.best_score = current_score
                self.best_epoch = epoch
                self.counter = 0

        # Print improvement message
        if improved:
            print(f"  ✓ {self.metric}: {self.best_score:.4f} (epoch {self.best_epoch + 1})")
        else:
            self.counter += 1
            print(f"  ℹ No improvement for {self.counter}/{self.patience} epochs")

            # Check if patience exceeded
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n[EARLY STOPPING] No improvement for {self.patience} epochs")
                print(f"  Best {self.metric}: {self.best_score:.4f} (epoch {self.best_epoch + 1})")
                return True

        return False


class TemporalAugmentation:
    """Augmentation for variable-length landmark sequences with class-aware probabilities."""

    def __init__(self, class_counts=None, base_prob=0.5, strength=0.5, prob=0.7):
        self.prob = prob
        self.base_prob = base_prob
        self.strength = strength
        self.class_prob_map = None
        if class_counts:
            self.class_prob_map = self._create_prob_map(class_counts, strength)

    def _create_prob_map(self, counts, strength):
        """Create class-specific augmentation probabilities based on inverse frequency."""
        # Normalize counts -> frequencies
        total_counts = sum(counts.values())
        freqs = {cls: c / total_counts for cls, c in counts.items()}

        # Inverse frequency mapping (less freq = higher prob)
        # Low freq -> higher aug prob. Strength controls how much.
        prob_map = {
            cls: self.base_prob + (1 - f) * self.strength * self.base_prob
            for cls, f in freqs.items()
        }
        return prob_map

    def time_warp(self, seq, warp_factor_min=0.8, warp_factor_max=1.25):
        if len(seq) <= 2:
            return seq
        factor = np.random.uniform(warp_factor_min, warp_factor_max)
        new_length = max(2, int(len(seq) / factor))
        indices = np.linspace(0, len(seq) - 1, new_length)
        return seq[indices.astype(int)]

    def temporal_dropout(self, seq, keep_prob_min=0.85, keep_prob_max=0.98):
        keep_prob = np.random.uniform(keep_prob_min, keep_prob_max)
        mask = np.random.rand(len(seq)) < keep_prob
        if mask.sum() <= 1:
            return seq
        return seq[mask]

    def add_noise(self, seq, sigma=0.008):
        noise = np.random.normal(0, sigma, seq.shape)
        return seq + noise

    def scaling(self, seq, scale_min=0.9, scale_max=1.1):
        scale = np.random.uniform(scale_min, scale_max)
        return seq * scale

    def structured_channel_dropout(self, seq, drop_prob=0.1):
        """Drop entire feature blocks to force multi-stream robustness."""
        seq = seq.copy()
        # Define feature block indices for your 5833-dim input
        # These are approximate - adjust based on your exact feature layout
        # Format: (start, end) for major blocks
        feature_blocks = [
            (0, 1659),      # Raw landmarks
            (1659, 3318),    # Velocities
            (3318, 4977),    # Bones (if included)
            (4977, 5833),    # Other derived features (hand distances, etc.)
        ]

        # Randomly drop one major block with probability
        if np.random.random() < drop_prob:
            block_idx = np.random.randint(0, len(feature_blocks))
            start, end = feature_blocks[block_idx]
            seq[:, start:end] = 0.0
        return seq

    def __call__(self, landmarks, class_label=None):
        # Use class-aware probability if available
        aug_prob = self.base_prob
        if self.class_prob_map and class_label is not None:
            aug_prob = self.class_prob_map.get(class_label, self.base_prob)

        if np.random.random() > aug_prob:
            return landmarks

        augmented = landmarks.astype(np.float32).copy()

        # 1) Time warp OR temporal dropout (at most one)
        if np.random.random() > 0.4:
            if np.random.random() < 0.5:
                augmented = self.time_warp(augmented, 0.80, 1.20)
            else:
                augmented = self.temporal_dropout(augmented, 0.90, 0.97)

        # 2) Small scaling
        if np.random.random() > 0.6:
            augmented = self.scaling(augmented, 0.92, 1.08)

        # 3) Noise
        if np.random.random() > 0.4:
            augmented = self.add_noise(augmented, sigma=0.008)

        # 4) Structured channel dropout
        if np.random.random() > 0.75:
            augmented = self.structured_channel_dropout(augmented, drop_prob=0.1)

        return augmented




class SignLanguageDataset(Dataset):
    """
    Load preprocessed landmarks from NPZ files with per-frame handedness.
    NOW SUPPORTS ENHANCED FEATURES (NEW)
    """
    def __init__(self, npz_dir, word_to_idx=None, debug=True, augment=False, augment_prob=0.7,
                 use_enhanced_features=False, include_accel=False,
                 include_bones=True, include_bone_velocity=False, class_counts=None):
        self.npz_dir = Path(npz_dir)
        self.npz_files = sorted(self.npz_dir.glob("*.npz"))
        self.debug = debug

        self.augment = augment
        self.use_enhanced_features = use_enhanced_features
        self.include_accel = include_accel
        self.include_bones = include_bones
        self.include_bone_velocity = include_bone_velocity
        self.class_counts = class_counts  # NEW: Store for class-aware augmentation

        if augment:
            self.augmentation = TemporalAugmentation(
                class_counts=class_counts,  # NEW: Pass class counts
                base_prob=0.5,
                strength=0.5,
                prob=augment_prob
            )

        if debug:
            print(f"\n[DEBUG] SignLanguageDataset.__init__")
            print(f"  NPZ directory: {self.npz_dir}")
            print(f"  Total NPZ files found: {len(self.npz_files)}")
            print(f"  Enhanced features: {use_enhanced_features}")  # NEW
            print(f"  Include acceleration: {include_accel}")  # NEW

        if word_to_idx is None:
            self.word_to_idx = {}
            for npz_file in self.npz_files:
                try:
                    data = np.load(npz_file, allow_pickle=True)
                    gloss = data["glosses"][0]
                    if gloss not in self.word_to_idx:
                        self.word_to_idx[gloss] = len(self.word_to_idx)
                except Exception as e:
                    print(f"  [WARNING] Error loading {npz_file}: {e}")
        else:
            self.word_to_idx = word_to_idx

        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}

        if debug:
            print(f"  Total unique words: {len(self.word_to_idx)}")
            print(f"  Word vocabulary: {list(self.word_to_idx.keys())[:10]}...")

    def __len__(self):
        return len(self.npz_files)

    def _get_dominant_handedness(self, handedness_data):
        """Aggregate per-frame handedness to sample-level"""
        left_count = 0
        right_count = 0

        for frame_hands in handedness_data:
            if isinstance(frame_hands, str):
                if frame_hands == "LEFT":
                    left_count += 1
                elif frame_hands == "RIGHT":
                    right_count += 1
            else:
                for hand in frame_hands:
                    if hand == "LEFT":
                        left_count += 1
                    elif hand == "RIGHT":
                        right_count += 1

        if left_count > 0 and right_count == 0:
            return 0  # LEFT only
        elif right_count > 0 and left_count == 0:
            return 1  # RIGHT only
        elif left_count > 0 and right_count > 0:
            return 2  # BOTH hands used
        else:
            return 3  # No hand detected (NONE)

    def __getitem__(self, idx):
        npz_file = self.npz_files[idx]
        data = np.load(npz_file, allow_pickle=True)

        # Load landmarks
        landmarks = data["landmarks"].astype(np.float32)

        # Load gloss and convert to string CORRECTLY
        glosses = data["glosses"]
        if len(glosses) > 0:
            gloss_item = glosses[0]  # ← FIX: Extract first element
            if isinstance(gloss_item, (np.ndarray, np.str_, bytes)):
                gloss = str(gloss_item)
            else:
                gloss = gloss_item
        else:
            gloss = "UNKNOWN"

        label = self.word_to_idx.get(gloss, 0)

        # APPLY FEATURE ENGINEERING IF ENABLED (with is_train flag)
        if self.use_enhanced_features:
            landmarks = EnhancedLandmarkFeatures.extract_all_features(
                landmarks,
                fps=25,
                is_train=self.augment,  # ← CHANGED: Use self.augment instead of self.split
                include_accel=self.include_accel,
                include_bones=self.include_bones,
                include_bone_velocity=self.include_bone_velocity
            )

        # Apply augmentation AFTER feature engineering (class-aware)
        if self.augment:
            landmarks = self.augmentation(landmarks, class_label=gloss)

        # Load and aggregate handedness
        if "handedness" in data:
            handedness_data = data["handedness"]
            handedness = self._get_dominant_handedness(handedness_data)
        else:
            handedness = 3

        # Convert to tensors
        landmarks_tensor = torch.from_numpy(landmarks).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        handedness_tensor = torch.tensor(handedness, dtype=torch.long)

        return landmarks_tensor, label_tensor, handedness_tensor


class RemappedDataset(Dataset):
    """Remaps old class labels to new class labels for filtered dataset."""

    def __init__(self, base_dataset, indices, old_to_new_idx):
        self.base_dataset = base_dataset
        self.indices = indices
        self.old_to_new_idx = old_to_new_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        landmarks, old_label, handedness = self.base_dataset[base_idx]

        old_label_val = old_label.item()

        if old_label_val not in self.old_to_new_idx:
            raise ValueError(f"Label {old_label_val} not in remapping dict!")

        new_label = self.old_to_new_idx[old_label_val]

        return landmarks, torch.tensor(new_label, dtype=torch.long), handedness

class OversampledDataset(RemappedDataset):
    """Extends RemappedDataset to oversample specific classes"""

    def __init__(self, base_dataset, indices, old_to_new_idx, oversample_config=None):
        super().__init__(base_dataset, indices, old_to_new_idx)

        if oversample_config:
            self.oversampled_indices = self._create_oversampled_indices(oversample_config)
        else:
            self.oversampled_indices = list(range(len(self.indices)))

    def _create_oversampled_indices(self, config):
        """Create list of indices with oversampling applied"""
        oversampled = []

        for i, real_idx in enumerate(self.indices):
            # Get original sample to check class
            _, label, _ = self.base_dataset[real_idx]
            class_name = self.base_dataset.idx_to_word[label.item()]

            # Add original index
            oversampled.append(i)

            # Add duplicates if class is in config
            if class_name in config:
                repeat_count = config[class_name] - 1  # -1 because we already added original
                for _ in range(repeat_count):
                    oversampled.append(i)

        return oversampled

    def __len__(self):
        return len(self.oversampled_indices)

    def __getitem__(self, idx):
        # Map oversampled index to original RemappedDataset index
        original_idx = self.oversampled_indices[idx]
        return super().__getitem__(original_idx)


class TransformerSignClassifierWithHandedness(nn.Module):
    """
    Transformer encoder model with:
    - Multi-task learning (sign + handedness)
    - Uses existing per-frame features (joints + bones + velocities + etc.)
    Drop-in replacement for LSTMSignClassifierWithHandedness.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        dropout_rate: float = 0.3,
        attention_dropout: float = 0.1,
        debug: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.debug = debug

        # Project input features to model dimension (hidden_size)
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Positional encoding (learned)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 2048, hidden_size))  # max_len=2048
        nn.init.normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=attention_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.dropout = nn.Dropout(dropout_rate)

        # Task 1: Sign classification head
        self.fc_sign = nn.Linear(hidden_size, num_classes)

        # Task 2: Handedness classification head (4 classes: LEFT, RIGHT, BOTH, NONE)
        self.fc_handedness = nn.Linear(hidden_size, 4)

        if debug:
            print(f"[DEBUG] TransformerSignClassifierWithHandedness initialized")
            print(f"  Input size: {input_size}")
            print(f"  Hidden size: {hidden_size}")
            print(f"  Num classes (sign): {num_classes}")
            print(f"  Num heads: {num_heads}")
            print(f"  Num layers: {num_layers}")
            print(f"  FFN dim: {dim_feedforward}")
            print(f"  Dropout: {dropout_rate}, attn_dropout: {attention_dropout}")
            print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, landmarks, src_key_padding_mask=None):
        """
        Args:
            landmarks: (batch_size, seq_len, input_size)
            src_key_padding_mask: optional (batch_size, seq_len) True=pad
        Returns:
            sign_logits: (batch_size, num_classes)
            handedness_logits: (batch_size, 4)
        """
        B, T, D = landmarks.shape

        # Project to hidden_size
        x = self.input_proj(landmarks)  # (B, T, hidden)

        # Add positional embeddings (crop to current length)
        if T > self.pos_embedding.size(1):
            raise ValueError(f"Sequence length {T} exceeds max positional length {self.pos_embedding.size(1)}")
        pos_emb = self.pos_embedding[:, :T, :]  # (1, T, hidden)
        x = x + pos_emb

        # Transformer encoder
        # src_key_padding_mask: (B, T) with True at PAD positions
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, hidden)

        # Global average pooling over time (mask-aware)
        if src_key_padding_mask is not None:
            # mask: True for PAD -> set to 0 weight
            mask = (~src_key_padding_mask).float().unsqueeze(-1)  # (B, T, 1)
            x_masked = x * mask
            lengths = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
            pooled = x_masked.sum(dim=1) / lengths    # (B, hidden)
        else:
            pooled = x.mean(dim=1)  # (B, hidden)

        pooled = self.dropout(pooled)

        sign_logits = self.fc_sign(pooled)
        handedness_logits = self.fc_handedness(pooled)

        return sign_logits, handedness_logits



class PadCollate:
    def __call__(self, batch):
        landmarks_list = [item[0] for item in batch]
        labels_list = [item[1] for item in batch]
        handedness_list = [item[2] for item in batch]

        lengths = [lm.shape[0] for lm in landmarks_list]
        max_seq_len = max(lengths)
        feature_dim = landmarks_list[0].shape[1]

        padded_landmarks = []
        for lm in landmarks_list:
            pad_size = max_seq_len - lm.shape[0]
            if pad_size > 0:
                lm_padded = F.pad(lm, (0, 0, 0, pad_size), value=0.0)
            else:
                lm_padded = lm
            padded_landmarks.append(lm_padded)

        landmarks_tensor = torch.stack(padded_landmarks).float()  # (B, T, D)
        labels = torch.tensor(labels_list, dtype=torch.long)
        handedness = torch.tensor(handedness_list, dtype=torch.long)

        # src_key_padding_mask: True at PAD positions
        # shape (B, T)
        padding_mask = torch.zeros(len(batch), max_seq_len, dtype=torch.bool)
        for i, l in enumerate(lengths):
            if l < max_seq_len:
                padding_mask[i, l:] = True

        return landmarks_tensor, labels, handedness, padding_mask



# ==================== FOCAL LOSS (NEW) ====================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax for long-tailed classification:
    p(y=c|x) ∝ exp(z_c + log n_c), where n_c = class frequency.
    """
    def __init__(self, class_counts: torch.Tensor):
        super().__init__()
        # class_counts is 1D tensor of size [C]
        priors = class_counts.float().clamp_min(1.0)
        self.log_priors = torch.log(priors / priors.sum())

    def forward(self, logits, targets):
        # Broadcast log_priors to batch; move to device
        log_priors = self.log_priors.to(logits.device)
        balanced_logits = logits + log_priors.unsqueeze(0)
        return F.cross_entropy(balanced_logits, targets)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with uncertainty-based dynamic weighting.
    Learns optimal task weights automatically during training.
    """
    def __init__(self, label_smoothing=0.0, use_focal=False, class_weights=None,
                 use_balanced_softmax=False, balanced_class_counts=None):
        super().__init__()

        # Set up sign classification loss
        if use_balanced_softmax and balanced_class_counts is not None:
            self.sign_loss = BalancedSoftmaxLoss(balanced_class_counts)
        elif use_focal:
            self.sign_loss = FocalLoss(alpha=0.25, gamma=2.0, weight=class_weights)
        else:
            self.sign_loss = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                weight=class_weights
            )

        # Handedness loss
        self.handedness_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Learnable uncertainty parameters (log-variance)
        # Initialize to 0 (variance = 1) for balanced start
        self.log_var_sign = nn.Parameter(torch.zeros(1))
        self.log_var_hand = nn.Parameter(torch.zeros(1))

    def forward(self, sign_logits, handedness_logits, sign_labels, handedness_labels):
        # Calculate individual task losses
        loss_sign = self.sign_loss(sign_logits, sign_labels)
        loss_handedness = self.handedness_loss(handedness_logits, handedness_labels)

        # Uncertainty-based weighting
        # precision = exp(-log_var) = 1 / variance
        precision_sign = torch.exp(-self.log_var_sign)
        precision_hand = torch.exp(-self.log_var_hand)

        # Weighted losses with regularization term
        weighted_sign = precision_sign * loss_sign + self.log_var_sign
        weighted_hand = precision_hand * loss_handedness + self.log_var_hand

        # Total loss
        total_loss = weighted_sign + weighted_hand

        return total_loss, loss_sign, loss_handedness



def compute_effective_number_weights(class_counts, beta=0.9999):
    """Compute class weights using effective number method."""
    num_classes = len(class_counts)
    weights = torch.zeros(num_classes)

    for cls, count in class_counts.items():
        effective_num = (1 - beta**count) / (1 - beta)
        weights[cls] = 1.0 / effective_num

    weights = weights / weights.sum() * num_classes

    return weights


def build_topk_vocabulary(npz_files, K=150, min_samples=50, debug=True):
    """Build vocabulary with minimum sample filtering."""
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
        print(f"[INFO] TopK builder skipped {skipped} unreadable files.")

    filtered_counts = {w: c for w, c in counts.items() if c >= min_samples}

    if debug:
        removed = len(counts) - len(filtered_counts)
        print(f"[INFO] Filtered {removed} classes with < {min_samples} samples")
        print(f"[INFO] Remaining vocabulary: {len(filtered_counts)} classes")

    most_common = Counter(filtered_counts).most_common(K)
    top_k_words = {w for (w, _) in most_common}

    if debug:
        print(f"[INFO] Selected top-{len(top_k_words)} classes from filtered vocabulary")
        if most_common:
            print(f"[INFO] Sample distribution: min={min(c for w, c in most_common)}, "
                  f"max={max(c for w, c in most_common)}, "
                  f"total_samples={sum(c for w, c in most_common)}")

    return top_k_words, dict(counts)


def compute_topk_accuracy(logits, labels, k_values=[1, 2, 3, 4, 5]):
    """Compute top-k accuracy for multiple k values."""
    batch_size = labels.size(0)
    num_classes = logits.size(1)

    max_k = max(k_values)
    max_k = min(max_k, num_classes)

    _, topk_pred = logits.topk(max_k, dim=1, largest=True, sorted=True)

    labels_expanded = labels.view(-1, 1).expand_as(topk_pred)

    correct = topk_pred.eq(labels_expanded)

    topk_accs = {}
    for k in k_values:
        if k > num_classes:
            k = num_classes

        correct_k = correct[:, :k].sum(dim=1).float()
        topk_acc = correct_k.sum() / batch_size
        topk_accs[k] = topk_acc.item()

    return topk_accs


TELEGRAM_BOT_TOKEN = '8327173184:AAGLA5pcLiAz-vMSVBq4tVJCHo7TPH3Zu8g'
CHAT_ID = '8541359800'

bot = Bot(token=TELEGRAM_BOT_TOKEN)

async def send_message(text, chat_id):
    async with bot:
        await bot.send_message(text=text, chat_id=chat_id)

async def run_bot(messages, chat_id):
    text = '\n'.join(messages)
    await send_message(text, chat_id)


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


def train_epoch_interruptible(model, train_loader, optimizer, criterion, device, epoch):
    """Training with multi-task learning and top-k accuracy metrics."""
    model.train()

    total_loss = 0.0
    total_sign_loss = 0.0
    total_hand_loss = 0.0

    topk_correct = {k: 0.0 for k in [1, 2, 3, 4, 5]}
    hand_acc = 0.0
    num_batches = 0
    total_samples = 0

    scaler = amp.GradScaler(enabled=(device.type == "cuda"))
    pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}] Train", leave=False)

    for batch in pbar:
        landmarks, sign_labels, handedness_labels, padding_mask = batch
        landmarks = landmarks.to(device)
        sign_labels = sign_labels.to(device)
        handedness_labels = handedness_labels.to(device)
        padding_mask = padding_mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type='cuda', enabled=(device.type == "cuda")):
            sign_logits, handedness_logits = model(landmarks, src_key_padding_mask=padding_mask)
            total_loss_batch, loss_sign, loss_hand = criterion(
                sign_logits, handedness_logits, sign_labels, handedness_labels
            )

        scaler.scale(total_loss_batch).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = sign_labels.size(0)
        total_loss += total_loss_batch.item()
        total_sign_loss += loss_sign.item()
        total_hand_loss += loss_hand.item()

        topk_batch_accs = compute_topk_accuracy(sign_logits, sign_labels, k_values=[1, 2, 3, 4, 5])
        for k, acc in topk_batch_accs.items():
            topk_correct[k] += acc * batch_size

        total_samples += batch_size

        hand_preds = torch.argmax(handedness_logits, dim=1)
        hand_batch_acc = (hand_preds == handedness_labels).float().mean().item()
        hand_acc += hand_batch_acc

        num_batches += 1
        pbar.set_postfix({
            'Loss': f'{total_loss/num_batches:.4f}',
            'Top1': f'{topk_correct[1]/total_samples:.4f}',
            'Top5': f'{topk_correct[5]/total_samples:.4f}',
            'Hand': f'{hand_acc/num_batches:.4f}'
        })

    avg_loss = total_loss / num_batches
    avg_sign_loss = total_sign_loss / num_batches
    avg_hand_loss = total_hand_loss / num_batches
    avg_hand_acc = hand_acc / num_batches
    avg_topk_accs = {k: correct / total_samples for k, correct in topk_correct.items()}
    return avg_loss, avg_topk_accs, avg_hand_acc, avg_sign_loss, avg_hand_loss



def validate_epoch(model, val_loader, criterion, device, idx_to_word):
    """Validation with top-k accuracy."""
    model.eval()

    total_loss = 0.0

    topk_correct = {k: 0.0 for k in [1, 2, 3, 4, 5]}
    hand_acc = 0.0
    num_batches = 0
    total_samples = 0

    handedness_distribution = {"LEFT": 0, "RIGHT": 0, "BOTH": 0, "NONE": 0}

    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="[Val]", leave=False)

        for batch in pbar:
            landmarks, sign_labels, handedness_labels, padding_mask = batch
            landmarks = landmarks.to(device)
            sign_labels = sign_labels.to(device)
            handedness_labels = handedness_labels.to(device)
            padding_mask = padding_mask.to(device)

            with torch.no_grad():
                sign_logits, handedness_logits = model(landmarks, src_key_padding_mask=padding_mask)

            total_loss_batch, _, _ = criterion(
                sign_logits, handedness_logits,
                sign_labels, handedness_labels
            )
            total_loss += total_loss_batch.item()

            batch_size = sign_labels.size(0)
            topk_batch_accs = compute_topk_accuracy(sign_logits, sign_labels, k_values=[1, 2, 3, 4, 5])
            for k, acc in topk_batch_accs.items():
                topk_correct[k] += acc * batch_size

            total_samples += batch_size

            sign_preds = torch.argmax(sign_logits, dim=1)

            hand_preds = torch.argmax(handedness_logits, dim=1)
            hand_batch_acc = (hand_preds == handedness_labels).float().mean().item()
            hand_acc += hand_batch_acc

            handedness_names = ["LEFT", "RIGHT", "BOTH", "NONE"]
            for hand_label in handedness_labels.cpu().numpy():
                handedness_distribution[handedness_names[hand_label]] += 1

            all_preds.extend(sign_preds.cpu().numpy())
            all_labels.extend(sign_labels.cpu().numpy())

            num_batches += 1
            pbar.set_postfix({
                'Top1': f'{topk_correct[1]/total_samples:.4f}',
                'Top5': f'{topk_correct[5]/total_samples:.4f}'
            })

    avg_loss = total_loss / num_batches
    avg_hand_acc = hand_acc / num_batches

    avg_topk_accs = {k: correct / total_samples for k, correct in topk_correct.items()}

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    num_classes = len(idx_to_word)
    confusion_mat = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    class_metrics = {}
    for class_idx in range(num_classes):
        class_name = idx_to_word[class_idx]
        class_mask = (all_labels == class_idx)

        if class_mask.sum() > 0:
            class_correct = (all_preds[class_mask] == class_idx).sum()
            class_total = class_mask.sum()
            class_acc = class_correct / class_total

            class_metrics[class_name] = {
                'accuracy': float(class_acc),
                'support': int(class_total),
                'class_idx': class_idx
            }
        else:
            class_metrics[class_name] = {
                'accuracy': 0.0,
                'support': 0,
                'class_idx': class_idx
            }

    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    class_metrics['_overall'] = {
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro)
    }

    return avg_loss, avg_topk_accs, avg_hand_acc, handedness_distribution, \
           confusion_mat, class_metrics, all_preds, all_labels


def plot_confusion_matrix(confusion_mat, idx_to_word, save_path, top_n=50):
    """Plot and save confusion matrix."""
    num_classes = len(idx_to_word)

    if num_classes > top_n:
        class_support = confusion_mat.sum(axis=1)
        top_indices = np.argsort(class_support)[-top_n:][::-1]
        confusion_mat_filtered = confusion_mat[top_indices][:, top_indices]
        class_names = [idx_to_word[i] for i in top_indices]
        title = f'Confusion Matrix (Top {top_n} Classes by Support)'
    else:
        confusion_mat_filtered = confusion_mat
        class_names = [idx_to_word[i] for i in range(num_classes)]
        title = 'Confusion Matrix (All Classes)'

    plt.figure(figsize=(max(12, top_n * 0.4), max(10, top_n * 0.35)))

    confusion_mat_norm = confusion_mat_filtered.astype('float') / confusion_mat_filtered.sum(axis=1)[:, np.newaxis]
    confusion_mat_norm = np.nan_to_num(confusion_mat_norm)

    sns.heatmap(confusion_mat_norm,
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Blues',
                fmt='.2f',
                cbar_kws={'label': 'Normalized Count'},
                square=True,
                linewidths=0.5,
                linecolor='gray')

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()

    return save_path


def log_class_metrics_to_mlflow(class_metrics, epoch):
    """Log per-class metrics to MLflow."""
    if '_overall' in class_metrics:
        overall = class_metrics['_overall']
        mlflow.log_metrics({
            'val_f1_macro': overall['f1_macro'],
            'val_f1_weighted': overall['f1_weighted'],
            'val_precision_macro': overall['precision_macro'],
            'val_recall_macro': overall['recall_macro']
        }, step=epoch)

    class_list = [(name, metrics) for name, metrics in class_metrics.items() if name != '_overall']
    class_list_sorted = sorted(class_list, key=lambda x: x[1]['support'], reverse=True)

    for name, metrics in class_list_sorted[:20]:
        metric_name = f"val_class_acc/{name}"
        mlflow.log_metric(metric_name, metrics['accuracy'], step=epoch)

    class_metrics_table = []
    for name, metrics in class_list_sorted:
        class_metrics_table.append({
            'class': name,
            'accuracy': f"{metrics['accuracy']:.4f}",
            'support': metrics['support']
        })

    metrics_json_path = f"class_metrics_epoch_{epoch}.json"
    with open(metrics_json_path, 'w') as f:
        json.dump(class_metrics_table, f, indent=2)

    mlflow.log_artifact(metrics_json_path)
    os.remove(metrics_json_path)

    print(f"✓ Logged per-class metrics for epoch {epoch}")


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    """Main training pipeline with class imbalance handling."""
    print("=" * 80)
    print("SIGN LANGUAGE CLASSIFIER - WITH ENHANCED FEATURES")
    print("=" * 80)

    shutdown_handler = GracefulShutdown()

    # MLFLOW SETUP
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'roman'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'SignNet'
    mlflow.set_tracking_uri("https://mlflow.schlaepfer.me")

    EXPERIMENT_NAME = "SignNetWord"
    RUN_NAME = f"Top150 Transformer"  # UPDATED
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ============================================================================
    # HYPERPARAMETERS (UPDATED)
    # ============================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MIN_SAMPLES_PER_CLASS = 70
    USE_CLASS_WEIGHTS = True
    USE_WEIGHTED_SAMPLER = True
    WEIGHT_BETA = 0.9999

    BATCH_SIZE = 256
    LEARNING_RATE = 1e-4  # Reduced from 3e-4
    HIDDEN_SIZE = 320
    DROPOUT_RATE = 0.55  # Increased from 0.35
    NUM_HEADS = 10
    NUM_LAYERS = 6
    ATTENTION_DROPOUT = 0.30  # NEW
    WEIGHT_DECAY = 1e-3  # Increased from 5e-4
    AUGMENT = True
    AUGMENT_PROBABILITY = 0.7

    # FEATURE ENGINEERING SETTINGS (NEW)
    USE_ENHANCED_FEATURES = False  # Toggle feature engineering
    INCLUDE_ACCELERATION = True  # Toggle acceleration features
    USE_FOCAL_LOSS = True  # Use Focal Loss
    USE_BALANCED_SOFTMAX = True
    if USE_BALANCED_SOFTMAX:
        USE_WEIGHTED_SAMPLER = False
        USE_CLASS_WEIGHTS = False
    INCLUDE_BONES = False                 # NEW
    INCLUDE_BONE_VELOCITY = False        # NEW (turn on later if memory allows)

    USE_EARLY_STOPPING = False
    PLATEAU_PATIENCE = 5  # Adaptive reduction trigger
    WARMUP_EPOCHS = 10
    NUM_EPOCHS = 400
    BASE_LR = 1e-4
    MIN_LR = 1e-6
    T_0 = 25           # First restart cycle length
    T_MULT = 2         # Multiply cycle length after each restart

    # ==================== OVERSAMPLING CONFIGURATION ====================
    OVERSAMPLE_CONFIG = {
        'ZWEI': 5,         # 10x oversampling (4.76% → target 40%+)
        'loc-SUED': 4,      # 8x oversampling (10.53%)
        'EINS': 3,          # 5x oversampling (32%)
        'MEISTENS': 3,      # 5x oversampling (21%)
        'UND': 4,           # 4x oversampling (25%)
        'ABER': 4,          # 4x oversampling (27%)
        'KOMMEN': 3,        # 3x oversampling (28%)
        'AUCH': 5,
        'cl-KOMMEN': 4,
    }


    number_of_classes = 300

    NPZ_DIR = "./word_landmarks_extracted"
    MODEL_SAVE_DIR = "./models_balanced"
    PLOTS_DIR = "./plots_balanced"

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    try:
        run = mlflow.start_run(log_system_metrics=True, run_name=RUN_NAME)

        try:
            mlflow.log_param("python_version", platform.python_version())
            mlflow.log_param("pytorch_version", torch.__version__)

            mlflow.log_params({
                "min_samples_per_class": MIN_SAMPLES_PER_CLASS,
                "use_class_weights": USE_CLASS_WEIGHTS,
                "use_weighted_sampler": USE_WEIGHTED_SAMPLER,
                "weight_beta": WEIGHT_BETA,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "optimizer": "AdamW",
                "num_epochs": NUM_EPOCHS,
                "hidden_size": HIDDEN_SIZE,
                "dropout_rate": DROPOUT_RATE,
                "augmentation_enabled": AUGMENT,
                "augmentation_probability": AUGMENT_PROBABILITY,
                "use_enhanced_features": USE_ENHANCED_FEATURES,  # NEW
                "include_acceleration": INCLUDE_ACCELERATION,  # NEW
                "use_focal_loss": USE_FOCAL_LOSS,  # NEW
                "attention_dropout": ATTENTION_DROPOUT,  # NEW
            })

            # STEP 1: Load dataset WITH ENHANCED FEATURES
            print(f"\n[STEP 1] Loading dataset with enhanced features={USE_ENHANCED_FEATURES}...")

            base_dataset = SignLanguageDataset(
                NPZ_DIR,
                debug=True,
                augment=False,
                use_enhanced_features=USE_ENHANCED_FEATURES,
                include_accel=INCLUDE_ACCELERATION,
                include_bones=INCLUDE_BONES,
                include_bone_velocity=INCLUDE_BONE_VELOCITY
            )


            # GET INPUT SIZE (will be larger if enhanced features enabled)
            sample_landmarks, _, _ = base_dataset[0]
            input_size = sample_landmarks.shape[1]
            print(f"\n>>> Input size: {input_size} (original: 1659)")
            mlflow.log_param("input_size", input_size)

            npz_files = sorted(Path(NPZ_DIR).glob("*.npz"))
            top_k_words, all_counts = build_topk_vocabulary(
                npz_files,
                K=number_of_classes,
                min_samples=MIN_SAMPLES_PER_CLASS,
                debug=True
            )

            print(f"\n[VOCABULARY] Using {len(top_k_words)} classes after filtering")


            # Compute class counts for class-aware augmentation
            train_class_counts = Counter()
            npz_files = sorted(Path(NPZ_DIR).glob("*.npz"))
            for npz_file in npz_files:
                try:
                    data = np.load(npz_file, allow_pickle=True)
                    gloss = data["glosses"][0]
                    if gloss in top_k_words:  # Only count classes in our vocabulary
                        train_class_counts[gloss] += 1
                except Exception:
                    continue

            dataset_train = SignLanguageDataset(
                NPZ_DIR,
                word_to_idx=base_dataset.word_to_idx,
                debug=False,
                augment=AUGMENT,
                augment_prob=AUGMENT_PROBABILITY,
                use_enhanced_features=USE_ENHANCED_FEATURES,
                include_accel=INCLUDE_ACCELERATION,
                include_bones=INCLUDE_BONES,
                include_bone_velocity=INCLUDE_BONE_VELOCITY,
                class_counts=train_class_counts  # NEW: Pass class counts
            )
            dataset_val = SignLanguageDataset(
                NPZ_DIR,
                word_to_idx=base_dataset.word_to_idx,
                debug=False,
                augment=False,
                use_enhanced_features=USE_ENHANCED_FEATURES,
                include_accel=INCLUDE_ACCELERATION,
                include_bones=INCLUDE_BONES,
                include_bone_velocity=INCLUDE_BONE_VELOCITY,
                class_counts=train_class_counts  # Even validation gets the mapping (though augment=False)
            )

            # STEP 2: Filter to top words
            print(f"\n[STEP 2] Filtering to vocabulary...")

            old_to_new_idx = {}
            for new_idx, word in enumerate(sorted(top_k_words)):
                old_idx = base_dataset.word_to_idx[word]
                old_to_new_idx[old_idx] = new_idx

            filtered_indices = []
            filtered_labels = []
            for i in range(len(base_dataset)):
                _, label, _ = base_dataset[i]
                old_label = label.item()
                if old_label in old_to_new_idx:
                    filtered_indices.append(i)
                    word = base_dataset.idx_to_word[old_label]
                    filtered_labels.append(word)

            new_idx_to_word = {new_idx: word for old_idx, new_idx in old_to_new_idx.items() for word in [base_dataset.idx_to_word[old_idx]]}


            print(f"  Filtered to {len(filtered_indices)} samples")

            # STEP 3: Split
            print(f"\n[STEP 3] Splitting dataset...")

            train_indices, val_indices = train_test_split(
                filtered_indices,
                test_size=0.2,
                random_state=42,
                stratify=filtered_labels
            )

            num_classes = len(top_k_words)
            print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}")
            print(f"  Classes: {num_classes}")

            # STEP 4: Compute class weights
            if USE_CLASS_WEIGHTS:
                print(f"\n[STEP 4] Computing class weights...")

                train_class_counts = Counter()
                for idx in train_indices:
                    _, old_label, _ = base_dataset[idx]
                    new_label = old_to_new_idx[old_label.item()]
                    train_class_counts[new_label] += 1

                class_weights = compute_effective_number_weights(
                    train_class_counts,
                    beta=WEIGHT_BETA
                )

                print(f"  Weight range: {class_weights.min():.2f} - {class_weights.max():.2f}")
                class_weights = class_weights.to(DEVICE)
            else:
                class_weights = None

                train_class_counts = Counter()
                for idx in train_indices:
                    _, old_label, _ = base_dataset[idx]
                    new_label = old_to_new_idx[old_label.item()]
                    train_class_counts[new_label] += 1

                balanced_counts_vec = torch.zeros(num_classes, dtype=torch.float32)
                for cls_idx, cnt in train_class_counts.items():
                    balanced_counts_vec[cls_idx] = float(cnt)

            # STEP 5: Create weighted sampler
            if USE_WEIGHTED_SAMPLER:
                print(f"\n[STEP 5] Creating weighted sampler...")

                sample_weights = []
                for idx in train_indices:
                    _, old_label, _ = base_dataset[idx]
                    new_label = old_to_new_idx[old_label.item()]
                    count = train_class_counts[new_label]
                    weight = 1.0 / np.sqrt(count)
                    sample_weights.append(weight)

                train_sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )

                train_subset = RemappedDataset(dataset_train, train_indices, old_to_new_idx)
                val_subset = RemappedDataset(dataset_val, val_indices, old_to_new_idx)

                train_loader = DataLoader(
                    train_subset,
                    batch_size=BATCH_SIZE,
                    sampler=train_sampler,
                    collate_fn=PadCollate(),
                    num_workers=4,
                    pin_memory=True,
                    prefetch_factor=4,
                    persistent_workers=True
                )
            else:
                train_subset = OversampledDataset(dataset_train, train_indices, old_to_new_idx, oversample_config=OVERSAMPLE_CONFIG)
                val_subset = RemappedDataset(dataset_val, val_indices, old_to_new_idx)

                train_loader = DataLoader(
                    train_subset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    collate_fn=PadCollate(),
                    num_workers=4,
                    pin_memory=True,
                    prefetch_factor=4,
                    persistent_workers=True
                )

            val_loader = DataLoader(
                val_subset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=PadCollate(),
                num_workers=4,
                pin_memory=True,
                prefetch_factor=4,
                persistent_workers=True
            )

            # STEP 6: Build model WITH RESIDUAL ATTENTION
            print(f"\n[STEP 6] Building model with residual attention...")


            model_raw = TransformerSignClassifierWithHandedness(
                input_size=input_size,
                hidden_size=HIDDEN_SIZE,      # e.g. 96 or 128
                num_classes=num_classes,
                num_layers=NUM_LAYERS,                 # 2–3 layers is a good start
                num_heads=NUM_HEADS,                  # keep consistent with hidden_size (must divide)
                dim_feedforward=4 * HIDDEN_SIZE,
                dropout_rate=DROPOUT_RATE,
                attention_dropout=ATTENTION_DROPOUT,
                debug=True
            ).to(DEVICE)

            if hasattr(torch, 'compile'):
                print("Compiling model with torch.compile...")
                model = torch.compile(model_raw, mode='max-autotune-no-cudagraphs')

            # STEP 7: Setup training WITH FOCAL LOSS
            print(f"\n[STEP 7] Setting up training...")

            criterion = MultiTaskLoss(
                label_smoothing=0.05,
                use_focal=(USE_FOCAL_LOSS and not USE_BALANCED_SOFTMAX),
                class_weights=(class_weights if USE_CLASS_WEIGHTS and not USE_BALANCED_SOFTMAX else None),
                use_balanced_softmax=USE_BALANCED_SOFTMAX,
                balanced_class_counts=balanced_counts_vec.to(DEVICE)
            )
            criterion = criterion.to(DEVICE)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=BASE_LR,
                weight_decay=WEIGHT_DECAY,
                betas=(0.9, 0.999)
            )

            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=WARMUP_EPOCHS
            )

            restart_scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=T_0,              # First cycle: 20 epochs
                T_mult=T_MULT,        # Next cycles: 40, 80, ... epochs
                eta_min=MIN_LR        # Minimum LR at cycle end
            )

            from torch.optim.lr_scheduler import SequentialLR
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, restart_scheduler],
                milestones=[WARMUP_EPOCHS]
            )

            early_stopping = EarlyStopping(
                patience=35,
                min_delta=0.0005,
                metric="val_acc",
                mode="max"
            )

            USE_SWA = True  # NEW
            SWA_START_FRACTION = 0.85  # start SWA after 70% of epochs  # NEW
            SWA_LR = 2e-5     # often same or slightly lower   # NEW

            if USE_SWA:
                swa_model = AveragedModel(model)
                swa_start = max(20, int(SWA_START_FRACTION * NUM_EPOCHS))
                swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)

            # STEP 8: Training loop
            print(f"\n[STEP 8] Starting training...")
            print("="*80)

            best_val_acc = 0
            best_epoch = 0
            train_losses = []
            val_losses = []
            train_accs = []
            val_accs = []
            epochs_trained = 0

            for epoch in range(NUM_EPOCHS):
                if shutdown_handler.is_interrupted():
                    break

                train_loss, train_topk_accs, train_hand_acc, train_sign_loss, train_hand_loss = train_epoch_interruptible(
                    model, train_loader, optimizer, criterion, DEVICE, epoch
                )

                if shutdown_handler.is_interrupted():
                    break

                val_loss, val_topk_accs, val_hand_acc, handedness_dist, confusion_mat, class_metrics, all_preds, all_labels = validate_epoch(
                    model, val_loader, criterion, DEVICE, new_idx_to_word
                )


                if shutdown_handler.is_interrupted():
                    break

                scheduler.step()

                if USE_SWA and epoch >= swa_start:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()


                epochs_trained += 1

                if val_topk_accs[1] > best_val_acc:
                    best_val_acc = val_topk_accs[1]
                    best_epoch = epoch
                    best_model_path = os.path.join(MODEL_SAVE_DIR, "sign_classifier_best_enhanced.pth")
                    torch.save(model.state_dict(), best_model_path)
                    log_class_metrics_to_mlflow(class_metrics, epoch)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accs.append(train_topk_accs[1])
                val_accs.append(val_topk_accs[1])

                lr = optimizer.param_groups[0]['lr']

                print(f"Epoch {epoch+1:4}/{NUM_EPOCHS} │ "
                    f"Loss: {train_loss:.4f}/{val_loss:.4f} │ "
                    f"Top1: {train_topk_accs[1]:.2%}/{val_topk_accs[1]:.2%} │ "
                    f"Top5: {train_topk_accs[5]:.2%}/{val_topk_accs[5]:.2%} │ "
                    f"LR: {lr:.2e}")

                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": train_topk_accs[1],
                    "train_top2_accuracy": train_topk_accs[2],
                    "train_top3_accuracy": train_topk_accs[3],
                    "train_top4_accuracy": train_topk_accs[4],
                    "train_top5_accuracy": train_topk_accs[5],
                    "val_accuracy": val_topk_accs[1],
                    "val_top2_accuracy": val_topk_accs[2],
                    "val_top3_accuracy": val_topk_accs[3],
                    "val_top4_accuracy": val_topk_accs[4],
                    "val_top5_accuracy": val_topk_accs[5],
                    "learning_rate": lr,
                }, step=epoch)

                if USE_EARLY_STOPPING:
                    if early_stopping(val_topk_accs[1], epoch):
                        print(f"\n[EARLY STOPPING] Training stopped at epoch {epoch+1}")
                        break

            # STEP 9: Save results
            print("\n" + "="*80)
            print(f"[TRAINING COMPLETE]")
            print(f"  Best Val Accuracy: {best_val_acc:.2%} at epoch {best_epoch+1}")
            print(f"  Total epochs: {epochs_trained}")
            print(f"  Enhanced features: {USE_ENHANCED_FEATURES}")
            print(f"  Input size: {input_size}")
            print("="*80)


            if USE_SWA and epochs_trained > swa_start:
                # Skipping update_bn since the model has no BatchNorm layers and forward needs lengths
                swa_model.eval()
                swa_val_loss, swa_val_topk, swa_hand_acc, *_ = validate_epoch(
                    swa_model, val_loader, criterion, DEVICE, new_idx_to_word
                )
                mlflow.log_metrics({
                    "swa_val_loss": swa_val_loss,
                    "swa_val_accuracy": swa_val_topk[1],
                    "swa_val_top5": swa_val_topk[5]
                }, step=epochs_trained)


                # Save SWA model
                swa_path = os.path.join(MODEL_SAVE_DIR, "sign_classifier_swa_enhanced.pth")
                torch.save(swa_model.module.state_dict() if hasattr(swa_model, "module") else swa_model.state_dict(), swa_path)
                mlflow.log_artifact(swa_path)

            final_model_path = os.path.join(MODEL_SAVE_DIR, "sign_classifier_final_enhanced.pth")
            torch.save(model_raw.state_dict(), final_model_path)

            mlflow.log_artifact(final_model_path)
            mlflow.pytorch.log_model(model_raw, "model")

            asyncio.run(send_message(
                f"Training complete (ENHANCED):\n"
                f"Best Val Acc: {best_val_acc:.2%}\n"
                f"Input size: {input_size}\n"
                f"Classes: {num_classes}\n"
                f"Epochs: {epochs_trained}",
                CHAT_ID
            ))

        finally:
            mlflow.end_run()

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        mlflow.end_run()


if __name__ == "__main__":
    main()
