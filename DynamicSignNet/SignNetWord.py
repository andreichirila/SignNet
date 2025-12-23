import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
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
import argparse
from SignNetConfig import (
    MAIN_MODEL_CONFIG,
    EXPERT_MODEL_CONFIG,
    HIERARCHY_CONFIG,
    OVERSAMPLE_CONFIG
)


# Add this function after all imports, before classes
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Sign Language Classifier Training')

    parser.add_argument(
        '--dataset-type',
        type=str,
        choices=['flat', 'split'],
        default='flat',
        help='Dataset structure: "flat" for single folder with train_test_split, "split" for train/test/val folders'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='./word_landmarks_extracted',
        help='Base directory containing dataset'
    )

    parser.add_argument(
        '--expert-name',
        type=str,
        default=None,
        choices=['direction_expert', 'kommen_expert', 'weather_expert'],
        help='If specified, trains a specialized expert model on a subset of classes.'
    )

    return parser.parse_args()



# ==================== FEATURE ENGINEERING MODULE (NEW) ====================
class EnhancedLandmarkFeatures:
    """
    Extract velocity, acceleration, and spatial features from MediaPipe landmarks
    State-of-the-art feature engineering for sign language recognition
    """

    # Define landmark counts based on extraction layout
    NUM_HAND_LANDMARKS = 21
    NUM_HANDS = 2
    NUM_FACE_LANDMARKS = 478
    NUM_POSE_LANDMARKS = 33
    TOTAL_LANDMARKS = NUM_HANDS * NUM_HAND_LANDMARKS + NUM_FACE_LANDMARKS + NUM_POSE_LANDMARKS  # 553

    # Flat feature offsets (each landmark has x, y, z)
    HAND_FLAT_SIZE = NUM_HANDS * NUM_HAND_LANDMARKS * 3  # 126
    FACE_FLAT_SIZE = NUM_FACE_LANDMARKS * 3              # 1434
    POSE_FLAT_SIZE = NUM_POSE_LANDMARKS * 3              # 99
    TOTAL_FLAT_SIZE = HAND_FLAT_SIZE + FACE_FLAT_SIZE + POSE_FLAT_SIZE  # 1659

    @staticmethod
    def reshape_landmarks(landmarks_flat):
        """
        Reshape flat landmarks to (T, num_landmarks, 3).
        
        Input layout from extraction:
        [hands (126), face (1434), pose (99)] = 1659 features
        
        Output layout:
        [left_hand (21), right_hand (21), face (478), pose (33)] × 3 = 553 landmarks
        """
        T = landmarks_flat.shape[0]
        F = landmarks_flat.shape[1]
        
        # Handle the expected 1659 feature case
        if F == 1659:
            # Extract each component
            hands_flat = landmarks_flat[:, :126]      # (T, 126)
            face_flat = landmarks_flat[:, 126:1560]   # (T, 1434)
            pose_flat = landmarks_flat[:, 1560:1659]  # (T, 99)
            
            # Reshape to (T, num_landmarks, 3)
            hands_3d = hands_flat.reshape(T, 42, 3)   # 2 hands × 21 landmarks
            face_3d = face_flat.reshape(T, 478, 3)
            pose_3d = pose_flat.reshape(T, 33, 3)
            
            # Concatenate: [hands, face, pose]
            landmarks_3d = np.concatenate([hands_3d, face_3d, pose_3d], axis=1)
            return landmarks_3d  # (T, 553, 3)
        
        # Fallback for other sizes
        if F % 3 != 0:
            pad_size = 3 - (F % 3)
            landmarks_flat = np.pad(landmarks_flat, ((0, 0), (0, pad_size)), mode='constant')
            F = landmarks_flat.shape[1]

        num_landmarks = F // 3
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
        """Compute inter-hand distance."""
        T, L, _ = landmarks_3d.shape
        
        # Hand landmark indices after reshape
        LH_START, LH_END = 0, 21      # Left hand: landmarks 0-20
        RH_START, RH_END = 21, 42     # Right hand: landmarks 21-41
        
        if L < RH_END:
            return np.zeros((T, 1))
            
        left_center = landmarks_3d[:, LH_START:LH_END, :].mean(axis=1)
        right_center = landmarks_3d[:, RH_START:RH_END, :].mean(axis=1)
        distances = np.linalg.norm(left_center - right_center, axis=1, keepdims=True)
        return distances

    @staticmethod
    def compute_hand_to_face_distances(landmarks_3d):
        """Distance from each hand center to face center."""
        T, L, _ = landmarks_3d.shape
        
        # Landmark indices after reshape
        LH_START, LH_END = 0, 21
        RH_START, RH_END = 21, 42
        FACE_START, FACE_END = 42, 520  # 478 face landmarks
        
        if L < FACE_END:
            return np.zeros((T, 1)), np.zeros((T, 1))
            
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
        Default bone edges for landmark layout after reshape:
        - Left hand:  0..20   (21 landmarks)
        - Right hand: 21..41  (21 landmarks)
        - Face:       42..519 (478 landmarks) -> sparse edges only
        - Pose:       520..552 (33 landmarks)
        
        Total: 553 landmarks
        """
        EXPECTED_L = 553
        if num_landmarks != EXPECTED_L:
            print(f"\n[WARNING] num_landmarks={num_landmarks} != EXPECTED_L={EXPECTED_L}")
            return [(i, i + 1) for i in range(max(0, num_landmarks - 1))]

        LH_START = 0
        RH_START = 21
        POSE_START = 520

        edges = []

        # MediaPipe Hands finger chains (relative to hand base)
        hand_chains = [
            [0, 1, 2, 3, 4],        # thumb
            [0, 5, 6, 7, 8],        # index
            [0, 9, 10, 11, 12],     # middle
            [0, 13, 14, 15, 16],    # ring
            [0, 17, 18, 19, 20],    # pinky
        ]

        def add_hand_edges(base):
            for chain in hand_chains:
                for a, b in zip(chain[:-1], chain[1:]):
                    edges.append((base + a, base + b))

        # Left and Right hands
        add_hand_edges(LH_START)
        add_hand_edges(RH_START)

        # Pose edges (upper body)
        def P(i):
            return POSE_START + i

        pose_edges = [
            (P(11), P(13)), (P(13), P(15)),     # left shoulder-elbow-wrist
            (P(12), P(14)), (P(14), P(16)),     # right shoulder-elbow-wrist
            (P(11), P(12)),                     # clavicle
            (P(11), P(23)), (P(12), P(24)),     # shoulders to hips
            (P(23), P(24)),                     # hip line
            (P(0),  P(11)), (P(0),  P(12)),     # nose to shoulders
        ]
        edges.extend(pose_edges)

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

class SkeletalAugmentation:
    """
    Skeleton-Based Augmentation with Bone Length Preservation.
    
    Applies Gaussian noise to joint positions while maintaining anatomically
    correct bone lengths through kinematic constraints.
    
    Expected input layout from landmark_extraction.py:
    [hands (126), face (1434), pose (99)] = 1659 features
    - Hands: 2 hands × 21 landmarks × 3 coords = 126
    - Face: 478 landmarks × 3 coords = 1434
    - Pose: 33 landmarks × 3 coords = 99
    """
    
    # Hand bone connections (landmark indices within a 21-point hand)
    HAND_BONES = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]
    
    # Pose bone connections (upper body - indices within 33-point pose)
    POSE_BONES = [
        # Torso
        (11, 12),  # Shoulders
        (11, 23), (12, 24),  # Shoulder to hip
        (23, 24),  # Hips
        # Left arm
        (11, 13), (13, 15),  # Shoulder -> Elbow -> Wrist
        # Right arm
        (12, 14), (14, 16),  # Shoulder -> Elbow -> Wrist
    ]
    
    # Flat feature offsets matching landmark_extraction.py layout
    # Layout: [hands (126), face (1434), pose (99)]
    LEFT_HAND_OFFSET = 0       # Left hand: features 0-62 (21 landmarks × 3)
    RIGHT_HAND_OFFSET = 63     # Right hand: features 63-125 (21 landmarks × 3)
    FACE_OFFSET = 126          # Face: features 126-1559 (478 landmarks × 3)
    POSE_OFFSET = 1560         # Pose: features 1560-1658 (33 landmarks × 3)
    
    def __init__(self, 
                 sigma: float = 0.015, 
                 probability: float = 0.5,
                 preserve_bones: bool = True,
                 num_iterations: int = 3):
        """
        Args:
            sigma: Standard deviation of Gaussian noise (0.01-0.02 recommended)
            probability: Probability of applying augmentation
            preserve_bones: Whether to enforce bone length constraints after perturbation
            num_iterations: Number of iterations for bone length correction
        """
        self.sigma = sigma
        self.probability = probability
        self.preserve_bones = preserve_bones
        self.num_iterations = num_iterations
    
    def _extract_landmarks_3d(self, frame: np.ndarray, offset: int, num_landmarks: int) -> np.ndarray:
        """Extract landmarks as (num_landmarks, 3) from flat feature vector."""
        start = offset
        end = offset + num_landmarks * 3
        if end > len(frame):
            return np.zeros((num_landmarks, 3), dtype=frame.dtype)
        return frame[start:end].reshape(num_landmarks, 3)
    
    def _insert_landmarks_3d(self, frame: np.ndarray, landmarks_3d: np.ndarray, offset: int) -> np.ndarray:
        """Insert (num_landmarks, 3) back into flat feature vector."""
        num_landmarks = landmarks_3d.shape[0]
        start = offset
        end = offset + num_landmarks * 3
        if end <= len(frame):
            frame[start:end] = landmarks_3d.flatten()
        return frame
    
    def _compute_bone_lengths(self, landmarks: np.ndarray, bone_connections: list) -> dict:
        """Compute original bone lengths for preservation."""
        bone_lengths = {}
        for i, (start_idx, end_idx) in enumerate(bone_connections):
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                bone_vec = landmarks[end_idx] - landmarks[start_idx]
                bone_lengths[i] = np.linalg.norm(bone_vec)
        return bone_lengths
    
    def _apply_bone_length_constraints(self, 
                                        landmarks: np.ndarray, 
                                        original_lengths: dict, 
                                        bone_connections: list) -> np.ndarray:
        """
        Iteratively adjust joint positions to preserve bone lengths.
        Uses a simple relaxation approach (FABRIK-inspired).
        """
        landmarks = landmarks.copy()
        
        for _ in range(self.num_iterations):
            for bone_idx, (start_idx, end_idx) in enumerate(bone_connections):
                if bone_idx not in original_lengths:
                    continue
                if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                    continue
                    
                target_length = original_lengths[bone_idx]
                if target_length < 1e-6:  # Skip zero-length bones
                    continue
                
                # Current bone vector
                bone_vec = landmarks[end_idx] - landmarks[start_idx]
                current_length = np.linalg.norm(bone_vec)
                
                if current_length < 1e-6:
                    continue
                
                # Scale factor to restore original length
                scale = target_length / current_length
                
                # Adjust both points symmetrically
                correction = bone_vec * (scale - 1.0) * 0.5
                landmarks[start_idx] -= correction
                landmarks[end_idx] += correction
        
        return landmarks
    
    def _perturb_landmarks(self, landmarks: np.ndarray, bone_connections: list) -> np.ndarray:
        """Apply Gaussian perturbation and optionally preserve bone lengths."""
        if len(landmarks) == 0:
            return landmarks
        
        # Store original bone lengths
        original_lengths = self._compute_bone_lengths(landmarks, bone_connections)
        
        # Apply Gaussian noise
        noise = np.random.normal(0, self.sigma, landmarks.shape).astype(landmarks.dtype)
        perturbed = landmarks + noise
        
        # Restore bone lengths if enabled
        if self.preserve_bones and len(original_lengths) > 0:
            perturbed = self._apply_bone_length_constraints(
                perturbed, original_lengths, bone_connections
            )
        
        return perturbed
    
    def __call__(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply skeletal augmentation to a sequence of landmarks.
        
        Args:
            landmarks: Shape (T, F) where T is time and F is flattened features (1659)
                       Layout: [hands (126), face (1434), pose (99)]
        
        Returns:
            Augmented landmarks with same shape
        """
        if np.random.random() > self.probability:
            return landmarks
        
        landmarks = landmarks.copy()
        T, F = landmarks.shape
        
        # Only process if we have the expected feature size
        if F < 1659:
            return landmarks
        
        for t in range(T):
            frame = landmarks[t].copy()
            
            # Process left hand (21 landmarks starting at offset 0)
            left_hand = self._extract_landmarks_3d(frame, self.LEFT_HAND_OFFSET, 21)
            if not np.allclose(left_hand, 0):  # Only if hand detected
                left_perturbed = self._perturb_landmarks(left_hand, self.HAND_BONES)
                frame = self._insert_landmarks_3d(frame, left_perturbed, self.LEFT_HAND_OFFSET)
            
            # Process right hand (21 landmarks starting at offset 63)
            right_hand = self._extract_landmarks_3d(frame, self.RIGHT_HAND_OFFSET, 21)
            if not np.allclose(right_hand, 0):  # Only if hand detected
                right_perturbed = self._perturb_landmarks(right_hand, self.HAND_BONES)
                frame = self._insert_landmarks_3d(frame, right_perturbed, self.RIGHT_HAND_OFFSET)
            
            # Process pose (33 landmarks starting at offset 1560)
            pose = self._extract_landmarks_3d(frame, self.POSE_OFFSET, 33)
            if not np.allclose(pose, 0):  # Only if pose detected
                pose_perturbed = self._perturb_landmarks(pose, self.POSE_BONES)
                frame = self._insert_landmarks_3d(frame, pose_perturbed, self.POSE_OFFSET)
            
            landmarks[t] = frame
        
        return landmarks
    
class LandmarkOcclusionAugmentation:
    """
    Landmark Occlusion Simulation for robust sign language recognition.
    
    Simulates real-world scenarios where MediaPipe fails to detect certain
    body parts due to occlusion, motion blur, or poor lighting.
    
    Expected input layout from landmark_extraction.py:
    [hands (126), face (1434), pose (99)] = 1659 features
    """
    
    # Anatomical region definitions (flat feature indices)
    REGIONS = {
        'left_hand': (0, 63),           # 21 landmarks × 3 coords
        'right_hand': (63, 126),        # 21 landmarks × 3 coords
        'face': (126, 1560),            # 478 landmarks × 3 coords
        'pose': (1560, 1659),           # 33 landmarks × 3 coords
    }
    
    # Sub-regions for more granular occlusion (relative to hand start)
    # MediaPipe hand landmarks: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
    HAND_SUB_REGIONS = {
        'thumb': (1*3, 5*3),            # landmarks 1-4 × 3 coords = 12 features
        'index': (5*3, 9*3),            # landmarks 5-8 × 3 coords = 12 features
        'middle': (9*3, 13*3),          # landmarks 9-12 × 3 coords = 12 features
        'ring': (13*3, 17*3),           # landmarks 13-16 × 3 coords = 12 features
        'pinky': (17*3, 21*3),          # landmarks 17-20 × 3 coords = 12 features
    }
    
    # Face sub-regions (approximate MediaPipe face mesh regions)
    FACE_SUB_REGIONS = {
        'left_eye': (126, 126 + 48),    # ~16 landmarks × 3
        'right_eye': (126 + 48, 126 + 96),
        'nose': (126 + 96, 126 + 126),  # ~10 landmarks × 3
        'mouth': (126 + 126, 126 + 186), # ~20 landmarks × 3
        'left_cheek': (126 + 186, 126 + 246),
        'right_cheek': (126 + 246, 126 + 306),
    }
    
    # Pose sub-regions - CORRECTED for non-contiguous MediaPipe layout
    # MediaPipe Pose: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    # Pose starts at flat index 1560, each landmark = 3 values (x,y,z)
    POSE_SUB_REGIONS = {
        # Left arm: landmarks 11 (shoulder), 13 (elbow), 15 (wrist)
        # These are NOT contiguous, so we target the key joints individually
        'left_arm': [
            (1560 + 11*3, 1560 + 12*3),  # Left shoulder (landmark 11)
            (1560 + 13*3, 1560 + 14*3),  # Left elbow (landmark 13)
            (1560 + 15*3, 1560 + 16*3),  # Left wrist (landmark 15)
        ],
        # Right arm: landmarks 12 (shoulder), 14 (elbow), 16 (wrist)
        'right_arm': [
            (1560 + 12*3, 1560 + 13*3),  # Right shoulder (landmark 12)
            (1560 + 14*3, 1560 + 15*3),  # Right elbow (landmark 14)
            (1560 + 16*3, 1560 + 17*3),  # Right wrist (landmark 16)
        ],
        # Torso: shoulders (11,12) and hips (23,24)
        'torso': [
            (1560 + 11*3, 1560 + 13*3),  # Both shoulders (landmarks 11-12)
            (1560 + 23*3, 1560 + 25*3),  # Both hips (landmarks 23-24)
        ],
    }
    
    # Realistic occlusion patterns (which regions tend to be occluded together)
    OCCLUSION_PATTERNS = [
        # Single hand occlusion (most common in signing)
        ['left_hand'],
        ['right_hand'],
        # Both hands near face (common gesture)
        ['left_hand', 'right_hand'],
        # Partial face occlusion (hand covering mouth/nose)
        ['face'],
        # Arm tracking loss
        ['left_arm'],
        ['right_arm'],
    ]
    
    def __init__(self,
                 region_dropout_prob: float = 0.15,
                 temporal_dropout_prob: float = 0.10,
                 max_temporal_dropout_frames: int = 5,
                 sub_region_dropout_prob: float = 0.20,
                 use_realistic_patterns: bool = True,
                 probability: float = 0.5):
        """
        Args:
            region_dropout_prob: Probability of dropping an entire anatomical region
            temporal_dropout_prob: Probability of temporal dropout (consecutive frame occlusion)
            max_temporal_dropout_frames: Maximum consecutive frames to occlude
            sub_region_dropout_prob: Probability of dropping sub-regions (fingers, facial features)
            use_realistic_patterns: Use realistic co-occlusion patterns
            probability: Overall probability of applying any occlusion
        """
        self.region_dropout_prob = region_dropout_prob
        self.temporal_dropout_prob = temporal_dropout_prob
        self.max_temporal_dropout_frames = max_temporal_dropout_frames
        self.sub_region_dropout_prob = sub_region_dropout_prob
        self.use_realistic_patterns = use_realistic_patterns
        self.probability = probability
    
    def _zero_region(self, landmarks: np.ndarray, start: int, end: int, 
                     frame_start: int = None, frame_end: int = None) -> np.ndarray:
        """Zero out a region of landmarks, optionally for specific frames only."""
        if frame_start is not None and frame_end is not None:
            landmarks[frame_start:frame_end, start:end] = 0.0
        else:
            landmarks[:, start:end] = 0.0
        return landmarks
    
    def _apply_region_dropout(self, landmarks: np.ndarray) -> np.ndarray:
        """Drop entire anatomical regions."""
        if self.use_realistic_patterns:
            # Use realistic co-occlusion patterns
            if np.random.random() < self.region_dropout_prob:
                pattern = self.OCCLUSION_PATTERNS[np.random.randint(len(self.OCCLUSION_PATTERNS))]
                for region_name in pattern:
                    if region_name in self.REGIONS:
                        start, end = self.REGIONS[region_name]
                        landmarks = self._zero_region(landmarks, start, end)
                    elif region_name in self.POSE_SUB_REGIONS:
                        # Handle list of ranges for non-contiguous pose regions
                        ranges = self.POSE_SUB_REGIONS[region_name]
                        for start, end in ranges:
                            landmarks = self._zero_region(landmarks, start, end)
        else:
            # Independent region dropout
            for region_name, (start, end) in self.REGIONS.items():
                if np.random.random() < self.region_dropout_prob:
                    landmarks = self._zero_region(landmarks, start, end)
        
        return landmarks
    
    def _apply_temporal_dropout(self, landmarks: np.ndarray) -> np.ndarray:
        """Simulate tracking loss over consecutive frames."""
        T = landmarks.shape[0]
        if T < 4:  # Need at least 4 frames for meaningful temporal dropout
            return landmarks
        
        for region_name, (start, end) in self.REGIONS.items():
            if np.random.random() < self.temporal_dropout_prob:
                # Calculate max dropout length (at most half the sequence)
                max_dropout = min(self.max_temporal_dropout_frames + 1, T // 2)
                if max_dropout <= 1:  # Safety check
                    continue
                    
                dropout_length = np.random.randint(1, max_dropout)
                max_start = T - dropout_length
                if max_start <= 0:  # Safety check
                    continue
                    
                start_frame = np.random.randint(0, max_start)
                end_frame = min(start_frame + dropout_length, T)
                
                landmarks = self._zero_region(landmarks, start, end, start_frame, end_frame)
        
        return landmarks
    
    def _apply_sub_region_dropout(self, landmarks: np.ndarray) -> np.ndarray:
        """Drop sub-regions like individual fingers or facial features."""
        # Hand sub-regions (apply to both hands)
        for hand_offset in [0, 63]:  # left_hand starts at 0, right_hand at 63
            for sub_name, (rel_start, rel_end) in self.HAND_SUB_REGIONS.items():
                if np.random.random() < self.sub_region_dropout_prob:
                    abs_start = hand_offset + rel_start
                    abs_end = hand_offset + rel_end
                    landmarks = self._zero_region(landmarks, abs_start, abs_end)
        
        # Face sub-regions
        for sub_name, (start, end) in self.FACE_SUB_REGIONS.items():
            if np.random.random() < self.sub_region_dropout_prob * 0.5:  # Less aggressive for face
                landmarks = self._zero_region(landmarks, start, end)
        
        return landmarks
    
    def _apply_gradual_occlusion(self, landmarks: np.ndarray) -> np.ndarray:
        """Simulate gradual tracking loss/recovery (fading in/out)."""
        T = landmarks.shape[0]
        if T < 6:  # Need at least 6 frames for fade effect
            return landmarks
        
        for region_name, (start, end) in self.REGIONS.items():
            if np.random.random() < self.temporal_dropout_prob * 0.5:
                # Create fade-out-fade-in pattern
                max_fade = min(5, T // 3)
                if max_fade < 2:  # Safety check
                    continue
                fade_length = np.random.randint(2, max_fade + 1)  # +1 because randint is exclusive on high
                
                # Ensure we have room for center_frame selection
                if fade_length >= T - fade_length:
                    continue
                    
                center_frame = np.random.randint(fade_length, T - fade_length)
                
                # Create alpha mask (1 = visible, 0 = occluded)
                alpha = np.ones(T)
                for i in range(fade_length):
                    # Fade out before center
                    if center_frame - fade_length + i >= 0:
                        alpha[center_frame - fade_length + i] = i / fade_length
                    # Fade in after center
                    if center_frame + i < T:
                        alpha[center_frame + i] = i / fade_length
                
                # Center frames fully occluded
                alpha[max(0, center_frame - 1):min(T, center_frame + 2)] = 0.0
                
                # Apply alpha mask
                for t in range(T):
                    landmarks[t, start:end] *= alpha[t]
        
        return landmarks
    
    def __call__(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply occlusion augmentation to a sequence of landmarks.
        
        Args:
            landmarks: Shape (T, F) where T is time and F is flattened features (1659)
        
        Returns:
            Augmented landmarks with simulated occlusions
        """
        if np.random.random() > self.probability:
            return landmarks
        
        landmarks = landmarks.copy()
        T, F = landmarks.shape
        
        # Only process if we have the expected feature size
        if F < 1659:
            return landmarks
        
        # Apply different types of occlusion (not all at once)
        occlusion_type = np.random.random()
        
        if occlusion_type < 0.4:
            # Region dropout (most common)
            landmarks = self._apply_region_dropout(landmarks)
        elif occlusion_type < 0.7:
            # Temporal dropout (tracking loss)
            landmarks = self._apply_temporal_dropout(landmarks)
        elif occlusion_type < 0.9:
            # Sub-region dropout (partial occlusion)
            landmarks = self._apply_sub_region_dropout(landmarks)
        else:
            # Gradual occlusion (realistic fade in/out)
            landmarks = self._apply_gradual_occlusion(landmarks)
        
        return landmarks


class MirrorAugmentation:
    """
    Horizontal Mirroring Augmentation for sign language recognition.
    
    Flips the sign horizontally by:
    1. Swapping left and right hand landmarks
    2. Flipping x-coordinates (1 - x) for all landmarks
    3. Swapping left/right pose landmarks (shoulders, elbows, wrists, etc.)
    
    This effectively creates the mirror image of a sign, which is useful for:
    - Data augmentation (doubles effective dataset size)
    - Learning hand-invariant features
    - Handling left-handed vs right-handed signers
    
    Expected input layout from landmark_extraction.py:
    [hands (126), face (1434), pose (99)] = 1659 features
    """
    
    # Feature layout (each landmark has x, y, z = 3 values)
    LEFT_HAND_START = 0
    LEFT_HAND_END = 63      # 21 landmarks × 3
    RIGHT_HAND_START = 63
    RIGHT_HAND_END = 126    # 21 landmarks × 3
    FACE_START = 126
    FACE_END = 1560         # 478 landmarks × 3
    POSE_START = 1560
    POSE_END = 1659         # 33 landmarks × 3
    
    # MediaPipe Pose landmark pairs that need swapping (left ↔ right)
    # Format: (left_landmark_idx, right_landmark_idx)
    POSE_SWAP_PAIRS = [
        (11, 12),  # Left/Right shoulder
        (13, 14),  # Left/Right elbow
        (15, 16),  # Left/Right wrist
        (17, 18),  # Left/Right pinky
        (19, 20),  # Left/Right index
        (21, 22),  # Left/Right thumb
        (23, 24),  # Left/Right hip
        (25, 26),  # Left/Right knee
        (27, 28),  # Left/Right ankle
        (29, 30),  # Left/Right heel
        (31, 32),  # Left/Right foot index
    ]
    
    # MediaPipe Face Mesh pairs for left/right symmetry (approximate)
    # These are the main symmetric landmarks - face mesh has complex topology
    # We'll flip x-coordinates for all face landmarks, which handles symmetry naturally
    
    def __init__(self, probability: float = 0.5):
        """
        Args:
            probability: Probability of applying mirroring augmentation
        """
        self.probability = probability
    
    def _flip_x_coordinates(self, landmarks: np.ndarray, start: int, end: int) -> np.ndarray:
        """
        Flip x-coordinates within a region.
        For MediaPipe normalized coordinates, x is in [0, 1], so flip = 1 - x
        
        Args:
            landmarks: Shape (T, F) where F is flattened features
            start: Start index of region
            end: End index of region
        """
        # x coordinates are at indices 0, 3, 6, 9, ... (every 3rd starting from 0)
        for i in range(start, end, 3):
            landmarks[:, i] = 1.0 - landmarks[:, i]
        return landmarks
    
    def _swap_hands(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Swap left and right hand landmarks.
        """
        # Copy left hand
        left_hand = landmarks[:, self.LEFT_HAND_START:self.LEFT_HAND_END].copy()
        # Copy right hand
        right_hand = landmarks[:, self.RIGHT_HAND_START:self.RIGHT_HAND_END].copy()
        
        # Swap
        landmarks[:, self.LEFT_HAND_START:self.LEFT_HAND_END] = right_hand
        landmarks[:, self.RIGHT_HAND_START:self.RIGHT_HAND_END] = left_hand
        
        return landmarks
    
    def _swap_pose_pairs(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Swap left/right pose landmark pairs.
        """
        for left_idx, right_idx in self.POSE_SWAP_PAIRS:
            # Calculate flat indices (each landmark = 3 values)
            left_start = self.POSE_START + left_idx * 3
            left_end = left_start + 3
            right_start = self.POSE_START + right_idx * 3
            right_end = right_start + 3
            
            # Bounds check
            if right_end <= self.POSE_END:
                # Swap the landmarks
                left_coords = landmarks[:, left_start:left_end].copy()
                right_coords = landmarks[:, right_start:right_end].copy()
                landmarks[:, left_start:left_end] = right_coords
                landmarks[:, right_start:right_end] = left_coords
        
        return landmarks
    
    def __call__(self, landmarks: np.ndarray, return_applied: bool = False):
        """
        Apply mirror augmentation to a sequence of landmarks.
        
        Args:
            landmarks: Shape (T, F) where T is time and F is flattened features (1659)
            return_applied: If True, returns tuple (landmarks, was_applied)
        
        Returns:
            Mirrored landmarks with same shape, or tuple (landmarks, was_applied)
        """
        if np.random.random() > self.probability:
            if return_applied:
                return landmarks, False
            return landmarks
        
        landmarks = landmarks.copy()
        T, F = landmarks.shape
        
        # Only process if we have the expected feature size
        if F < 1659:
            if return_applied:
                return landmarks, False
            return landmarks
        
        # Step 1: Swap left and right hands
        landmarks = self._swap_hands(landmarks)
        
        # Step 2: Swap left/right pose landmark pairs
        landmarks = self._swap_pose_pairs(landmarks)
        
        # Step 3: Flip all x-coordinates (creates the mirror image)
        # Hands
        landmarks = self._flip_x_coordinates(landmarks, self.LEFT_HAND_START, self.RIGHT_HAND_END)
        # Face
        landmarks = self._flip_x_coordinates(landmarks, self.FACE_START, self.FACE_END)
        # Pose
        landmarks = self._flip_x_coordinates(landmarks, self.POSE_START, self.POSE_END)
        
        if return_applied:
            return landmarks, True
        return landmarks


class SignLanguageDataset(Dataset):
    """
    RAM-Cached Dataset for high-speed training.
    Loads all .npz files into memory at startup to eliminate disk I/O bottlenecks.
    """
    def __init__(self, npz_dir, word_to_idx=None, debug=True, augment=False, augment_prob=0.7,
                 use_enhanced_features=False, include_accel=False,
                 include_bones=True, include_bone_velocity=False, class_counts=None,
                 use_skeletal_augmentation=True, skeletal_sigma=0.015, skeletal_probability=0.5,
                 use_occlusion_augmentation=True, occlusion_probability=0.3,
                 use_mirror_augmentation=True, mirror_probability=0.5):
        self.npz_dir = Path(npz_dir)
        self.npz_files = sorted(self.npz_dir.glob("*.npz"))
        self.debug = debug

        self.augment = augment
        self.use_enhanced_features = use_enhanced_features
        self.include_accel = include_accel
        self.include_bones = include_bones
        self.include_bone_velocity = include_bone_velocity
        self.class_counts = class_counts

        # --- AUGMENTATION SETUP ---
        if augment:
            self.augmentation = TemporalAugmentation(
                class_counts=class_counts,
                base_prob=0.5,
                strength=0.5,
                prob=augment_prob
            )
            if use_skeletal_augmentation:
                self.skeletal_augmentation = SkeletalAugmentation(
                    sigma=skeletal_sigma,
                    probability=skeletal_probability,
                    preserve_bones=True,
                    num_iterations=3
                )
            else:
                self.skeletal_augmentation = None
            
            if use_occlusion_augmentation:
                self.occlusion_augmentation = LandmarkOcclusionAugmentation(
                    region_dropout_prob=0.20,
                    temporal_dropout_prob=0.15,
                    max_temporal_dropout_frames=7,
                    sub_region_dropout_prob=0.25,
                    use_realistic_patterns=True,
                    probability=occlusion_probability
                )
            else:
                self.occlusion_augmentation = None
            
            if use_mirror_augmentation:
                self.mirror_augmentation = MirrorAugmentation(
                    probability=mirror_probability
                )
            else:
                self.mirror_augmentation = None
        else:
            self.skeletal_augmentation = None
            self.occlusion_augmentation = None
            self.mirror_augmentation = None

        # --- VOCABULARY SETUP ---
        if word_to_idx is None:
            self.word_to_idx = {}
            # We will build vocab during the caching loop to avoid double-reading files
            build_vocab = True
        else:
            self.word_to_idx = word_to_idx
            build_vocab = False

        # --- RAM CACHING LOOP ---
        self.data_cache = []
        print(f"\n[DATASET] Caching {len(self.npz_files)} files to RAM...")
        
        # Use tqdm to show progress
        for npz_file in tqdm(self.npz_files, desc="Loading Data"):
            try:
                data = np.load(npz_file, allow_pickle=True)
                
                # Extract and cast to float32 immediately to save memory
                landmarks = data["landmarks"].astype(np.float32)
                
                # Extract gloss
                glosses = data["glosses"]
                if len(glosses) > 0:
                    gloss_item = glosses[0]
                    if isinstance(gloss_item, (np.ndarray, np.str_, bytes)):
                        gloss = str(gloss_item)
                    else:
                        gloss = gloss_item
                else:
                    gloss = "UNKNOWN"
                
                # Build vocab if needed
                if build_vocab:
                    if gloss not in self.word_to_idx:
                        self.word_to_idx[gloss] = len(self.word_to_idx)
                
                label = self.word_to_idx.get(gloss, 0)
                
                # Store lightweight tuple in list
                self.data_cache.append({
                    "landmarks": landmarks,
                    "label": label,
                    "gloss": gloss
                })
                
            except Exception as e:
                print(f"  [WARNING] Skipping corrupt file {npz_file}: {e}")

        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}

        if debug:
            print(f"\n[DEBUG] Dataset Loaded")
            print(f"  Cached samples: {len(self.data_cache)}")
            print(f"  Vocabulary size: {len(self.word_to_idx)}")

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        # FAST: Retrieve from RAM
        item = self.data_cache[idx]
        
        # We must copy() because augmentations modify data in-place
        # and we don't want to corrupt the cached original version
        landmarks = item["landmarks"].copy()
        label = item["label"]
        gloss = item["gloss"]

        # 1. Skeletal Augmentation (modifies raw coordinates)
        if self.augment and self.skeletal_augmentation is not None:
            landmarks = self.skeletal_augmentation(landmarks)

        # 2. Occlusion Augmentation (masks raw coordinates)
        if self.augment and self.occlusion_augmentation is not None:
            landmarks = self.occlusion_augmentation(landmarks)

        # 3. Mirror Augmentation (flips raw coordinates)
        if self.augment and self.mirror_augmentation is not None:
            landmarks, _ = self.mirror_augmentation(landmarks, return_applied=True)

        # 4. Feature Engineering (calculates velocity/bones from modified landmarks)
        if self.use_enhanced_features:
            landmarks = EnhancedLandmarkFeatures.extract_all_features(
                landmarks,
                fps=25,
                is_train=self.augment,
                include_accel=self.include_accel,
                include_bones=self.include_bones,
                include_bone_velocity=self.include_bone_velocity
            )

        # 5. Temporal Augmentation (modifies final feature vector)
        if self.augment:
            landmarks = self.augmentation(landmarks, class_label=gloss)

        # Convert to tensors
        landmarks_tensor = torch.from_numpy(landmarks).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return landmarks_tensor, label_tensor


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
        landmarks, old_label = self.base_dataset[base_idx]

        old_label_val = old_label.item()

        if old_label_val not in self.old_to_new_idx:
            raise ValueError(f"Label {old_label_val} not in remapping dict!")

        new_label = self.old_to_new_idx[old_label_val]

        return landmarks, torch.tensor(new_label, dtype=torch.long)

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
            _, label = self.base_dataset[real_idx]
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


class TransformerSignClassifier(nn.Module):
    """
    Transformer encoder model for sign language classification.
    Single-task model (sign classification only).
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

        # Sign classification head
        self.fc_sign = nn.Linear(hidden_size, num_classes)

        if debug:
            print(f"[DEBUG] TransformerSignClassifier initialized")
            print(f"  Input size: {input_size}")
            print(f"  Hidden size: {hidden_size}")
            print(f"  Num classes: {num_classes}")
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
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, hidden)

        # Global average pooling over time (mask-aware)
        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).float().unsqueeze(-1)  # (B, T, 1)
            x_masked = x * mask
            lengths = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
            pooled = x_masked.sum(dim=1) / lengths    # (B, hidden)
        else:
            pooled = x.mean(dim=1)  # (B, hidden)

        pooled = self.dropout(pooled)

        sign_logits = self.fc_sign(pooled)

        return sign_logits



class PadCollate:
    def __call__(self, batch):
        landmarks_list = [item[0] for item in batch]
        labels_list = [item[1] for item in batch]

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

        # src_key_padding_mask: True at PAD positions
        padding_mask = torch.zeros(len(batch), max_seq_len, dtype=torch.bool)
        for i, l in enumerate(lengths):
            if l < max_seq_len:
                padding_mask[i, l:] = True

        return landmarks_tensor, labels, padding_mask



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
    def __init__(self, class_counts: torch.Tensor, label_smoothing=0.0):
        super().__init__()
        # class_counts is 1D tensor of size [C]
        priors = class_counts.float().clamp_min(1.0)
        self.log_priors = torch.log(priors / priors.sum())
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # Broadcast log_priors to batch; move to device
        log_priors = self.log_priors.to(logits.device)
        balanced_logits = logits + log_priors.unsqueeze(0)
        return F.cross_entropy(balanced_logits, targets, label_smoothing=self.label_smoothing)


class SignLoss(nn.Module):
    """
    Single-task loss for sign classification.
    Supports Focal Loss and Balanced Softmax.
    """
    def __init__(self, label_smoothing=0.0, use_focal=False, class_weights=None,
                 use_balanced_softmax=False, balanced_class_counts=None):
        super().__init__()

        # Set up sign classification loss
        if use_balanced_softmax and balanced_class_counts is not None:
            self.loss_fn = BalancedSoftmaxLoss(balanced_class_counts, label_smoothing=label_smoothing)
        elif use_focal:
            self.loss_fn = FocalLoss(alpha=0.25, gamma=2.0, weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                weight=class_weights
            )

    def forward(self, sign_logits, sign_labels):
        loss = self.loss_fn(sign_logits, sign_labels)
        return loss



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
    """Training with top-k accuracy metrics."""
    model.train()

    total_loss = 0.0
    topk_correct = {k: 0.0 for k in [1, 2, 3, 4, 5]}
    num_batches = 0
    total_samples = 0

    scaler = amp.GradScaler(enabled=(device.type == "cuda"))
    pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}] Train", leave=False)

    for batch in pbar:
        landmarks, sign_labels, padding_mask = batch
        landmarks = landmarks.to(device)
        sign_labels = sign_labels.to(device)
        padding_mask = padding_mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type='cuda', enabled=(device.type == "cuda")):
            sign_logits = model(landmarks, src_key_padding_mask=padding_mask)
            loss = criterion(sign_logits, sign_labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = sign_labels.size(0)
        total_loss += loss.item()

        topk_batch_accs = compute_topk_accuracy(sign_logits, sign_labels, k_values=[1, 2, 3, 4, 5])
        for k, acc in topk_batch_accs.items():
            topk_correct[k] += acc * batch_size

        total_samples += batch_size
        num_batches += 1
        
        pbar.set_postfix({
            'Loss': f'{total_loss/num_batches:.4f}',
            'Top1': f'{topk_correct[1]/total_samples:.4f}',
            'Top5': f'{topk_correct[5]/total_samples:.4f}'
        })

    avg_loss = total_loss / num_batches
    avg_topk_accs = {k: correct / total_samples for k, correct in topk_correct.items()}
    return avg_loss, avg_topk_accs



def validate_epoch(model, val_loader, criterion, device, idx_to_word):
    """Validation with top-k accuracy."""
    model.eval()

    total_loss = 0.0
    topk_correct = {k: 0.0 for k in [1, 2, 3, 4, 5]}
    num_batches = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="[Val]", leave=False)

        for batch in pbar:
            landmarks, sign_labels, padding_mask = batch
            landmarks = landmarks.to(device)
            sign_labels = sign_labels.to(device)
            padding_mask = padding_mask.to(device)

            sign_logits = model(landmarks, src_key_padding_mask=padding_mask)
            loss = criterion(sign_logits, sign_labels)
            total_loss += loss.item()

            batch_size = sign_labels.size(0)
            topk_batch_accs = compute_topk_accuracy(sign_logits, sign_labels, k_values=[1, 2, 3, 4, 5])
            for k, acc in topk_batch_accs.items():
                topk_correct[k] += acc * batch_size

            total_samples += batch_size

            sign_preds = torch.argmax(sign_logits, dim=1)
            all_preds.extend(sign_preds.cpu().numpy())
            all_labels.extend(sign_labels.cpu().numpy())

            num_batches += 1
            pbar.set_postfix({
                'Top1': f'{topk_correct[1]/total_samples:.4f}',
                'Top5': f'{topk_correct[5]/total_samples:.4f}'
            })

    avg_loss = total_loss / num_batches
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

    return avg_loss, avg_topk_accs, confusion_mat, class_metrics, all_preds, all_labels


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

    # mlflow.log_artifact(metrics_json_path)
    os.remove(metrics_json_path)

    print(f"✓ Logged per-class metrics for epoch {epoch}")

def load_data_by_type(args, base_dataset, top_k_words, old_to_new_idx,
                      use_enhanced_features, include_accel, include_bones,
                      include_bone_velocity, augment, augment_prob, train_class_counts,
                      use_skeletal_augmentation=True, skeletal_sigma=0.015, skeletal_probability=0.5,
                      use_occlusion_augmentation=True, occlusion_probability=0.3,
                      use_mirror_augmentation=True, mirror_probability=0.5):  # MIRROR AUGMENTATION
    """Load dataset according to structure type."""

    if args.dataset_type == 'flat':
        print(f"\n[STEP 2] Using flat structure with train_test_split")

        # Filter to vocabulary
        filtered_indices = []
        filtered_labels = []
        for i in range(len(base_dataset)):
            _, label = base_dataset[i]
            old_label = label.item()
            if old_label in old_to_new_idx:
                filtered_indices.append(i)
                word = base_dataset.idx_to_word[old_label]
                filtered_labels.append(word)

        print(f"  Filtered to {len(filtered_indices)} samples")

        # Split into train/val
        train_indices, val_indices = train_test_split(
            filtered_indices,
            test_size=0.2,
            random_state=42,
            stratify=filtered_labels
        )

        np.save('val_indices.npy', val_indices)
        mlflow.log_artifact("val_indices.npy")

        print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}")

        # Create properly configured datasets for flat type
        data_dir = args.data_dir
        
        dataset_train = SignLanguageDataset(
            data_dir,
            word_to_idx=base_dataset.word_to_idx,
            debug=False,
            augment=augment,
            augment_prob=augment_prob,
            use_enhanced_features=use_enhanced_features,
            include_accel=include_accel,
            include_bones=include_bones,
            include_bone_velocity=include_bone_velocity,
            class_counts=train_class_counts,
            use_skeletal_augmentation=use_skeletal_augmentation,
            skeletal_sigma=skeletal_sigma,
            skeletal_probability=skeletal_probability,
            use_occlusion_augmentation=use_occlusion_augmentation,
            occlusion_probability=occlusion_probability,
            use_mirror_augmentation=use_mirror_augmentation,
            mirror_probability=mirror_probability
        )

        dataset_val = SignLanguageDataset(
            data_dir,
            word_to_idx=base_dataset.word_to_idx,
            debug=False,
            augment=False,  # No augmentation for validation
            use_enhanced_features=use_enhanced_features,
            include_accel=include_accel,
            include_bones=include_bones,
            include_bone_velocity=include_bone_velocity,
            class_counts=train_class_counts,
            use_skeletal_augmentation=False,
            use_occlusion_augmentation=False,
            use_mirror_augmentation=False
        )

        return train_indices, val_indices, None, dataset_train, dataset_val, None

    else:  # 'split'
        print(f"\n[STEP 2] Using split structure (train/val/test folders)")

        train_dir = str(Path(args.data_dir) / "train")
        val_dir = str(Path(args.data_dir) / "val")
        test_dir = str(Path(args.data_dir) / "test")

        # Create separate datasets
        dataset_train = SignLanguageDataset(
            train_dir,
            word_to_idx=base_dataset.word_to_idx,
            debug=False,
            augment=augment,
            augment_prob=augment_prob,
            use_enhanced_features=use_enhanced_features,
            include_accel=include_accel,
            include_bones=include_bones,
            include_bone_velocity=include_bone_velocity,
            class_counts=train_class_counts,
            use_skeletal_augmentation=use_skeletal_augmentation,
            skeletal_sigma=skeletal_sigma,
            skeletal_probability=skeletal_probability,
            use_occlusion_augmentation=use_occlusion_augmentation,  # NEW
            occlusion_probability=occlusion_probability,             # NEW
            use_mirror_augmentation=use_mirror_augmentation,
            mirror_probability=mirror_probability
        )

        dataset_val = SignLanguageDataset(
            val_dir,
            word_to_idx=base_dataset.word_to_idx,
            debug=False,
            augment=False,
            use_enhanced_features=use_enhanced_features,
            include_accel=include_accel,
            include_bones=include_bones,
            include_bone_velocity=include_bone_velocity,
            class_counts=train_class_counts,
            use_skeletal_augmentation=False,
            use_occlusion_augmentation=False,  # Always False for val
            use_mirror_augmentation=False
        )

        dataset_test = SignLanguageDataset(
            test_dir,
            word_to_idx=base_dataset.word_to_idx,
            debug=False,
            augment=False,
            use_enhanced_features=use_enhanced_features,
            include_accel=include_accel,
            include_bones=include_bones,
            include_bone_velocity=include_bone_velocity,
            class_counts=train_class_counts,
            use_skeletal_augmentation=False,
            use_occlusion_augmentation=False,  # Always False for test
            use_mirror_augmentation=False
        )

        # Filter each to vocabulary
        train_indices = [i for i in range(len(dataset_train))
                        if dataset_train[i][1].item() in old_to_new_idx]
        val_indices = [i for i in range(len(dataset_val))
                      if dataset_val[i][1].item() in old_to_new_idx]
        test_indices = [i for i in range(len(dataset_test))
                       if dataset_test[i][1].item() in old_to_new_idx]

        print(f"  Train: {len(train_indices)} samples")
        print(f"  Val: {len(val_indices)} samples")
        print(f"  Test: {len(test_indices)} samples")

        return train_indices, val_indices, test_indices, dataset_train, dataset_val, dataset_test




def main():
    args = parse_args()

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

    dataset_name = Path(args.data_dir).name
    EXPERIMENT_NAME = "SignNetWord"
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ============================================================================
    # HYPERPARAMETERS (UPDATED)
    # ============================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MIN_SAMPLES_PER_CLASS = 70
    USE_CLASS_WEIGHTS = True
    USE_WEIGHTED_SAMPLER = True
    WEIGHT_BETA = 0.9999

    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4  # Reduced from 3e-4
    HIDDEN_SIZE = MAIN_MODEL_CONFIG['hidden_size']
    DROPOUT_RATE = 0.65  # Increased from 0.60 to reduce 7% train-val gap
    NUM_HEADS = MAIN_MODEL_CONFIG['num_heads']
    NUM_LAYERS = MAIN_MODEL_CONFIG['num_layers']
    ATTENTION_DROPOUT = 0.35  # Increased from 0.30 for stronger regularization
    WEIGHT_DECAY = 1e-2  # Keep strong L2 regularization
    AUGMENT = True
    AUGMENT_PROBABILITY = 0.75  # Increased from 0.7 for more augmentation

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

    # SKELETAL AUGMENTATION SETTINGS
    USE_SKELETAL_AUGMENTATION = False
    SKELETAL_SIGMA = 0.025          # Increased from 0.015 (more noise)
    SKELETAL_PROBABILITY = 0.7      # Increased from 0.5 (more samples augmented)

    # OCCLUSION AUGMENTATION SETTINGS (NEW)
    USE_OCCLUSION_AUGMENTATION = True
    OCCLUSION_PROBABILITY = 0.3  # 30% of samples get occlusion

    # MIRROR AUGMENTATION SETTINGS (NEW)
    USE_MIRROR_AUGMENTATION = True
    MIRROR_PROBABILITY = 0.3  # 30% of samples get horizontal flip (reduced from 0.5)

    # LR SCHEDULER SETTINGS
    USE_EARLY_STOPPING = False
    PLATEAU_PATIENCE = 5  # Adaptive reduction trigger
    WARMUP_EPOCHS = 20
    NUM_EPOCHS = 500
    BASE_LR = 1e-4
    MIN_LR = 1e-7         # Lower floor for long training
    T_0 = 100             # Longer first cycle for 4000 epochs
    T_MULT = 2            # Multiply cycle length after each restart

    # Check if we are in expert training mode
    if args.expert_name:
        print(f"\n{'='*80}")
        print(f"===== EXPERT TRAINING MODE: {args.expert_name} =====")
        print(f"{'='*80}\n")

        # Override hyperparameters for the smaller, focused model
        # FORCE SMALLER ARCHITECTURE: 128h/3L is too big. We need 64h/2L.    
        HIDDEN_SIZE = EXPERT_MODEL_CONFIG['hidden_size']
        NUM_LAYERS = EXPERT_MODEL_CONFIG['num_layers']
        NUM_HEADS = EXPERT_MODEL_CONFIG['num_heads']
        
        # === CRITICAL FIXES FOR SMALL DATASETS ===
        # 1. Stability: Lower LR and Batch Size to prevent overshooting
        BATCH_SIZE = 32          
        LEARNING_RATE = 3e-5     
        
        # 2. Disable "Big Data" Imbalance Tricks
        USE_BALANCED_SOFTMAX = False
        USE_FOCAL_LOSS = False
        USE_CLASS_WEIGHTS = False
        USE_WEIGHTED_SAMPLER = False
        
        # 3. Regularization - TUNED (Relaxed slightly to cure underfitting)
        DROPOUT_RATE = 0.5       # Reduced from 0.6 -> 0.5
        WEIGHT_DECAY = 0.02      # Reduced from 0.05 -> 0.02
        ATTENTION_DROPOUT = 0.3  # Reduced from 0.4 -> 0.3
        
        # 4. Training Duration & Augmentation
        NUM_EPOCHS = 120         
        WARMUP_EPOCHS = 5
        AUGMENT_PROBABILITY = 0.5 

    number_of_classes = 300

    NPZ_DIR = NPZ_DIR = args.data_dir
    MODEL_SAVE_DIR = "./models_balanced"
    PLOTS_DIR = "./plots_balanced"

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


    run_name = f"{dataset_name}_Transformer_{HIDDEN_SIZE}h_{NUM_LAYERS}L"
    if args.expert_name:
        run_name = f"expert_{args.expert_name}_{run_name}"
    try:
        run = mlflow.start_run(log_system_metrics=True, run_name=run_name)

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
                "use_enhanced_features": USE_ENHANCED_FEATURES,
                "include_acceleration": INCLUDE_ACCELERATION,
                "use_focal_loss": USE_FOCAL_LOSS,
                "attention_dropout": ATTENTION_DROPOUT,
                "use_skeletal_augmentation": USE_SKELETAL_AUGMENTATION,
                "skeletal_sigma": SKELETAL_SIGMA,
                "skeletal_probability": SKELETAL_PROBABILITY,
                "include_bones": INCLUDE_BONES,
                "include_bone_velocity": INCLUDE_BONE_VELOCITY,
                "use_occlusion_augmentation": USE_OCCLUSION_AUGMENTATION,  # NEW
                "occlusion_probability": OCCLUSION_PROBABILITY,            # NEW
            })

            # STEP 1: Load dataset WITH ENHANCED FEATURES
            print("\n" + "=" * 80)
            print(f"[STEP 1] Loading dataset from {args.data_dir} (type: {args.dataset_type})")
            print("=" * 80)

            # Determine data directory based on type
            if args.dataset_type == 'flat':
                data_dir_for_base = args.data_dir
            else:  # 'split'
                data_dir_for_base = str(Path(args.data_dir) / "train")

            print(f"  Loading base dataset from: {data_dir_for_base}")

            # Create base dataset to get vocabulary and determine input size
            base_dataset = SignLanguageDataset(
                data_dir_for_base,
                debug=False,
                augment=False,
                use_enhanced_features=USE_ENHANCED_FEATURES,
                include_accel=INCLUDE_ACCELERATION,
                include_bones=INCLUDE_BONES,
                include_bone_velocity=INCLUDE_BONE_VELOCITY
            )

            if len(base_dataset) == 0:
                raise ValueError(f"No .npz files found in {data_dir_for_base}. Please check your data directory.")

            # Get sample to determine input size
            sample_landmarks, _ = base_dataset[0]
            input_size = sample_landmarks.shape[-1]

            print(f"  Total samples in base: {len(base_dataset)}")
            print(f"  Input size: {input_size}")
            print(f"  Classes: {len(base_dataset.word_to_idx)}")
            mlflow.log_param("input_size", input_size)

            if args.dataset_type == 'flat':
                npz_files = sorted(Path(NPZ_DIR).glob("*.npz"))
            else:  # 'split'
                # Use train folder for vocabulary building
                npz_files = sorted((Path(NPZ_DIR) / "train").glob("*.npz"))

            # NOW the debug print
            print(f"\n[DEBUG] NPZ files check:")
            print(f"  NPZ_DIR: {NPZ_DIR}")
            print(f"  Number of npz_files: {len(npz_files)}")
            if len(npz_files) > 0:
                print(f"  First file: {npz_files[0]}")
                print(f"  Last file: {npz_files[-1]}")
            else:
                print(f"  ERROR: No npz_files found!")

            if args.expert_name:
                print(f"[VOCABULARY] Using pre-defined expert vocabulary for '{args.expert_name}'")
                top_k_words = HIERARCHY_CONFIG[args.expert_name]
                print(f"  Training on {len(top_k_words)} specific classes.")

                # We don't have 'all_counts' in this mode, so create a dummy one
                class_counts_analysis = Counter(word for f in npz_files for word in [Path(f).stem.split('_')[0]] if word in top_k_words)
                all_counts = class_counts_analysis

            else:
                # This is your original code for training the main model
                print("\n[VOCABULARY] Building vocabulary from dataset stats...")
                top_k_words, all_counts = build_topk_vocabulary(
                    npz_files,
                    K=number_of_classes,
                    min_samples=MIN_SAMPLES_PER_CLASS,
                    debug=True
                )

            if not top_k_words:
                print("[ERROR] Vocabulary is empty! Check MIN_SAMPLES_PER_CLASS or expert class names.")
                sys.exit(1)

            print(f"\n[VOCABULARY] Using {len(top_k_words)} classes after filtering.")
            mlflow.log_param("num_classes", len(top_k_words))

            # Compute class counts for class-aware augmentation
            train_class_counts = Counter()
            for npz_file in npz_files:
                try:
                    data = np.load(npz_file, allow_pickle=True)
                    gloss = data["glosses"][0]
                    if gloss in top_k_words:  # Only count classes in our vocabulary
                        train_class_counts[gloss] += 1
                except Exception:
                    continue

            # STEP 2: Filter to top words
            print(f"\n[STEP 2] Loading data (type: {args.dataset_type})...")

            old_to_new_idx = {}
            for new_idx, word in enumerate(sorted(top_k_words)):
                old_idx = base_dataset.word_to_idx[word]
                old_to_new_idx[old_idx] = new_idx

            new_idx_to_word = {
                new_idx: word
                for old_idx, new_idx in old_to_new_idx.items()
                for word in [base_dataset.idx_to_word[old_idx]]
            }

            # Build the forward and reverse mappings
            main_word_to_idx = {word: new_idx for new_idx, word in new_idx_to_word.items()}
            main_idx_to_word = new_idx_to_word  # Already have this

            # Create vocabulary dict
            vocab_dict = {
                'word_to_idx': main_word_to_idx,
                'idx_to_word': {int(k): v for k, v in main_idx_to_word.items()},
                'num_classes': len(top_k_words)
            }

            # Save to file
            if args.expert_name:
                vocab_filename = f'{args.expert_name}_vocab.json'
            else:
                vocab_filename = 'main_vocab.json'

            with open(vocab_filename, 'w') as f:
                json.dump(vocab_dict, f, indent=2)

            # Log to MLflow
            mlflow.log_artifact(vocab_filename)
            print(f"[VOCAB] Saved vocabulary to {vocab_filename} and logged to MLflow")

            # Clean up local file
            os.remove(vocab_filename)

            # Load data according to structure type
            train_indices, val_indices, test_indices, dataset_train, dataset_val, dataset_test = load_data_by_type(
                args=args,
                base_dataset=base_dataset,
                top_k_words=top_k_words,
                old_to_new_idx=old_to_new_idx,
                use_enhanced_features=USE_ENHANCED_FEATURES,
                include_accel=INCLUDE_ACCELERATION,
                include_bones=INCLUDE_BONES,
                include_bone_velocity=INCLUDE_BONE_VELOCITY,
                augment=AUGMENT,
                augment_prob=AUGMENT_PROBABILITY,
                train_class_counts=train_class_counts,
                use_skeletal_augmentation=USE_SKELETAL_AUGMENTATION,
                skeletal_sigma=SKELETAL_SIGMA,
                skeletal_probability=SKELETAL_PROBABILITY,
                use_occlusion_augmentation=USE_OCCLUSION_AUGMENTATION,
                occlusion_probability=OCCLUSION_PROBABILITY,
                use_mirror_augmentation=USE_MIRROR_AUGMENTATION,
                mirror_probability=MIRROR_PROBABILITY
            )

            num_classes = len(top_k_words)
            print(f"  Classes: {num_classes}")
            print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}")
            print(f"  Classes: {num_classes}")

            # STEP 4: Compute class weights
            if USE_CLASS_WEIGHTS:
                print(f"\n[STEP 4] Computing class weights...")

                train_class_counts = Counter()
                for idx in train_indices:
                    _, old_label = base_dataset[idx]
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
                    _, old_label = base_dataset[idx]
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
                    _, old_label = base_dataset[idx]
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
                    num_workers=2,
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
                    num_workers=2,
                    pin_memory=True,
                    prefetch_factor=4,
                    persistent_workers=True
                )

            val_loader = DataLoader(
                val_subset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=PadCollate(),
                num_workers=2,
                pin_memory=True,
                prefetch_factor=4,
                persistent_workers=True
            )

            # STEP 6: Build model
            print(f"\n[STEP 6] Building model...")


            model_raw = TransformerSignClassifier(
                input_size=input_size,
                hidden_size=HIDDEN_SIZE,
                num_classes=num_classes,
                num_layers=NUM_LAYERS,
                num_heads=NUM_HEADS,
                dim_feedforward=4 * HIDDEN_SIZE,
                dropout_rate=DROPOUT_RATE,
                attention_dropout=ATTENTION_DROPOUT,
                debug=True
            ).to(DEVICE)

            if hasattr(torch, 'compile'):
                print("Compiling model with torch.compile...")
                model = torch.compile(model_raw, mode='max-autotune-no-cudagraphs', dynamic=True)

            # STEP 7: Setup training
            print(f"\n[STEP 7] Setting up training...")

            criterion = SignLoss(
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

            # CHANGE: Use smooth decay for experts to avoid restart shocks
            if args.expert_name:
                restart_scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=NUM_EPOCHS - WARMUP_EPOCHS,
                    eta_min=MIN_LR
                )
            else:
                restart_scheduler = CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=T_0,              # First cycle: 25 epochs
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

                train_loss, train_topk_accs = train_epoch_interruptible(
                    model, train_loader, optimizer, criterion, DEVICE, epoch
                )

                if shutdown_handler.is_interrupted():
                    break

                val_loss, val_topk_accs, confusion_mat, class_metrics, all_preds, all_labels = validate_epoch(
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
                    # log_class_metrics_to_mlflow(class_metrics, epoch)

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
                swa_val_loss, swa_val_topk, *_ = validate_epoch(
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

            # CHANGE: Load the best weights before saving 'final' model
            best_model_path = os.path.join(MODEL_SAVE_DIR, "sign_classifier_best_enhanced.pth")
            if os.path.exists(best_model_path):
                print(f"Loading best model from epoch {best_epoch+1} for final save...")
                
                # Load state dict
                state_dict = torch.load(best_model_path)
                
                # Fix for torch.compile prefix '_orig_mod.'
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("_orig_mod."):
                        new_state_dict[k[10:]] = v  # Remove "_orig_mod."
                    else:
                        new_state_dict[k] = v
                
                model_raw.load_state_dict(new_state_dict)

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