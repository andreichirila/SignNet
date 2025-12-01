"""
Multi-Stream Feature Processor for Sign Language Recognition

Based on SL-GCN (SAM-SLR) concept:
- Joint Stream: Original landmark positions
- Bone Stream: Vectors between connected landmarks
- Joint Motion: Temporal difference of positions
- Bone Motion: Temporal difference of bone vectors

Author: Andrei Chirila, Roman Schläpfer
Date: 2025-12-01
"""

import numpy as np
import torch
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

# ============================================================================
# 🦴 SKELETON DEFINITIONS
# ============================================================================

# MediaPipe Landmark Structure:
# - Left Hand:  0-20   (21 landmarks)
# - Right Hand: 21-41  (21 landmarks)
# - Pose:       42-74  (33 landmarks)
# - Face:       75-542 (468 landmarks)

# Hand bone connections (same for left and right)
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
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm connections
    (5, 9), (9, 13), (13, 17)
]

# Pose bone connections (upper body focus)
POSE_BONES_LOCAL = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Upper body
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    # Hands
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    # Torso
    (11, 23), (12, 24), (23, 24),
]


@dataclass
class MultiStreamConfig:
    """Configuration for multi-stream processing."""
    num_left_hand: int = 21
    num_right_hand: int = 21
    num_pose: int = 33
    num_face: int = 468

    use_joint: bool = True
    use_bone: bool = True
    use_joint_motion: bool = True
    use_bone_motion: bool = True
    normalize_bones: bool = True

    @property
    def num_streams(self) -> int:
        return sum([self.use_joint, self.use_bone,
                    self.use_joint_motion, self.use_bone_motion])


class MultiStreamProcessor:
    """Process landmarks into multiple streams."""

    def __init__(self, config: Optional[MultiStreamConfig] = None):
        self.num_landmarks = (self.config.num_left_hand + self.config.num_right_hand +
                              self.config.num_pose + self.config.num_face)
        self.config = config or MultiStreamConfig()
        self._build_bone_connections()
        self.num_bones = len(self.all_bones)
        print(f"📊 MultiStreamProcessor: {self.config.num_streams} streams, {self.num_bones} bones")

    def _build_bone_connections(self):
        """Build global bone connection list."""
        self.all_bones = []

        # Left hand bones (0-20)
        for parent, child in HAND_BONES:
            self.all_bones.append((parent, child))

        # Right hand bones (21-41)
        for parent, child in HAND_BONES:
            self.all_bones.append((21 + parent, 21 + child))

        # Pose bones (42-74)
        for parent, child in POSE_BONES_LOCAL:
            self.all_bones.append((42 + parent, 42 + child))

        # Hand to wrist
        self.all_bones.append((0, 42 + 15))
        self.all_bones.append((21, 42 + 16))

        self.all_bones = np.array(self.all_bones)

    def compute_bones(self, joints: np.ndarray) -> np.ndarray:
        """Compute bone vectors: (T, num_bones, 2)"""
        T = joints.shape[0]
        bones = np.zeros((T, len(self.all_bones), 2), dtype=np.float32)

        for i, (parent, child) in enumerate(self.all_bones):
            if parent < joints.shape[1] and child < joints.shape[1]:
                bones[:, i, :] = joints[:, child, :] - joints[:, parent, :]

        if self.config.normalize_bones:
            lengths = np.linalg.norm(bones, axis=-1, keepdims=True)
            lengths = np.maximum(lengths, 1e-8)
            bones = bones / lengths

        return bones

    def compute_motion(self, features: np.ndarray) -> np.ndarray:
        """Compute temporal difference."""
        motion = np.zeros_like(features)
        if len(features) > 1:
            motion[1:] = features[1:] - features[:-1]
        return motion

    def process(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """Process landmarks into 4 streams."""
        streams = {}

        if self.config.use_joint:
            streams['joint'] = landmarks.copy()

        bones = self.compute_bones(landmarks)
        if self.config.use_bone:
            streams['bone'] = bones

        if self.config.use_joint_motion:
            streams['joint_motion'] = self.compute_motion(landmarks)

        if self.config.use_bone_motion:
            streams['bone_motion'] = self.compute_motion(bones)

        return streams


if __name__ == "__main__":
    # Test
    landmarks = np.random.randn(100, 543, 2).astype(np.float32)
    processor = MultiStreamProcessor()
    streams = processor.process(landmarks)

    for name, arr in streams.items():
        print(f"{name}: {arr.shape}")