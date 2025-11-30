# Dataset.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import random
from dataclasses import dataclass

from Config import Config, DataConfig


# ============================================
# VOCABULARY & LABEL MAPPING
# ============================================

class Vocabulary:
    """Vocabulary for gloss-to-index mapping."""

    def __init__(self, top_k_glosses: Optional[List[str]] = None):
        """
        Initialize vocabulary.

        Args:
            top_k_glosses: List of glosses to include (if None, all glosses)
        """
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.BLANK_TOKEN = '<BLANK>'  # For CTC

        # Initialize mappings
        self.gloss2idx = {
            self.PAD_TOKEN: 0,
            self.BLANK_TOKEN: 1,
            self.UNK_TOKEN: 2,
        }
        self.idx2gloss = {
            0: self.PAD_TOKEN,
            1: self.BLANK_TOKEN,
            2: self.UNK_TOKEN,
        }

        # Add glosses
        if top_k_glosses:
            for gloss in top_k_glosses:
                if gloss not in self.gloss2idx:
                    idx = len(self.gloss2idx)
                    self.gloss2idx[gloss] = idx
                    self.idx2gloss[idx] = gloss

    @classmethod
    def from_top_k_file(cls, filepath: str, top_k: int):
        """
        Create vocabulary from top-K glosses file.

        Args:
            filepath: Path to gloss_frequencies.csv
            top_k: Number of top glosses to use
        """
        df = pd.read_csv(filepath)
        top_glosses = df.head(top_k)['gloss'].tolist()
        return cls(top_k_glosses=top_glosses)

    def encode(self, glosses: List[str]) -> List[int]:
        """Convert glosses to indices."""
        return [self.gloss2idx.get(g, self.gloss2idx[self.UNK_TOKEN]) for g in glosses]

    def decode(self, indices: List[int]) -> List[str]:
        """Convert indices to glosses."""
        return [self.idx2gloss.get(i, self.UNK_TOKEN) for i in indices]

    def __len__(self):
        return len(self.gloss2idx)

    @property
    def vocab_size(self):
        return len(self.gloss2idx)

    @property
    def pad_idx(self):
        return self.gloss2idx[self.PAD_TOKEN]

    @property
    def blank_idx(self):
        return self.gloss2idx[self.BLANK_TOKEN]

    @property
    def unk_idx(self):
        return self.gloss2idx[self.UNK_TOKEN]


# ============================================
# AUGMENTATION FUNCTIONS
# ============================================

class LandmarkAugmenter:
    """Augmentation for landmark data."""

    def __init__(self, config: DataConfig):
        self.config = config

    def apply_spatial_augmentation(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply spatial augmentation (rotation, scale, translation).

        Args:
            landmarks: [T, N, 2] array (N=543 landmarks)

        Returns:
            Augmented landmarks [T, N, 2]
        """
        T, N, C = landmarks.shape
        assert C == 2, "Expected 2D landmarks (x, y)"

        # Random rotation
        if self.config.rotation_range > 0:
            angle = np.random.uniform(-self.config.rotation_range,
                                      self.config.rotation_range)
            angle_rad = np.radians(angle)

            # Rotation matrix
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            # Center around origin (0.5, 0.5)
            centered = landmarks - 0.5

            # Apply rotation
            x_rot = centered[:, :, 0] * cos_a - centered[:, :, 1] * sin_a
            y_rot = centered[:, :, 0] * sin_a + centered[:, :, 1] * cos_a

            landmarks = np.stack([x_rot, y_rot], axis=-1) + 0.5

        # Random scaling
        if self.config.scale_range:
            scale = np.random.uniform(*self.config.scale_range)
            centered = landmarks - 0.5
            landmarks = centered * scale + 0.5

        # Random translation
        if self.config.translation_range > 0:
            tx = np.random.uniform(-self.config.translation_range,
                                   self.config.translation_range)
            ty = np.random.uniform(-self.config.translation_range,
                                   self.config.translation_range)
            landmarks[:, :, 0] += tx
            landmarks[:, :, 1] += ty

        # Clip to valid range [0, 1]
        landmarks = np.clip(landmarks, 0, 1)

        return landmarks

    def apply_occlusion(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply landmark occlusion augmentation.
        PRIORITY: From professor's list!

        Args:
            landmarks: [T, N, 2] array

        Returns:
            Augmented landmarks with occlusions
        """
        T, N, C = landmarks.shape

        # Overall occlusion probability
        if np.random.random() > self.config.occlusion_prob:
            return landmarks  # No occlusion

        # Determine which group to occlude
        groups = []
        probs = []

        if self.config.left_hand_occlusion_prob > 0:
            groups.append(('left_hand', self.config.left_hand_indices))
            probs.append(self.config.left_hand_occlusion_prob)

        if self.config.right_hand_occlusion_prob > 0:
            groups.append(('right_hand', self.config.right_hand_indices))
            probs.append(self.config.right_hand_occlusion_prob)

        if self.config.face_occlusion_prob > 0:
            groups.append(('face', self.config.pose_face_indices[33:]))  # Face only
            probs.append(self.config.face_occlusion_prob)

        if not groups:
            return landmarks

        # Normalize probabilities
        probs = np.array(probs)
        probs = probs / probs.sum()

        # Select group
        group_idx = np.random.choice(len(groups), p=probs)
        group_name, landmark_indices = groups[group_idx]

        # Random occlusion duration
        duration = np.random.randint(*self.config.occlusion_duration_range)
        duration = min(duration, T)  # Don't exceed sequence length

        # Random start frame
        max_start = max(0, T - duration)
        start_frame = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        end_frame = start_frame + duration

        # Apply occlusion (set to 0)
        landmarks_aug = landmarks.copy()
        landmarks_aug[start_frame:end_frame, landmark_indices, :] = 0.0

        return landmarks_aug

    def apply_temporal_augmentation(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply temporal augmentation (frame dropping).

        Args:
            landmarks: [T, N, 2] array

        Returns:
            Augmented landmarks (may have different T)
        """
        T, N, C = landmarks.shape

        if np.random.random() > self.config.temporal_dropout_prob:
            return landmarks

        # Random drop ratio
        keep_ratio = np.random.uniform(*self.config.frame_drop_range)
        num_keep = max(1, int(T * keep_ratio))

        # Randomly select frames to keep
        keep_indices = sorted(np.random.choice(T, num_keep, replace=False))

        return landmarks[keep_indices]

    def augment(self, landmarks: np.ndarray, apply_temporal: bool = True) -> np.ndarray:
        """
        Apply all augmentations.

        Args:
            landmarks: [T, N, 2] array
            apply_temporal: Whether to apply temporal augmentation

        Returns:
            Fully augmented landmarks
        """
        # Spatial augmentation
        landmarks = self.apply_spatial_augmentation(landmarks)

        # Occlusion augmentation (HIGH PRIORITY)
        landmarks = self.apply_occlusion(landmarks)

        # Temporal augmentation (optional, can change sequence length)
        if apply_temporal:
            landmarks = self.apply_temporal_augmentation(landmarks)

        return landmarks


# ============================================
# DATASET CLASS
# ============================================

class SignLanguageDataset(Dataset):
    """RWTH-PHOENIX-2014 Dataset with Top-K filtering."""

    def __init__(
            self,
            data_dir: str,
            vocabulary: Vocabulary,
            config: DataConfig,
            split: str = 'train',
            augment: bool = True
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory with .npz files
            vocabulary: Vocabulary object for gloss mapping
            config: Data configuration
            split: 'train', 'dev', or 'test'
            augment: Whether to apply augmentation
        """
        self.data_dir = Path(data_dir)
        self.vocabulary = vocabulary
        self.config = config
        self.split = split
        self.augment = augment and config.use_augmentation and split == 'train'

        # Initialize augmenter
        self.augmenter = LandmarkAugmenter(config) if self.augment else None

        # Load samples
        self.samples = self._load_samples()

        print(f"✅ {split.upper()} Dataset loaded: {len(self.samples)} samples")

    def _load_samples(self) -> List[Dict]:
        """Load all valid samples from directory."""
        samples = []
        skipped_no_valid_gloss = 0
        total_glosses_converted = 0
        total_unk_glosses = 0

        npz_files = list(self.data_dir.glob("*.npz"))

        for npz_file in npz_files:
            try:
                data = np.load(npz_file, allow_pickle=True)
                landmarks = data['landmarks']
                glosses = data['glosses']

                # Convert glosses to list
                if isinstance(glosses, np.ndarray):
                    gloss_list = glosses.tolist()
                else:
                    gloss_list = [str(glosses)]

                # NEW LOGIC: Keep samples that have AT LEAST ONE known gloss
                # Unknown glosses will be mapped to UNK token
                if self.config.use_top_k_classes:
                    # Check if at least one gloss is in vocabulary
                    known_glosses = [g for g in gloss_list if g in self.vocabulary.gloss2idx]

                    if len(known_glosses) == 0:
                        # Skip samples with NO known glosses
                        skipped_no_valid_gloss += 1
                        continue

                    # Count UNK glosses for statistics
                    unk_count = len(gloss_list) - len(known_glosses)
                    total_unk_glosses += unk_count
                    total_glosses_converted += len(gloss_list)

                samples.append({
                    'file': npz_file.name,
                    'landmarks': landmarks,
                    'glosses': gloss_list,
                    'num_frames': landmarks.shape[0],
                    'num_glosses': len(gloss_list),
                })

            except Exception as e:
                print(f"⚠️  Error loading {npz_file.name}: {e}")
                continue

        # Print statistics
        if self.config.use_top_k_classes and total_glosses_converted > 0:
            unk_pct = total_unk_glosses / total_glosses_converted * 100
            print(f"   📊 {self.split}: {len(samples)} samples, "
                  f"{skipped_no_valid_gloss} skipped (no known glosses), "
                  f"{unk_pct:.1f}% UNK glosses")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            dict with:
                - landmarks: [T, N, 2] tensor
                - glosses: [G] tensor (gloss indices)
                - num_frames: int
                - num_glosses: int
        """
        sample = self.samples[idx]

        # Get landmarks: [T, 1086] → reshape to [T, 543, 2]
        landmarks_flat = sample['landmarks']
        T = landmarks_flat.shape[0]
        landmarks = landmarks_flat.reshape(T, 543, 2)

        # Apply augmentation
        if self.augment:
            landmarks = self.augmenter.augment(landmarks, apply_temporal=True)
            # Note: T may have changed due to temporal augmentation
            T = landmarks.shape[0]

        # Encode glosses
        gloss_indices = self.vocabulary.encode(sample['glosses'])

        # Convert to tensors
        landmarks_tensor = torch.FloatTensor(landmarks)  # [T, 543, 2]
        glosses_tensor = torch.LongTensor(gloss_indices)  # [G]

        return {
            'landmarks': landmarks_tensor,
            'glosses': glosses_tensor,
            'num_frames': T,
            'num_glosses': len(gloss_indices),
            'file': sample['file'],
        }


# ============================================
# COLLATE FUNCTION (for batching)
# ============================================

def collate_fn(batch: List[Dict], max_frames: int = 214) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    Pads sequences to max_frames.

    Args:
        batch: List of samples from __getitem__
        max_frames: Maximum sequence length (from config)

    Returns:
        Batched dict with:
            - landmarks: [B, T_max, 543, 2]
            - glosses: [B, G_max]
            - landmarks_lengths: [B]
            - glosses_lengths: [B]
            - files: List[str]
    """
    batch_size = len(batch)

    # Find max lengths in this batch
    max_T = min(max(s['num_frames'] for s in batch), max_frames)
    max_G = max(s['num_glosses'] for s in batch)

    # Initialize padded tensors
    landmarks_padded = torch.zeros(batch_size, max_T, 543, 2)
    glosses_padded = torch.zeros(batch_size, max_G, dtype=torch.long)

    landmarks_lengths = []
    glosses_lengths = []
    files = []

    for i, sample in enumerate(batch):
        # Get actual lengths
        T = min(sample['num_frames'], max_frames)
        G = sample['num_glosses']

        # Truncate if too long
        landmarks = sample['landmarks'][:T]
        glosses = sample['glosses'][:G]

        # Store in padded tensor
        landmarks_padded[i, :T] = landmarks
        glosses_padded[i, :G] = glosses

        landmarks_lengths.append(T)
        glosses_lengths.append(G)
        files.append(sample['file'])

    return {
        'landmarks': landmarks_padded,  # [B, T, 543, 2]
        'glosses': glosses_padded,  # [B, G]
        'landmarks_lengths': torch.LongTensor(landmarks_lengths),  # [B]
        'glosses_lengths': torch.LongTensor(glosses_lengths),  # [B]
        'files': files,
    }


# ============================================
# DATALOADER CREATION
# ============================================

def create_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary]:
    """
    Create train, dev, test dataloaders and vocabulary.

    Args:
        config: Configuration object

    Returns:
        train_loader, dev_loader, test_loader, vocabulary
    """
    print("\n" + "=" * 80)
    print("CREATING DATALOADERS")
    print("=" * 80)

    # Create vocabulary from Top-K glosses
    print(f"\n📖 Creating vocabulary (Top-{config.data.use_top_k_classes})...")
    vocabulary = Vocabulary.from_top_k_file(
        filepath=config.data.top_k_glosses_file,
        top_k=config.data.use_top_k_classes
    )
    print(f"   Vocabulary size: {vocabulary.vocab_size}")
    print(f"   PAD idx: {vocabulary.pad_idx}")
    print(f"   BLANK idx: {vocabulary.blank_idx}")
    print(f"   UNK idx: {vocabulary.unk_idx}")

    # Create datasets
    print("\n📊 Creating datasets...")

    train_dataset = SignLanguageDataset(
        data_dir=config.data.train_dir,
        vocabulary=vocabulary,
        config=config.data,
        split='train',
        augment=True
    )

    dev_dataset = SignLanguageDataset(
        data_dir=config.data.dev_dir,
        vocabulary=vocabulary,
        config=config.data,
        split='dev',
        augment=False
    )

    test_dataset = SignLanguageDataset(
        data_dir=config.data.test_dir,
        vocabulary=vocabulary,
        config=config.data,
        split='test',
        augment=False
    )

    # Create dataloaders
    print("\n🔄 Creating dataloaders...")

    def get_collate_fn(max_frames):
        return lambda batch: collate_fn(batch, max_frames=max_frames)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=get_collate_fn(config.data.max_frames),
        pin_memory=True if config.model.device == 'cuda' else False,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=get_collate_fn(config.data.max_frames),
        pin_memory=True if config.model.device == 'cuda' else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=get_collate_fn(config.data.max_frames),
        pin_memory=True if config.model.device == 'cuda' else False,
    )

    print(f"\n✅ Dataloaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Dev batches: {len(dev_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    print("\n" + "=" * 80)

    return train_loader, dev_loader, test_loader, vocabulary


# ============================================
# TEST FUNCTION
# ============================================

if __name__ == "__main__":
    # Test dataset loading
    from Config import get_config

    print("\n" + "=" * 80)
    print("TESTING DATASET.PY")
    print("=" * 80)

    # Get config
    config = get_config(top_k=50, use_augmentation=True)

    # Create dataloaders
    train_loader, dev_loader, test_loader, vocab = create_dataloaders(config)

    # Test loading one batch
    print("\n🧪 Testing batch loading...")
    batch = next(iter(train_loader))

    print(f"\n📦 Batch contents:")
    print(f"   Landmarks shape: {batch['landmarks'].shape}")
    print(f"   Glosses shape: {batch['glosses'].shape}")
    print(f"   Landmarks lengths: {batch['landmarks_lengths']}")
    print(f"   Glosses lengths: {batch['glosses_lengths']}")
    print(f"   Files: {batch['files'][:3]}...")

    # Decode some glosses
    print(f"\n📝 Sample glosses:")
    for i in range(min(3, len(batch['glosses']))):
        indices = batch['glosses'][i][:batch['glosses_lengths'][i]].tolist()
        glosses = vocab.decode(indices)
        print(f"   Sample {i}: {glosses}")

    print("\n✅ Dataset.py test complete!")