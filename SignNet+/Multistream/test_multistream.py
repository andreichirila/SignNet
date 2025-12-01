"""
🧪 Unit Tests for Multi-Stream Components

Tests:
- MultiStreamProcessor: Bone computation, motion computation
- MultiStreamModel: Forward pass, shapes
- MultiStreamDataset: Data loading, augmentation

Run: python -m pytest test_multistream.py -v
Or:  python test_multistream.py

Author: Andrei Chirila, Roman Schläpfer
Date: 2025-12-01
"""

import sys
import numpy as np
import torch
import tempfile
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from MultiStream import MultiStreamProcessor
from MultiStreamModel import MultiStreamSignLanguageTransformer, MultiStreamModelConfig
from MultiStreamDataset import MultiStreamDataset, DatasetConfig, collate_fn


# ============================================================================
# 🧪 TEST: MultiStreamProcessor
# ============================================================================

class TestMultiStreamProcessor:
    """Tests for stream computation."""

    def test_init(self):
        """Test processor initialization."""
        processor = MultiStreamProcessor()
        assert processor.num_landmarks == 543
        assert processor.num_bones > 0  # Should have bones defined
        print("✅ MultiStreamProcessor init: OK")

    def test_compute_bones_shape(self):
        """Test bone computation output shape."""
        processor = MultiStreamProcessor()

        # Create dummy landmarks: (T, 543, 2)
        T = 50
        landmarks = np.random.randn(T, 543, 2).astype(np.float32)

        bones = processor.compute_bones(landmarks)

        assert bones.shape == (T, processor.num_bones, 2), \
            f"Expected ({T}, {processor.num_bones}, 2), got {bones.shape}"
        print(f"✅ compute_bones shape: {bones.shape}")

    def test_compute_bones_values(self):
        """Test bone computation correctness."""
        processor = MultiStreamProcessor()

        # Create simple landmarks
        T = 10
        landmarks = np.zeros((T, 543, 2), dtype=np.float32)

        # Set specific values to test bone computation
        # Bone 0: left hand tip (4) - joint (3)
        landmarks[:, 4, :] = [1.0, 0.0]  # tip
        landmarks[:, 3, :] = [0.0, 0.0]  # joint

        bones = processor.compute_bones(landmarks)

        # First bone should be [1.0, 0.0] (normalized)
        expected = np.array([1.0, 0.0])
        np.testing.assert_array_almost_equal(bones[0, 0], expected, decimal=5)
        print("✅ compute_bones values: OK")

    def test_compute_motion_shape(self):
        """Test motion computation output shape."""
        processor = MultiStreamProcessor()

        T = 50
        data = np.random.randn(T, 543, 2).astype(np.float32)

        motion = processor.compute_motion(data)

        assert motion.shape == data.shape, \
            f"Expected {data.shape}, got {motion.shape}"
        print(f"✅ compute_motion shape: {motion.shape}")

    def test_compute_motion_values(self):
        """Test motion computation correctness."""
        processor = MultiStreamProcessor()

        T = 5
        data = np.zeros((T, 10, 2), dtype=np.float32)

        # Frame 0: [0, 0], Frame 1: [1, 0], Frame 2: [2, 0], ...
        for t in range(T):
            data[t, 0, 0] = t

        motion = processor.compute_motion(data)

        # First frame motion should be 0
        assert motion[0, 0, 0] == 0.0, "First frame motion should be 0"

        # Other frames should have motion of 1
        for t in range(1, T):
            assert motion[t, 0, 0] == 1.0, f"Frame {t} motion should be 1"

        print("✅ compute_motion values: OK")

    def test_process_all_streams(self):
        """Test full processing pipeline."""
        processor = MultiStreamProcessor()

        T = 30
        landmarks = np.random.randn(T, 543, 2).astype(np.float32)

        streams = processor.process(landmarks)

        assert 'joint' in streams
        assert 'bone' in streams
        assert 'joint_motion' in streams
        assert 'bone_motion' in streams

        assert streams['joint'].shape == (T, 543, 2)
        assert streams['bone'].shape == (T, processor.num_bones, 2)
        assert streams['joint_motion'].shape == (T, 543, 2)
        assert streams['bone_motion'].shape == (T, processor.num_bones, 2)

        print("✅ process all streams: OK")
        print(f"   Joint: {streams['joint'].shape}")
        print(f"   Bone: {streams['bone'].shape}")
        print(f"   Joint Motion: {streams['joint_motion'].shape}")
        print(f"   Bone Motion: {streams['bone_motion'].shape}")


# ============================================================================
# 🧪 TEST: MultiStreamModel
# ============================================================================

class TestMultiStreamModel:
    """Tests for model architecture."""

    def test_model_init(self):
        """Test model initialization."""
        config = MultiStreamModelConfig(
            num_landmarks=543,
            landmark_dim=2,
            num_bones=70,
            stream_hidden_dim=128,
            d_model=256,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            dropout=0.1,
            num_classes=100,
            fusion_type='attention',
            device='cpu'
        )

        model = MultiStreamSignLanguageTransformer(config)

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"✅ Model init: OK")
        print(f"   Total params: {num_params:,}")
        print(f"   Trainable: {trainable:,}")

    def test_forward_pass(self):
        """Test forward pass with dummy data."""
        config = MultiStreamModelConfig(
            num_landmarks=543,
            landmark_dim=2,
            num_bones=70,
            stream_hidden_dim=128,
            d_model=256,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            dropout=0.1,
            num_classes=100,
            fusion_type='attention',
            device='cpu'
        )

        model = MultiStreamSignLanguageTransformer(config)
        model.eval()

        B, T = 4, 50

        streams = {
            'joint': torch.randn(B, T, 543, 2),
            'bone': torch.randn(B, T, 70, 2),
            'joint_motion': torch.randn(B, T, 543, 2),
            'bone_motion': torch.randn(B, T, 70, 2),
        }
        lengths = torch.tensor([50, 40, 30, 20])

        with torch.no_grad():
            log_probs, out_lens = model(streams, lengths)

        assert log_probs.dim() == 3, "Output should be 3D (B, T, C)"
        assert log_probs.shape[0] == B
        assert log_probs.shape[2] == 100  # num_classes

        # Check log probabilities sum to ~1 (after exp)
        probs = torch.exp(log_probs[0, 0])
        assert abs(probs.sum().item() - 1.0) < 0.01, "Probabilities should sum to 1"

        print(f"✅ Forward pass: OK")
        print(f"   Input: {B}x{T} sequences")
        print(f"   Output: {log_probs.shape}")
        print(f"   Output lengths: {out_lens.tolist()}")

    def test_fusion_types(self):
        """Test different fusion strategies."""
        fusion_types = ['concat', 'attention', 'gated', 'weighted']

        for fusion in fusion_types:
            config = MultiStreamModelConfig(
                num_landmarks=543,
                landmark_dim=2,
                num_bones=70,
                stream_hidden_dim=128,
                d_model=256,
                n_heads=4,
                n_layers=2,
                d_ff=512,
                dropout=0.1,
                num_classes=100,
                fusion_type=fusion,
                device='cpu'
            )

            model = MultiStreamSignLanguageTransformer(config)
            model.eval()

            B, T = 2, 30
            streams = {
                'joint': torch.randn(B, T, 543, 2),
                'bone': torch.randn(B, T, 70, 2),
                'joint_motion': torch.randn(B, T, 543, 2),
                'bone_motion': torch.randn(B, T, 70, 2),
            }
            lengths = torch.tensor([30, 20])

            with torch.no_grad():
                log_probs, _ = model(streams, lengths)

            assert log_probs.shape == (B, T, 100)
            print(f"✅ Fusion '{fusion}': OK")

    def test_variable_lengths(self):
        """Test handling of variable sequence lengths."""
        config = MultiStreamModelConfig(
            num_landmarks=543,
            landmark_dim=2,
            num_bones=70,
            stream_hidden_dim=128,
            d_model=256,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            dropout=0.1,
            num_classes=100,
            fusion_type='attention',
            device='cpu'
        )

        model = MultiStreamSignLanguageTransformer(config)
        model.eval()

        # Different batch sizes and lengths
        test_cases = [
            (1, 10),   # Single short sequence
            (1, 100),  # Single long sequence
            (8, 50),   # Batch of medium sequences
        ]

        for B, T in test_cases:
            streams = {
                'joint': torch.randn(B, T, 543, 2),
                'bone': torch.randn(B, T, 70, 2),
                'joint_motion': torch.randn(B, T, 543, 2),
                'bone_motion': torch.randn(B, T, 70, 2),
            }
            lengths = torch.randint(T//2, T+1, (B,))

            with torch.no_grad():
                log_probs, out_lens = model(streams, lengths)

            assert log_probs.shape[0] == B
            print(f"✅ Variable lengths B={B}, T={T}: OK")


# ============================================================================
# 🧪 TEST: MultiStreamDataset
# ============================================================================

class TestMultiStreamDataset:
    """Tests for dataset."""

    def create_dummy_data(self, tmp_dir: Path, num_samples: int = 5):
        """Create dummy NPZ files for testing."""
        samples = []

        for i in range(num_samples):
            # Random sequence length
            T = np.random.randint(20, 60)

            # Create dummy landmarks
            landmarks = np.random.randn(T, 543, 2).astype(np.float32)

            # Save NPZ
            npz_path = tmp_dir / f"sample_{i:03d}.npz"
            np.savez(npz_path, landmarks=landmarks)

            # Save annotation
            glosses = ['GLOSS_A', 'GLOSS_B', 'GLOSS_C'][:np.random.randint(1, 4)]
            ann_path = tmp_dir / f"sample_{i:03d}.json"
            with open(ann_path, 'w') as f:
                json.dump({'glosses': glosses}, f)

            samples.append({
                'features': str(npz_path),
                'glosses': glosses
            })

        return samples

    def create_dummy_vocab(self):
        """Create dummy vocabulary."""
        class DummyVocab:
            def __init__(self):
                self.gloss2idx = {
                    '<PAD>': 0, '<BLANK>': 1, '<UNK>': 2,
                    'GLOSS_A': 3, 'GLOSS_B': 4, 'GLOSS_C': 5
                }
                self.idx2gloss = {v: k for k, v in self.gloss2idx.items()}

            @property
            def vocab_size(self):
                return len(self.gloss2idx)

        return DummyVocab()

    def test_dataset_init(self):
        """Test dataset initialization."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            samples = self.create_dummy_data(tmp_path)
            vocab = self.create_dummy_vocab()

            config = DatasetConfig(augment=False)
            dataset = MultiStreamDataset(samples, vocab, config, is_train=False)

            assert len(dataset) == len(samples)
            print(f"✅ Dataset init: {len(dataset)} samples")

    def test_dataset_getitem(self):
        """Test single item retrieval."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            samples = self.create_dummy_data(tmp_path)
            vocab = self.create_dummy_vocab()

            config = DatasetConfig(augment=False)
            dataset = MultiStreamDataset(samples, vocab, config, is_train=False)

            item = dataset[0]

            assert 'joint' in item
            assert 'bone' in item
            assert 'joint_motion' in item
            assert 'bone_motion' in item
            assert 'labels' in item
            assert 'length' in item

            print(f"✅ Dataset getitem: OK")
            print(f"   Joint shape: {item['joint'].shape}")
            print(f"   Bone shape: {item['bone'].shape}")
            print(f"   Labels: {item['labels']}")

    def test_collate_fn(self):
        """Test batch collation with padding."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            samples = self.create_dummy_data(tmp_path, num_samples=4)
            vocab = self.create_dummy_vocab()

            config = DatasetConfig(augment=False)
            dataset = MultiStreamDataset(samples, vocab, config, is_train=False)

            # Get multiple items
            items = [dataset[i] for i in range(4)]

            # Collate
            batch = collate_fn(items)

            assert batch['joint'].dim() == 4  # (B, T, landmarks, 2)
            assert batch['bone'].dim() == 4
            assert batch['lengths'].dim() == 1
            assert batch['labels'].dim() == 2

            print(f"✅ Collate function: OK")
            print(f"   Batch joint: {batch['joint'].shape}")
            print(f"   Batch bone: {batch['bone'].shape}")
            print(f"   Lengths: {batch['lengths'].tolist()}")

    def test_augmentation(self):
        """Test data augmentation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            samples = self.create_dummy_data(tmp_path, num_samples=1)
            vocab = self.create_dummy_vocab()

            # With augmentation
            config = DatasetConfig(
                augment=True,
                rotation_range=30.0,
                scale_range=(0.8, 1.2),
                temporal_dropout_prob=0.2
            )
            dataset = MultiStreamDataset(samples, vocab, config, is_train=True)

            # Get same item multiple times
            items = [dataset[0] for _ in range(5)]

            # Check that augmentation produces different results
            joints = [item['joint'].numpy() for item in items]

            # At least some should be different (due to random augmentation)
            different = False
            for i in range(1, len(joints)):
                if not np.allclose(joints[0], joints[i]):
                    different = True
                    break

            assert different, "Augmentation should produce different results"
            print("✅ Augmentation: OK (produces variation)")

    def test_dataloader(self):
        """Test full DataLoader integration."""
        from torch.utils.data import DataLoader

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            samples = self.create_dummy_data(tmp_path, num_samples=8)
            vocab = self.create_dummy_vocab()

            config = DatasetConfig(augment=False)
            dataset = MultiStreamDataset(samples, vocab, config, is_train=False)

            loader = DataLoader(
                dataset, batch_size=4, shuffle=True,
                num_workers=0, collate_fn=collate_fn
            )

            for batch in loader:
                assert batch['joint'].shape[0] <= 4  # batch size
                break

            print("✅ DataLoader integration: OK")


# ============================================================================
# 🧪 TEST: End-to-End Pipeline
# ============================================================================

class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline(self):
        """Test complete pipeline from data to loss."""
        print("\n" + "="*60)
        print("🔄 End-to-End Pipeline Test")
        print("="*60)

        # 1. Create processor
        processor = MultiStreamProcessor()
        print("✅ 1. Processor created")

        # 2. Generate dummy landmarks
        B, T = 4, 40
        landmarks_batch = np.random.randn(B, T, 543, 2).astype(np.float32)
        print(f"✅ 2. Dummy data: {landmarks_batch.shape}")

        # 3. Process streams
        streams_list = []
        for i in range(B):
            streams = processor.process(landmarks_batch[i])
            streams_list.append(streams)

        # Stack into batches
        streams_batch = {
            'joint': torch.tensor(np.stack([s['joint'] for s in streams_list])),
            'bone': torch.tensor(np.stack([s['bone'] for s in streams_list])),
            'joint_motion': torch.tensor(np.stack([s['joint_motion'] for s in streams_list])),
            'bone_motion': torch.tensor(np.stack([s['bone_motion'] for s in streams_list])),
        }
        print(f"✅ 3. Streams processed")

        # 4. Create model
        config = MultiStreamModelConfig(
            num_landmarks=543,
            landmark_dim=2,
            num_bones=processor.num_bones,
            stream_hidden_dim=128,
            d_model=256,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            dropout=0.1,
            num_classes=50,
            fusion_type='attention',
            device='cpu'
        )
        model = MultiStreamSignLanguageTransformer(config)
        print(f"✅ 4. Model created")

        # 5. Forward pass
        lengths = torch.tensor([40, 35, 30, 25])
        log_probs, out_lens = model(streams_batch, lengths)
        print(f"✅ 5. Forward pass: {log_probs.shape}")

        # 6. Compute CTC loss
        labels = torch.randint(3, 50, (B, 5))  # Random labels
        label_lengths = torch.tensor([5, 4, 3, 2])

        ctc_loss = torch.nn.CTCLoss(blank=1, zero_infinity=True)
        loss = ctc_loss(
            log_probs.transpose(0, 1),  # (T, B, C)
            labels,
            out_lens,
            label_lengths
        )
        print(f"✅ 6. CTC Loss: {loss.item():.4f}")

        # 7. Backward pass
        loss.backward()
        print(f"✅ 7. Backward pass: OK")

        # Check gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "Should have gradients after backward"
        print(f"✅ 8. Gradients computed")

        print("\n🎉 End-to-End Pipeline: ALL TESTS PASSED!")


# ============================================================================
# 🚀 MAIN
# ============================================================================

def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("🧪 Multi-Stream Unit Tests")
    print("="*60)

    # Processor tests
    print("\n📦 Testing MultiStreamProcessor...")
    test_processor = TestMultiStreamProcessor()
    test_processor.test_init()
    test_processor.test_compute_bones_shape()
    test_processor.test_compute_bones_values()
    test_processor.test_compute_motion_shape()
    test_processor.test_compute_motion_values()
    test_processor.test_process_all_streams()

    # Model tests
    print("\n📦 Testing MultiStreamModel...")
    test_model = TestMultiStreamModel()
    test_model.test_model_init()
    test_model.test_forward_pass()
    test_model.test_fusion_types()
    test_model.test_variable_lengths()

    # Dataset tests
    print("\n📦 Testing MultiStreamDataset...")
    test_dataset = TestMultiStreamDataset()
    test_dataset.test_dataset_init()
    test_dataset.test_dataset_getitem()
    test_dataset.test_collate_fn()
    test_dataset.test_augmentation()
    test_dataset.test_dataloader()

    # End-to-end test
    test_e2e = TestEndToEnd()
    test_e2e.test_full_pipeline()

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()