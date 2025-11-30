# Test_Dataset.py
import sys

sys.path.append('D:/OST/SignNet/SignNet+/Architecture')


def main():
    """Main test function."""
    from Dataset import create_dataloaders
    from Config import get_config

    print("\n" + "=" * 80)
    print("TESTING DATASET.PY")
    print("=" * 80)

    # Create config
    config = get_config(top_k=50, use_augmentation=True)

    # IMPORTANT: Set num_workers=0 for Windows
    config.training.num_workers = 0

    # Test dataloaders
    train_loader, dev_loader, test_loader, vocab = create_dataloaders(config)

    # Test one batch
    print("\n🧪 Testing batch loading...")
    batch = next(iter(train_loader))

    print(f"\n📦 Batch contents:")
    print(f"   Landmarks shape: {batch['landmarks'].shape}")
    print(f"   Glosses shape: {batch['glosses'].shape}")
    print(f"   Landmarks lengths: {batch['landmarks_lengths']}")
    print(f"   Glosses lengths: {batch['glosses_lengths']}")

    # Decode some glosses
    print(f"\n📝 Sample glosses (first 3 samples):")
    for i in range(min(3, len(batch['glosses']))):
        indices = batch['glosses'][i][:batch['glosses_lengths'][i]].tolist()
        glosses = vocab.decode(indices)
        print(f"   Sample {i}: {glosses[:10]}...")  # First 10 glosses

    print("\n✅ Dataset.py test PASSED!")
    print("=" * 80)


if __name__ == '__main__':
    # REQUIRED for Windows multiprocessing!
    main()