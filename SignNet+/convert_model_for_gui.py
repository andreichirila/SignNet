"""
🔧 Model Converter for GUI
Converts trained model to GUI-compatible format with vocab

Run this ONCE on vast.ai after training!
"""

import torch
from pathlib import Path
import sys

print("="*70)
print("🔧 SignNet+ Model Converter for GUI")
print("="*70)

# ============================================================================
# 📂 PATHS
# ============================================================================

# Input: Your trained model from vast.ai
INPUT_MODEL = Path("Fertige_Models/current_model.pth")

# Output: GUI-compatible model
OUTPUT_MODEL = Path("Fertige_Models/current_model_converted.pth")
OUTPUT_MODEL.parent.mkdir(exist_ok=True)

print(f"\n📥 Input:  {INPUT_MODEL}")
print(f"📤 Output: {OUTPUT_MODEL}")

# ============================================================================
# 🔄 CONVERT
# ============================================================================

print("\n📦 Loading model...")
checkpoint = torch.load(INPUT_MODEL, map_location='cpu', weights_only=False)

print(f"✅ Model loaded!")

# Check if checkpoint is a model or dict
if isinstance(checkpoint, torch.nn.Module):
    print(f"\n⚠️  Model saved in DIRECT format (not dict)")
    print(f"   Model type: {type(checkpoint).__name__}")

    # Extract what we can
    model_direct = checkpoint

    # We need to create vocab from dataset
    print(f"\n📦 Loading dataset to extract vocab...")

    try:
        # Try current directory first
        from SignNetPlusModel_Base import PhoenixDataset

        # Try different dataset paths
        dataset_paths = [
            Path("LandmarksPhoenixDataset"),
            Path("../LandmarksPhoenixDataset"),
            Path("D:/OST/SignNet/SignNet+/LandmarksPhoenixDataset"),
        ]

        dataset_path = None
        for path in dataset_paths:
            if path.exists() and (path / "landmarks_train").exists():
                dataset_path = path
                break

        if not dataset_path:
            print(f"\n❌ Could not find dataset!")
            print(f"   Searched in:")
            for path in dataset_paths:
                print(f"   - {path}")
            print(f"\n💡 Please ensure LandmarksPhoenixDataset is in current directory")
            sys.exit(1)

        print(f"✅ Found dataset: {dataset_path}")

        train_files = list((dataset_path / "landmarks_train").glob("*.npz"))
        print(f"   Loading ALL {len(train_files)} samples to extract COMPLETE vocab...")

        # Load ALL files to get complete vocab!
        dataset = PhoenixDataset(train_files, augment=False)
        vocab = dataset.gloss_to_idx

        print(f"✅ Extracted COMPLETE vocab: {len(vocab)} glosses")

        # Create checkpoint dict
        checkpoint = {
            'model_state_dict': model_direct.state_dict(),
            'vocab': vocab,
            'config': {
                'vocab_size': len(vocab),
                'hidden_dim': 320,
                'num_layers': 3,
                'dropout': 0.4
            },
            'val_loss': 0.0,
            'epoch': 0
        }

        print(f"\n📋 Created checkpoint with:")
        print(f"   ✅ model_state_dict: present")
        print(f"   ✅ vocab: {len(vocab)} glosses")
        print(f"   ✅ config: present")

    except Exception as e:
        print(f"\n❌ Could not extract vocab: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

else:
    # Normal dict format
    print(f"\n📋 Checkpoint contents:")
    for key in checkpoint.keys():
        if key == 'vocab':
            print(f"   ✅ {key}: {len(checkpoint[key])} glosses")
        elif key == 'model_state_dict':
            print(f"   ✅ {key}: present")
        else:
            print(f"   ✅ {key}: {checkpoint[key]}")

    # Check if vocab exists
    if 'vocab' not in checkpoint:
        print("\n⚠️  WARNING: No vocab in checkpoint!")
        print("   This shouldn't happen with dict format!")
        sys.exit(1)

# ============================================================================
# 💾 SAVE
# ============================================================================

print(f"\n💾 Saving GUI-compatible model...")

# Ensure all required fields
required_fields = {
    'model_state_dict': checkpoint['model_state_dict'],
    'vocab': checkpoint['vocab'],
    'config': checkpoint.get('config', {
        'vocab_size': len(checkpoint['vocab']),
        'hidden_dim': 320,
        'num_layers': 3,
        'dropout': 0.4
    }),
    'val_loss': checkpoint.get('val_loss', 0.0),
    'epoch': checkpoint.get('epoch', 0)
}

torch.save(required_fields, OUTPUT_MODEL)

print(f"✅ Saved to: {OUTPUT_MODEL}")

# ============================================================================
# ✅ VERIFY
# ============================================================================

print(f"\n🔍 Verifying saved model...")

try:
    verify = torch.load(OUTPUT_MODEL, map_location='cpu', weights_only=False)

    print(f"✅ Model can be loaded!")
    print(f"\n📊 Model info:")
    print(f"   Vocab size: {len(verify['vocab'])}")
    print(f"   Val loss: {verify.get('val_loss', 'N/A')}")
    print(f"   Epoch: {verify.get('epoch', 'N/A')}")

    # Show sample glosses
    print(f"\n📝 Sample glosses (first 20):")
    glosses = list(verify['vocab'].keys())[:20]
    for i, gloss in enumerate(glosses, 1):
        print(f"   {i:2d}. {gloss}")

    if len(verify['vocab']) > 20:
        print(f"   ... and {len(verify['vocab']) - 20} more")

except Exception as e:
    print(f"❌ Verification failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ CONVERSION COMPLETE!")
print("="*70)
