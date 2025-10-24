"""
🚀 SignNet+ Training Script with MLflow

Works everywhere with automatic dataset detection!
"""

from pathlib import Path
import torch
import sys

print("="*70)
print("🚀 SignNet+ Training with MLflow")
print("="*70)

# ============================================================================
# 📂 AUTOMATIC DATASET DETECTION
# ============================================================================

def find_dataset():
    """Find dataset automatically - works on Windows, Linux, vast.ai!"""

    # Get script location
    script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()

    # Search locations (relative to script)
    search_paths = [
        script_dir / "LandmarksPhoenixDataset",                    # Same folder
        Path.cwd() / "LandmarksPhoenixDataset",                    # Current dir
        script_dir.parent / "LandmarksPhoenixDataset",             # Parent folder
        Path("/workspace/SignNet+/LandmarksPhoenixDataset"),       # vast.ai
        Path("D:/OST/SignNet/SignNet+/LandmarksPhoenixDataset"),   # Windows
    ]

    print(f"\n🔍 Searching for dataset...")
    print(f"   Script location: {script_dir}")
    print(f"   Working directory: {Path.cwd()}")

    # Try each location
    for path in search_paths:
        if path.exists() and (path / "landmarks_train").exists():
            print(f"   ✅ Found at: {path}")
            return path

    # Not found!
    print(f"\n❌ Dataset not found!")
    print(f"\n📁 Searched locations:")
    for path in search_paths:
        exists = "✅" if path.exists() else "❌"
        print(f"   {exists} {path}")

    print(f"\n💡 Solution: Put 'LandmarksPhoenixDataset' folder next to this script!")
    print(f"   Expected structure:")
    print(f"   SignNet+/")
    print(f"   ├── SignNetPlusMLFlowScript.py  (this file)")
    print(f"   ├── SignNetPlusModel.py")
    print(f"   └── LandmarksPhoenixDataset/")
    print(f"       ├── landmarks_train/")
    print(f"       ├── landmarks_dev/")
    print(f"       └── landmarks_test/")

    sys.exit(1)

# Find dataset
BASE_PATH = find_dataset()

# ============================================================================
# 📊 LOAD DATA
# ============================================================================

train_files = list((BASE_PATH / "landmarks_train").glob("*.npz"))
dev_files = list((BASE_PATH / "landmarks_dev").glob("*.npz"))
test_files = list((BASE_PATH / "landmarks_test").glob("*.npz"))

print(f"\n📁 Dataset loaded:")
print(f"   Train: {len(train_files)} files")
print(f"   Dev:   {len(dev_files)} files")
print(f"   Test:  {len(test_files)} files")

if len(train_files) == 0:
    print("\n❌ ERROR: No training files found!")
    print(f"   Checked: {BASE_PATH / 'landmarks_train'}/*.npz")
    sys.exit(1)

# ============================================================================
# 🔥 DEVICE CHECK
# ============================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n🔥 Device: {device}")
if device == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("   ⚠️  No GPU - training will be slower!")

# ============================================================================
# 📦 IMPORT TRAINING FUNCTION
# ============================================================================

print("\n📦 Loading training framework...")

try:
    from SignNetPlusModel import train_signbert_with_mlflow
    print("   ✅ SignNetPlusModel.py loaded!")
except ImportError as e:
    print(f"   ❌ ERROR: Cannot import from SignNetPlusModel.py")
    print(f"   {e}")
    print(f"\n💡 Ensure SignNetPlusModel.py is in the same folder:")
    print(f"   {Path(__file__).parent}")
    sys.exit(1)

# Install matplotlib if needed
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("\n📦 Installing matplotlib...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "matplotlib", "--break-system-packages"],
        capture_output=True
    )
    if result.returncode == 0:
        print("   ✅ matplotlib installed!")
    else:
        print("   ⚠️  Could not install matplotlib - plots may not work")

# ============================================================================
# 🧪 TRAINING CONFIGURATION
# ============================================================================

print("\n" + "="*70)
print("🧪 TRAINING CONFIGURATION")
print("="*70)

# Configuration
NUM_SAMPLES = 100      # Start small for testing
NUM_EPOCHS = 10
BATCH_SIZE = 8
OPTIMIZERS = ['adamw']  # Best optimizer

print(f"   Samples: {NUM_SAMPLES}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Optimizer: {OPTIMIZERS}")
print(f"   Augmentation: ON")
print(f"   Time estimate: ~10 minutes (GPU) / ~30 minutes (CPU)")

# ============================================================================
# 🚀 START TRAINING
# ============================================================================

print("\n" + "="*70)
print("🚀 STARTING TRAINING")
print("="*70)
print()

try:
    results = train_signbert_with_mlflow(
        data_files=train_files[:NUM_SAMPLES],
        optimizer_configs=OPTIMIZERS,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        use_augmentation=True,
        device=device
    )

    # ========================================================================
    # ✅ TRAINING COMPLETED
    # ========================================================================

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED!")
    print("="*70)

    for opt_name, result in results.items():
        print(f"\n📊 {opt_name.upper()} Results:")
        print(f"   Best Val Loss: {result['best_val_loss']:.4f}")
        print(f"   Training Time: {result['train_time']/60:.1f} minutes")

    print("\n🌐 View detailed results in MLflow:")
    print("   URL: https://mlflow.schlaepfer.me")
    print("   Experiment: SignNet+")
    print("   Run: SignBERT_adamw")

    print("\n" + "="*70)

except Exception as e:
    print("\n" + "="*70)
    print("❌ TRAINING FAILED")
    print("="*70)
    print(f"\nError: {e}")
    print("\nFull traceback:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 🎯 NEXT STEPS
# ============================================================================

print("\n" + "="*70)
print("🎯 NEXT STEPS")
print("="*70)
print("""
Current: 100 samples, 10 epochs → Val Loss: ~5.0

To improve results:

1. 🔥 Train on 500 samples (30-40 min):
   Change: NUM_SAMPLES = 500
           NUM_EPOCHS = 20
           BATCH_SIZE = 16

2. 🚀 Train on ALL data (3-4 hours):
   Change: NUM_SAMPLES = len(train_files)
           NUM_EPOCHS = 50
           BATCH_SIZE = 32

3. 🔬 Compare optimizers:
   Change: OPTIMIZERS = ['adam', 'adamw', 'adamw_low_lr', 'signbert_paper']

4. 📊 Check MLflow for:
   - Training curves
   - Loss comparison
   - Download trained model
""")

print("="*70)
print("🎉 Training script completed successfully!")
print("="*70)