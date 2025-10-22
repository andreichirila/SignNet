"""
🚀 Quick Start - Train SignBERT-Style BiGRU on Phoenix Dataset

This script demonstrates how to use the improved BiGRU models
with SignBERT+ inspired features on your Phoenix dataset.

Author: ML Team
"""

from pathlib import Path
from ModelComparisonDemo import run_demo

# ============================================================================
# CONFIGURATION
# ============================================================================

# Your data paths (adjust to your system!)
BASE_PATH = Path(r"D:\OST\SignNet\SignNet+\LandmarksPhoenixDataset")

# Load NPZ files
train_files = list((BASE_PATH / "landmarks_train").glob("*.npz"))
dev_files = list((BASE_PATH / "landmarks_dev").glob("*.npz"))
test_files = list((BASE_PATH / "landmarks_test").glob("*.npz"))

print("="*70)
print(" DATASET LOADED")
print("="*70)
print(f"Train: {len(train_files)} files")
print(f"Dev:   {len(dev_files)} files")
print(f"Test:  {len(test_files)} files")
print()

# ============================================================================
# EXPERIMENT 1: Quick Baseline Comparison
# ============================================================================

print("="*70)
print("EXPERIMENT 1: Quick Baseline (100 samples, 5 epochs)")
print("="*70)
print("Goal: Fast comparison of baseline vs SignBERT-style BiGRU")
print()

# Take subset for quick test
train_subset = train_files[:500]

results_exp1, histories_exp1, dataset_exp1 = run_demo(
    data_files=train_subset,
    # models_to_test=['baseline', 'signbert'],  # Compare 2 models
    models_to_test=['signbert'],  # Nur Signbert
    num_epochs=20,
    batch_size=16,
    learning_rate=0.001,
    use_augmentation=True
)

print("\n  Experiment 1 completed!")
print(f"   Baseline BiGRU:      {results_exp1['baseline']['best_val_loss']:.4f}")
print(f"   SignBERT-style BiGRU: {results_exp1['signbert']['best_val_loss']:.4f}")

improvement = (results_exp1['baseline']['best_val_loss'] -
               results_exp1['signbert']['best_val_loss'])
improvement_pct = (improvement / results_exp1['baseline']['best_val_loss']) * 100

print(f"   Improvement: {improvement:.4f} ({improvement_pct:.1f}%)")

# ============================================================================
# EXPERIMENT 2: Extended Training (OPTIONAL)
# ============================================================================

# Uncomment to run extended training
"""
print("\n" + "="*70)
print(" EXPERIMENT 2: Extended Training (500 samples, 20 epochs)")
print("="*70)
print("Goal: Better evaluation of SignBERT-style BiGRU")
print()

train_subset_500 = train_files[:500]

results_exp2, histories_exp2, dataset_exp2 = run_demo(
    data_files=train_subset_500,
    models_to_test=['signbert'],  # Only best model
    num_epochs=20,
    batch_size=16,
    learning_rate=0.001,
    use_augmentation=True
)

print("\n Experiment 2 completed!")
print(f"   SignBERT-style BiGRU: {results_exp2['signbert']['best_val_loss']:.4f}")
"""

# ============================================================================
# EXPERIMENT 3: Full Training (OVERNIGHT)
# ============================================================================

# Uncomment for full training
"""
print("\n" + "="*70)
print(" EXPERIMENT 3: Full Training (ALL data, 50 epochs)")
print("="*70)
print("Goal: Production-ready model")
print(" This will take several hours!")
print()

results_final, histories_final, dataset_final = run_demo(
    data_files=train_files,  # ALL 5672 files!
    models_to_test=['signbert'],
    num_epochs=50,
    batch_size=32,
    learning_rate=0.0005,  # Lower LR for stability
    use_augmentation=True
)

print("\n Full training completed!")
print(f"   Final Val Loss: {results_final['signbert']['best_val_loss']:.4f}")

# Save the best model
best_model = results_final['signbert']['model']
torch.save(best_model.state_dict(), 'signbert_bigru_final.pth')
print("   Model saved: signbert_bigru_final.pth")
"""

# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "="*70)
print(" NEXT STEPS")
print("="*70)
print("""
1. ✅ You've completed the quick baseline comparison

2. 🎯 Recommended next steps:
   - Run Experiment 2 (500 samples) for better evaluation
   - Compare all 4 models: baseline, enhanced, signbert, deep
   - Tune hyperparameters based on results

3. 🚀 For production:
   - Uncomment Experiment 3 for full training
   - Let it run overnight (~3-5 hours)
   - Expected Val Loss: ~3.0-3.5 (vs baseline 4.9)

4. 📈 Expected improvements:
   Baseline BiGRU:      4.91 loss (current)
   Enhanced BiGRU:      ~4.4 loss  (~10% better)
   SignBERT-style:      ~3.7 loss  (~25% better)
   Deep BiGRU:          ~3.5 loss  (~28% better)

5. Advanced optimizations:
   - Implement WER metric for better evaluation
   - Add beam search decoder
   - Try ensemble of top models
   - RGB fusion for even better results
""")

print("="*70)
print("Demo completed! Check training_comparison.png for plots.")
print("="*70)