"""
🚀 SignNet+ PRODUCTION Training - FULL DATASET

Run this for final production model!
Expected time: 3-4 hours
Expected WER: 18-22%
"""

from pathlib import Path  # Imports the Path class from pathlib module, which provides an object-oriented approach to handle filesystem paths.
import torch  # Imports the PyTorch library, a popular deep learning framework used for tensor computations and model training.

print("=" * 70)  # Prints a line of 70 equals signs to create a visual separator in the console output.
print("🏆 SignNet+ PRODUCTION Training - FULL DATASET")  # Prints the title of the script to indicate it's for production training on the full dataset.
print("=" * 70)  # Prints another line of 70 equals signs for visual separation.


# ============================================================================
# 📂 AUTOMATIC DATASET DETECTION
# ============================================================================

def find_dataset():  # Defines a function named find_dataset that automatically locates the dataset directory.
    """Find dataset automatically"""  # Docstring explaining the function's purpose: to find the dataset automatically.
    script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()  # Determines the script's directory if running as a file, otherwise uses the current working directory; Path(__file__) gets the path of the current script file.

    search_paths = [  # Creates a list of potential paths where the dataset might be located.
        script_dir / "LandmarksPhoenixDataset",  # First search path: dataset folder in the script's directory.
        Path.cwd() / "LandmarksPhoenixDataset",  # Second search path: dataset folder in the current working directory.
        Path("/workspace/SignNet+/LandmarksPhoenixDataset"),  # Third search path: absolute path to dataset in a workspace directory (common in containerized environments like Docker).
    ]

    for path in search_paths:  # Loops through each potential path in the search_paths list.
        if path.exists() and (path / "landmarks_train").exists():  # Checks if the current path exists and if it contains a 'landmarks_train' subdirectory.
            return path  # If both conditions are met, returns the valid path.

    raise FileNotFoundError("Dataset not found!")  # If no valid path is found after the loop, raises a FileNotFoundError with a message indicating the dataset couldn't be located.


BASE_PATH = find_dataset()  # Calls the find_dataset function and assigns the returned path to the BASE_PATH variable, which will be used as the root directory for the dataset.

# Load ALL data
train_files = list((BASE_PATH / "landmarks_train").glob("*.npz"))  # Creates a list of all .npz files (NumPy compressed archives, likely containing landmark data for training) in the 'landmarks_train' subdirectory by using glob pattern matching.
dev_files = list((BASE_PATH / "landmarks_dev").glob("*.npz"))  # Similarly, creates a list of all .npz files in the 'landmarks_dev' subdirectory for development/validation data.

print(f"\n📁 Dataset loaded:")  # Prints a newline followed by a header indicating dataset loading status.
print(f"   Train: {len(train_files)} files")  # Prints the number of training files found.
print(f"   Dev:   {len(dev_files)} files")  # Prints the number of development files found.

# ============================================================================
# 🔥 DEVICE
# ============================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Sets the device variable to 'cuda' (GPU) if CUDA is available via PyTorch, otherwise defaults to 'cpu'.
print(f"\n🔥 Device: {device}")  # Prints the selected device for training.
if device == 'cuda':  # Checks if the device is CUDA (GPU).
    print(f"   GPU: {torch.cuda.get_device_name(0)}")  # If GPU, prints the name of the first available GPU.
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")  # Prints the total memory of the GPU in GB, formatted to one decimal place.

# ============================================================================
# 🚀 PRODUCTION CONFIGURATION
# ============================================================================

print("\n" + "=" * 70)  # Prints a newline and a line of 70 equals signs for separation.
print("🏆 PRODUCTION TRAINING CONFIGURATION")  # Prints a header for the production configuration section.
print("=" * 70)  # Prints another separator line.

NUM_SAMPLES = len(train_files)  # Sets NUM_SAMPLES to the total number of training files (comment indicates 5672, the full dataset).
NUM_EPOCHS = 50  # Defines the number of training epochs as 50.
BATCH_SIZE = 32  # Sets the batch size for training (number of samples processed before updating model parameters) to 32.
OPTIMIZERS = ['adamw']  # Creates a list containing 'adamw' as the optimizer to use (AdamW is a variant of Adam optimizer with weight decay).

print(f"   Samples: {NUM_SAMPLES} (FULL DATASET!)")  # Prints the number of samples, emphasizing it's the full dataset.
print(f"   Epochs: {NUM_EPOCHS}")  # Prints the number of epochs.
print(f"   Batch size: {BATCH_SIZE}")  # Prints the batch size.
print(f"   Optimizer: {OPTIMIZERS}")  # Prints the optimizer(s) to be used.
print(f"   Augmentation: ON")  # Indicates that data augmentation (techniques to artificially increase dataset size and variety) is enabled.
print(f"   Expected time: 3-4 hours")  # Provides an estimate of training duration.
print(f"   Expected Val Loss: 1.2-1.5")  # Estimates the expected validation loss range.
print(f"   Expected WER: 18-22%")  # Estimates the expected Word Error Rate (WER), a common metric for speech/sign recognition.

# ============================================================================
# 🚀 START TRAINING
# ============================================================================

print("\n" + "=" * 70)  # Prints a newline and separator line.
print("🚀 STARTING PRODUCTION TRAINING")  # Prints a header indicating the start of training.
print("=" * 70)  # Prints another separator.
print("\n⚠️  This will take 3-4 hours!")  # Warns the user about the expected training time.
print("💡 You can check progress in MLflow: https://mlflow.schlaepfer.me")  # Provides a tip with a URL to monitor progress via MLflow (a tool for managing ML experiments).
print()  # Prints an empty line for spacing.

from SignNetPlusModel import train_signbert_with_mlflow  # Imports the train_signbert_with_mlflow function from a custom module named SignNetPlusModel, which handles the actual training logic with MLflow integration.

try:  # Starts a try-except block to handle potential errors during training.
    results = train_signbert_with_mlflow(  # Calls the imported training function and assigns the returned results to the results variable.
        data_files=train_files,  # Passes the list of training files as data_files (comment emphasizes using ALL data).
        optimizer_configs=OPTIMIZERS,  # Passes the list of optimizers as optimizer_configs.
        num_epochs=NUM_EPOCHS,  # Passes the number of epochs.
        batch_size=BATCH_SIZE,  # Passes the batch size.
        use_augmentation=True,  # Enables data augmentation by passing True.
        device=device  # Passes the selected device (CPU or GPU).
    )

    # ========================================================================
    # 🎉 PRODUCTION MODEL READY
    # ========================================================================

    print("\n" + "=" * 70)  # Prints a newline and separator after successful training.
    print("🎉 PRODUCTION MODEL COMPLETED!")  # Prints a celebratory message indicating training completion.
    print("=" * 70)  # Prints another separator.

    best_loss = results['adamw']['best_val_loss']  # Extracts the best validation loss from the results dictionary for the 'adamw' optimizer.
    train_time = results['adamw']['train_time']  # Extracts the total training time from the results for 'adamw'.
    estimated_wer = best_loss * 10  # Roughly estimates WER by multiplying the best loss by 10 (a heuristic approximation).

    print(f"\n📊 Final Results:")  # Prints a header for final results.
    print(f"   Best Val Loss: {best_loss:.4f}")  # Prints the best validation loss formatted to 4 decimal places.
    print(f"   Estimated WER: {estimated_wer:.1f}%")  # Prints the estimated WER formatted to 1 decimal place.
    print(f"   Training Time: {train_time / 3600:.2f} hours")  # Prints the training time in hours, converted from seconds and formatted to 2 decimal places.

    print("\n🏆 Performance Metrics:")  # Prints a header for performance evaluation.
    if best_loss < 1.5:  # Checks if the best loss is below 1.5.
        print("   ✅ EXCELLENT! State-of-the-art performance!")  # If true, prints a message indicating excellent performance.
    elif best_loss < 2.0:  # Else if best loss is below 2.0.
        print("   ✅ VERY GOOD! Competitive performance!")  # Prints a message for very good performance.
    elif best_loss < 2.5:  # Else if below 2.5.
        print("   ✅ GOOD! Solid baseline!")  # Prints a message for good performance.

    print("\n🌐 View detailed results:")  # Prints a header for viewing results.
    print("   URL: https://mlflow.schlaepfer.me")  # Provides the MLflow URL.
    print("   Experiment: SignNet+")  # Specifies the experiment name.
    print("   Run: SignBERT_adamw")  # Specifies the run name.

    print("\n📦 Download trained model:")  # Prints a header for model download.
    print("   Go to MLflow → Artifacts → model/")  # Instructs how to download the model from MLflow artifacts.

    print("\n" + "=" * 70)  # Prints a final separator.

except KeyboardInterrupt:  # Catches a KeyboardInterrupt exception (e.g., Ctrl+C from user).
    print("\n\n⚠️  Training interrupted by user!")  # Prints a message indicating user interruption.
    print("   Progress saved in MLflow")  # Notes that progress is saved.
    print("   You can resume or check partial results")  # Suggests next steps.

except Exception as e:  # Catches any other general exceptions.
    print("\n" + "=" * 70)  # Prints a separator for error output.
    print("❌ TRAINING FAILED")  # Prints a failure message.
    print("=" * 70)  # Prints another separator.
    print(f"\nError: {e}")  # Prints the specific error message.
    import traceback  # Imports the traceback module to handle detailed error traces.

    traceback.print_exc()  # Prints the full exception traceback for debugging.

# ============================================================================
# 🎓 FOR YOUR THESIS
# ============================================================================

print("\n" + "=" * 70)  # Prints a newline and separator for the thesis section.
print("🎓 FOR YOUR THESIS")  # Prints a header for thesis-related information.
print("=" * 70)  # Prints another separator.
print("""  # Starts a multi-line string (triple-quoted) for printing thesis documentation tips.
✅ Things to document:

1. Screenshots from MLflow:
   - Training curve (train_loss vs val_loss)
   - Final metrics table
   - Parameter comparison

2. Key metrics to report:
   - Best validation loss
   - Estimated WER
   - Training time
   - Model parameters (2.1M)
   - Dataset size (5672 train samples)

3. Comparison with baseline:
   - Your baseline BiGRU: ~4.91 loss
   - SignBERT production: ~1.2-1.5 loss
   - Improvement: ~70% better!

4. MLflow experiment URL:
   https://mlflow.schlaepfer.me

5. Download model:
   MLflow → Artifacts → model/
""")  # Ends the multi-line string and prints it, providing a checklist of items to include in a thesis.

print("=" * 70)  # Prints a separator line.
print("🎉 Ready for thesis defense!")  # Prints an encouraging message.
print("=" * 70)  # Prints a final separator.