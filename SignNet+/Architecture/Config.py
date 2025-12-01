# Config.py
import os
import platform
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import torch

def get_platform_paths():
    """Detect OS and return appropriate paths."""
    system = platform.system()

    if system == "Windows":
        # Windows paths
        base_dir = Path("D:/OST/SignNet/SignNet+")
        return {
            "train_dir": str(base_dir / "landmarks_train_cleaned"),
            "dev_dir": str(base_dir / "landmarks_dev_cleaned"),
            "test_dir": str(base_dir / "landmarks_test_cleaned"),
            "top_k_glosses_file": str(base_dir / "data_analysis_comprehensive" / "top200_glosses.csv"),
            "checkpoint_dir": str(base_dir / "checkpoints"),
            "num_workers": 0,  # Windows multiprocessing issues
        }
    else:
        # Linux / vast.ai paths
        base_dir = Path("/workspace")
        return {
            "train_dir": str(base_dir / "Data" / "landmarks_train_cleaned"),
            "dev_dir": str(base_dir / "Data" / "landmarks_dev_cleaned"),
            "test_dir": str(base_dir / "Data" / "landmarks_test_cleaned"),
            "top_k_glosses_file": str(base_dir / "Data" / "top200_glosses.csv"),
            "checkpoint_dir": str(base_dir / "checkpoints"),
            "num_workers": 4,  # Linux can handle more workers
        }


# Get paths for current platform
PLATFORM_PATHS = get_platform_paths()
CURRENT_OS = platform.system()
print(f"🖥️  Detected OS: {CURRENT_OS}")
print(f"📁 Base paths configured for: {'Windows' if CURRENT_OS == 'Windows' else 'Linux/vast.ai'}")


@dataclass
class DataConfig:
    """Data configuration."""
    # Paths - automatically set based on OS
    train_dir: str = PLATFORM_PATHS["train_dir"]
    dev_dir: str = PLATFORM_PATHS["dev_dir"]
    test_dir: str = PLATFORM_PATHS["test_dir"]
    top_k_glosses_file: str = PLATFORM_PATHS["top_k_glosses_file"]

    # Class selection
    use_top_k_classes: int = 50

    # Sequence settings
    max_frames: int = 214

    # Landmark configuration
    num_landmarks: int = 543
    landmark_dim: int = 2

    # Landmark groups
    left_hand_indices: List[int] = field(default_factory=lambda: list(range(0, 21)))
    right_hand_indices: List[int] = field(default_factory=lambda: list(range(21, 42)))
    pose_face_indices: List[int] = field(default_factory=lambda: list(range(42, 543)))

    # Augmentation settings
    use_augmentation: bool = True
    rotation_range: float = 25.0
    scale_range: tuple = (0.80, 1.20)
    translation_range: float = 0.15
    horizontal_flip_prob: float = 0.0

    # Occlusion augmentation
    occlusion_prob: float = 0.30
    left_hand_occlusion_prob: float = 0.20
    right_hand_occlusion_prob: float = 0.20
    face_occlusion_prob: float = 0.10
    occlusion_duration_range: tuple = (5, 15)

    # Temporal augmentation
    temporal_dropout_prob: float = 0.20
    frame_drop_range: tuple = (0.85, 1.0)

    # Bone-preserving augmentation
    use_bone_preserving: bool = True
    bone_noise_std: float = 0.02


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    num_landmarks: int = 543
    landmark_dim: int = 2
    num_classes: int = 50

    # GCN configuration
    gcn_input_dim: int = 2
    gcn_hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    gcn_dropout: float = 0.2

    # Transformer configuration
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    transformer_dropout: float = 0.2
    max_seq_length: int = 214

    # Positional encoding
    use_learned_pos_encoding: bool = False

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 8
    num_epochs: int = 100
    num_workers: int = PLATFORM_PATHS["num_workers"]  # OS-dependent!

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    gradient_clip: float = 1.0

    # Scheduler
    warmup_ratio: float = 0.05
    min_lr: float = 1e-6

    # Loss
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Early stopping
    patience: int = 20

    # Mixed precision
    use_mixed_precision: bool = True

    # Checkpoints - OS-dependent path
    checkpoint_dir: str = PLATFORM_PATHS["checkpoint_dir"]

    # Label Smoothing
    label_smoothing: float = 0.1


@dataclass
class MLflowConfig:
    """MLflow tracking configuration."""
    tracking_uri: str = "https://mlflow.schlaepfer.me"
    experiment_name: str = "SignNetWord_01122025"
    username: str = "andrei"
    password: str = "andrei"

    log_params: bool = True
    log_metrics: bool = True
    log_artifacts: bool = True
    log_model: bool = True
    log_confusion_matrix: bool = True
    confusion_matrix_normalize: str = "true"


@dataclass
class Config:
    """Main configuration object."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    experiment_name: str = "SignNet_Top50_Baseline"
    description: str = "Baseline Transformer model on Top-50 glosses"
    tags: dict = field(default_factory=lambda: {
        "model": "Transformer",
        "dataset": "RWTH-PHOENIX-2014",
        "platform": CURRENT_OS
    })

    def __post_init__(self):
        """Validate and sync configurations."""
        self.model.num_classes = self.data.use_top_k_classes
        self.model.max_seq_length = self.data.max_frames

        if not torch.cuda.is_available() and self.model.device == "cuda":
            print("⚠️  CUDA not available, falling back to CPU")
            self.model.device = "cpu"
            self.training.use_mixed_precision = False

        # Validate paths exist
        self._validate_paths()

    def _validate_paths(self):
        """Check if data paths exist."""
        paths_to_check = [
            ("train_dir", self.data.train_dir),
            ("dev_dir", self.data.dev_dir),
            ("test_dir", self.data.test_dir),
            ("top_k_glosses_file", self.data.top_k_glosses_file),
        ]

        missing = []
        for name, path in paths_to_check:
            if not Path(path).exists():
                missing.append(f"{name}: {path}")

        if missing:
            print(f"\n⚠️  WARNING: Missing paths:")
            for m in missing:
                print(f"   • {m}")
            print(f"\n   Make sure to upload data to the correct location!")

    def print_summary(self):
        """Print configuration summary."""
        print("\n" + "=" * 80)
        print("CONFIGURATION SUMMARY")
        print("=" * 80)

        print(f"\n🖥️  PLATFORM: {CURRENT_OS}")
        print(f"   num_workers: {self.training.num_workers}")

        print(f"\n📊 DATA:")
        print(f"   Train dir: {self.data.train_dir}")
        print(f"   Top-K Classes: {self.data.use_top_k_classes}")
        print(f"   Max Frames: {self.data.max_frames}")
        print(f"   Augmentation: {self.data.use_augmentation}")

        print(f"\n🧠 MODEL:")
        print(f"   Classes: {self.model.num_classes}")
        print(f"   d_model: {self.model.d_model}")
        print(f"   Layers: {self.model.n_layers}")
        print(f"   Device: {self.model.device}")

        print(f"\n🎯 TRAINING:")
        print(f"   Batch Size: {self.training.batch_size}")
        print(f"   Learning Rate: {self.training.learning_rate}")
        print(f"   Epochs: {self.training.num_epochs}")
        print(f"   Mixed Precision: {self.training.use_mixed_precision}")

        print(f"\n📈 MLFLOW:")
        print(f"   URI: {self.mlflow.tracking_uri}")
        print(f"   Experiment: {self.mlflow.experiment_name}")

        print("\n" + "=" * 80 + "\n")


def get_config(top_k: int = 50, use_augmentation: bool = True) -> Config:
    """Get configuration with custom settings."""
    config = Config()
    config.data.use_top_k_classes = top_k
    config.data.use_augmentation = use_augmentation
    config.model.num_classes = top_k

    if top_k:
        config.experiment_name = f"SignNet_Top{top_k}_Baseline"
        config.tags["classes"] = f"top-{top_k}"
    else:
        config.experiment_name = "SignNet_AllClasses"
        config.tags["classes"] = "all"

    config.__post_init__()
    return config


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CONFIG TEST")
    print("=" * 80)

    config = get_config(top_k=50, use_augmentation=True)
    config.print_summary()