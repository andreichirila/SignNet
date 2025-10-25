"""
🚀 SignNet+ Production Training Framework
═══════════════════════════════════════════

Features:
- SignBERT-style BiGRU as best model
- MLflow experiment tracking
- Optimizer comparison (Adam, AdamW, AdamW+Cosine, etc.)
- Full metrics logging
- Production-ready deployment

Author: Roman Schläpfer, Andrei Chirila
Date: 2025-10-24
"""
import numpy as np  # Imports NumPy library as np, used for numerical computations and handling arrays, especially for loading .npz files.
import torch  # Imports PyTorch library, the core framework for building and training neural networks.
import torch.nn as nn  # Imports the neural networks module from PyTorch, providing building blocks like layers and loss functions.
import torch.nn.functional as F  # Imports functional API from PyTorch's nn module, containing functions like activation functions and operations.
from torch.utils.data import Dataset, DataLoader  # Imports Dataset and DataLoader classes from PyTorch, used for handling and batching data.
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence  # Imports utilities for handling variable-length sequences in RNNs, such as padding and packing.
import matplotlib.pyplot as plt  # Imports Matplotlib's pyplot module as plt, used for plotting training curves.
from pathlib import Path  # Imports Path class from pathlib, providing an object-oriented way to handle filesystem paths.
from typing import Dict, List, Tuple, Optional  # Imports type hints from typing module to specify types for function parameters and returns.
import time  # Imports time module, used for measuring training duration.
import random  # Imports random module, used for random sampling in data augmentation.
from collections import defaultdict  # Imports defaultdict from collections, a dictionary that provides default values for missing keys.
import mlflow  # Imports MLflow library, a platform for managing ML experiments, including tracking parameters and metrics.
import mlflow.pytorch  # Imports PyTorch-specific integration from MLflow, allowing logging of PyTorch models.
import warnings  # Imports warnings module, used to suppress specific warning messages.

import os  # Imports os module, used for operating system interactions like setting environment variables.

# Suppress PyTorch version warning from MLflow
warnings.filterwarnings('ignore', category=UserWarning, module='mlflow')  # Configures warnings to ignore UserWarning from the 'mlflow' module, suppressing PyTorch version mismatch warnings.


# ============================================================================
# 🔧 HELPER FUNCTIONS
# ============================================================================

def find_dataset_path(relative_path: str = "LandmarksPhoenixDataset") -> Path:  # Defines a function find_dataset_path that takes an optional relative_path string (default "LandmarksPhoenixDataset") and returns a Path object.
    """
    Find dataset path - works on Windows, Linux, vast.ai!

    Search order:
    1. Relative to script location
    2. Current working directory
    3. Common locations (vast.ai, Windows default)

    Args:
        relative_path: Folder name of dataset

    Returns:
        Path to dataset
    """  # This is a multi-line docstring explaining the function's purpose, search order, arguments, and return value.
    # Try relative to script
    script_dir = Path(__file__).parent  # Gets the parent directory of the current script file using Path(__file__).parent.
    dataset_path = script_dir / relative_path  # Constructs the dataset path by joining script_dir with relative_path using Path's / operator.

    if dataset_path.exists():  # Checks if the constructed dataset_path exists.
        return dataset_path  # If it exists, returns the path.

    # Try current directory
    dataset_path = Path.cwd() / relative_path  # Constructs the dataset path relative to the current working directory.
    if dataset_path.exists():  # Checks if this new dataset_path exists.
        return dataset_path  # If it exists, returns the path.

    # Try common locations
    common_paths = [  # Creates a list of common paths where the dataset might be located.
        Path("/workspace/SignNet+") / relative_path,  # First common path: /workspace/SignNet+ joined with relative_path (e.g., for vast.ai environments).
        Path("/workspace") / relative_path,  # Second: /workspace joined with relative_path.
        Path("D:/OST/SignNet/SignNet+") / relative_path,  # Third: Windows-specific path D:/OST/SignNet/SignNet+ joined with relative_path.
        Path.home() / "SignNet" / relative_path,  # Fourth: User's home directory joined with "SignNet" and relative_path.
    ]

    for path in common_paths:  # Loops over each path in common_paths.
        if path.exists():  # Checks if the current path exists.
            return path  # If it exists, returns the path.

    # Not found
    raise FileNotFoundError(  # Raises a FileNotFoundError if no path is found.
        f"Dataset '{relative_path}' not found!\n"  # Starts the error message with the dataset name.
        f"Searched locations:\n"  # Adds a header for searched locations.
        f"  - {script_dir / relative_path}\n"  # Lists the script-relative path.
        f"  - {Path.cwd() / relative_path}\n" +  # Lists the cwd-relative path, concatenated with the next string.
        "\n".join(f"  - {p}" for p in common_paths) +  # Joins the common paths into a newline-separated string with indentation.
        f"\n\nCurrent directory: {Path.cwd()}\n"  # Adds current working directory info.
        f"Script directory: {script_dir}\n\n"  # Adds script directory info.
        "Please ensure dataset is in the same folder as this script!"  # Ends with a helpful message.
    )  # Closes the f-string and the raise statement.


# ============================================================================
# 🔧 MLFLOW CONFIGURATION
# ============================================================================

def setup_mlflow():  # Defines a function setup_mlflow with no arguments or return value.
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'andrei'  # Sets the MLFLOW_TRACKING_USERNAME environment variable to 'andrei'.
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'andrei'  # Sets the MLFLOW_TRACKING_PASSWORD environment variable to 'andrei'.
    """Initialize MLflow tracking"""  # A single-line docstring explaining the function's purpose.
    mlflow.set_tracking_uri("https://mlflow.schlaepfer.me")  # Sets the MLflow tracking server URI to the specified URL.
    mlflow.set_experiment("SignNet+")  # Sets the current MLflow experiment name to "SignNet+".
    print("✅ MLflow configured:")  # Prints a confirmation message for MLflow setup.
    print(f"   Tracking URI: https://mlflow.schlaepfer.me")  # Prints the tracking URI.
    print(f"   Experiment: SignNet+")  # Prints the experiment name.


# ============================================================================
# 📊 DATASET CLASS
# ============================================================================

class PhoenixDataset(Dataset):  # Defines a class PhoenixDataset that inherits from PyTorch's Dataset class, for handling the RWTH-PHOENIX dataset.
    """RWTH-PHOENIX Dataset with landmarks and masked augmentation"""  # A docstring describing the class: handles RWTH-PHOENIX dataset with landmarks and augmentation.

    def __init__(self, npz_files: List[Path], augment: bool = False):  # Defines the initializer method, taking a list of Path objects for npz_files and an optional augment boolean (default False).
        self.files = npz_files  # Assigns the provided npz_files list to self.files.
        self.augment = augment  # Assigns the augment flag to self.augment.

        # Build vocabulary
        self.gloss_to_idx = {'<BLANK>': 0, '<PAD>': 1}  # Initializes a dictionary gloss_to_idx mapping gloss strings to indices, starting with special tokens <BLANK> (0) and <PAD> (1).
        self._build_vocab()  # Calls the private method _build_vocab to populate the vocabulary.

        print(f"📚 Dataset: {len(self.files)} samples, {len(self.gloss_to_idx)} glosses")  # Prints a summary of the dataset size and vocabulary size.

    def _build_vocab(self):  # Defines a private method _build_vocab with no arguments.
        idx = len(self.gloss_to_idx)  # Sets idx to the current length of gloss_to_idx (starting from 2).
        for file_path in self.files:  # Loops over each file path in self.files.
            data = np.load(file_path)  # Loads the .npz file using NumPy, returning a NpzFile object.
            for gloss in data['glosses']:  # Loops over each gloss in the 'glosses' array from the loaded data.
                if gloss not in self.gloss_to_idx:  # Checks if the gloss is not already in the dictionary.
                    self.gloss_to_idx[gloss] = idx  # Adds the gloss with the current idx.
                    idx += 1  # Increments idx for the next gloss.
        self.idx_to_gloss = {v: k for k, v in self.gloss_to_idx.items()}  # Creates a reverse mapping idx_to_gloss by swapping keys and values.

    def __len__(self):  # Defines the special method __len__, called by len() on the dataset.
        return len(self.files)  # Returns the number of files in the dataset.

    def __getitem__(self, idx):  # Defines the special method __getitem__, called to fetch the item at index idx.
        data = np.load(self.files[idx])  # Loads the .npz file at the given idx.
        landmarks = torch.FloatTensor(data['landmarks'])  # Converts the 'landmarks' array to a PyTorch FloatTensor.

        if self.augment:  # Checks if augmentation is enabled.
            landmarks = self._masked_augmentation(landmarks)  # If yes, applies masked augmentation to the landmarks.

        gloss_indices = [self.gloss_to_idx[g] for g in data['glosses']]  # Converts each gloss in 'glosses' to its index using gloss_to_idx.
        gloss_tensor = torch.LongTensor(gloss_indices)  # Converts the list of indices to a PyTorch LongTensor.

        return landmarks, gloss_tensor, len(landmarks)  # Returns the landmarks tensor, gloss tensor, and the sequence length (number of frames).

    def _masked_augmentation(self, landmarks: torch.Tensor, mask_ratio: float = 0.2):  # Defines a private method for masked augmentation, taking landmarks tensor and optional mask_ratio (default 0.2).
        """Masked augmentation (SignBERT+ style)"""  # Docstring indicating it's SignBERT+-style augmentation.
        augmented = landmarks.clone()  # Creates a clone of the landmarks tensor to avoid modifying the original.
        num_frames = len(landmarks)  # Gets the number of frames (sequence length).
        num_mask = int(num_frames * mask_ratio)  # Calculates the number of frames to mask based on the ratio.

        if num_mask > 0:  # Checks if there's at least one frame to mask.
            mask_frames = random.sample(range(num_frames), num_mask)  # Randomly samples num_mask frame indices.
            for frame_idx in mask_frames:  # Loops over each selected frame index.
                strategy = random.choice(['mask_joints', 'mask_frame', 'gaussian_noise', 'identity'])  # Randomly selects an augmentation strategy from the list.

                if strategy == 'mask_joints':  # If strategy is 'mask_joints'.
                    num_joints = random.randint(5, 10)  # Randomly selects number of joints to mask (5-10).
                    max_joints = landmarks.shape[1] // 3  # Calculates max joints based on landmark dimension (assuming 3 coords per joint: x,y,z).
                    joint_indices = random.sample(range(max_joints), min(num_joints, max_joints))  # Samples joint indices up to the available number.
                    for j in joint_indices:  # Loops over selected joint indices.
                        augmented[frame_idx, j*3:(j+1)*3] = 0  # Sets the 3 coordinates of the joint to 0 (masks it).
                elif strategy == 'mask_frame':  # If strategy is 'mask_frame'.
                    augmented[frame_idx] = 0  # Sets the entire frame to 0.
                elif strategy == 'gaussian_noise':  # If strategy is 'gaussian_noise'.
                    noise = torch.randn_like(augmented[frame_idx]) * 0.01  # Generates Gaussian noise with std 0.01, same shape as the frame.
                    augmented[frame_idx] += noise  # Adds the noise to the frame.

        return augmented  # Returns the augmented landmarks tensor.


def collate_fn(batch):  # Defines a collate function for DataLoader, taking a batch (list of items from __getitem__).
    """Collate function for variable length sequences"""  # Docstring explaining it's for handling variable-length sequences.
    landmarks, glosses, lengths = zip(*batch)  # Unzips the batch into landmarks, glosses, and lengths tuples.
    padded_landmarks = pad_sequence(landmarks, batch_first=True, padding_value=0.0)  # Pads the landmarks sequences to the same length, batch_first=True, padding with 0.0.
    gloss_lengths = torch.LongTensor([len(g) for g in glosses])  # Creates a tensor of gloss sequence lengths.
    lengths = torch.LongTensor(lengths)  # Converts lengths tuple to LongTensor.
    return padded_landmarks, glosses, lengths, gloss_lengths  # Returns padded landmarks, glosses list, lengths tensor, and gloss_lengths tensor.


# ============================================================================
# 🏗️ SIGNBERT-STYLE BiGRU MODEL (BEST MODEL)
# ============================================================================

class PositionalEncoding(nn.Module):  # Defines a class PositionalEncoding inheriting from nn.Module, for adding positional information to sequences.
    """Temporal position encoding"""  # Docstring indicating it's for temporal (sequence) position encoding.
    def __init__(self, d_model: int, max_len: int = 5000):  # Initializer taking d_model (embedding dimension) and optional max_len (default 5000).
        super().__init__()  # Calls the parent nn.Module initializer.
        position = torch.arange(max_len).unsqueeze(1)  # Creates a tensor of positions 0 to max_len-1, unsqueezed to (max_len, 1).
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))  # Computes the division term for sinusoidal encoding using exponential.
        pe = torch.zeros(max_len, 1, d_model)  # Initializes a zero tensor for positional encodings, shape (max_len, 1, d_model).
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # Sets even indices to sin(position * div_term).
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # Sets odd indices to cos(position * div_term).
        self.register_buffer('pe', pe)  # Registers 'pe' as a buffer (persistent tensor not considered a parameter).

    def forward(self, x):  # Defines the forward pass, taking input tensor x.
        seq_len = x.size(1)  # Gets the sequence length from x's shape.
        x = x + self.pe[:seq_len, 0, :].unsqueeze(0)  # Adds the positional encoding to x, slicing to seq_len and unsqueezing for broadcasting.
        return x  # Returns the encoded x.


class SignBERTBiGRU(nn.Module):  # Defines the main model class SignBERTBiGRU inheriting from nn.Module.
    """
    🏆 SignBERT-style BiGRU - Best Model

    Features:
    - Spatial-Temporal Position Encoding
    - Gesture/Spatial feature decomposition
    - Hierarchical feature fusion
    """  # Multi-line docstring describing the model, its name, and key features.

    def __init__(self, input_dim=1659, hidden_dim=320, num_layers=3,
                 vocab_size=100, dropout=0.4):  # Initializer with defaults: input_dim=1659 (landmark features), hidden_dim=320, num_layers=3, vocab_size=100, dropout=0.4.
        super().__init__()  # Calls parent initializer.

        self.name = "SignBERT_BiGRU"  # Sets the model name attribute.

        # Feature dimensions
        self.hand_dim = 126  # Sets hand_dim to 126 (likely 42 hand joints * 3 coords).
        self.arm_dim = 21    # Sets arm_dim to 21 (likely 7 arm/body joints * 3 coords).

        # Gesture State Encoder
        self.gesture_encoder = nn.Sequential(  # Creates a sequential module for gesture (hand) encoding.
            nn.Linear(self.hand_dim, 256),  # Linear layer from hand_dim to 256.
            nn.LayerNorm(256),  # Layer normalization over 256 features.
            nn.ReLU(),  # ReLU activation.
            nn.Dropout(dropout),  # Dropout with given rate.
            nn.Linear(256, 256),  # Another linear layer 256 to 256.
            nn.ReLU()  # ReLU activation.
        )

        # Spatial Position Encoder
        self.spatial_encoder = nn.Sequential(  # Sequential module for spatial (arm/body) encoding.
            nn.Linear(self.arm_dim, 64),  # Linear from arm_dim to 64.
            nn.ReLU(),  # ReLU.
            nn.Dropout(dropout)  # Dropout.
        )

        # Feature fusion
        self.fusion = nn.Sequential(  # Sequential for fusing gesture and spatial features.
            nn.Linear(256 + 64, hidden_dim),  # Linear from concatenated dim (320) to hidden_dim.
            nn.LayerNorm(hidden_dim),  # Layer norm.
            nn.ReLU(),  # ReLU.
            nn.Dropout(dropout)  # Dropout.
        )

        # Temporal Position Encoding
        self.temporal_pe = PositionalEncoding(d_model=hidden_dim)  # Instantiates PositionalEncoding with hidden_dim.

        # BiGRU
        self.gru = nn.GRU(  # Creates a bidirectional GRU layer.
            input_size=hidden_dim,  # Input size is hidden_dim.
            hidden_size=hidden_dim // 2,  # Hidden size half of hidden_dim (for bidir doubling).
            num_layers=num_layers,  # Number of GRU layers.
            batch_first=True,  # Expects batch-first input (batch, seq, feature).
            dropout=dropout if num_layers > 1 else 0,  # Dropout only if multiple layers.
            bidirectional=True  # Bidirectional processing.
        )

        # Output head
        self.output = nn.Sequential(  # Sequential for output projection to vocab.
            nn.Linear(hidden_dim, hidden_dim // 2),  # Linear to half hidden_dim.
            nn.ReLU(),  # ReLU.
            nn.Dropout(dropout),  # Dropout.
            nn.Linear(hidden_dim // 2, vocab_size)  # Final linear to vocab_size (logits).
        )

    def forward(self, x, lengths):  # Forward pass taking input x (batch, seq, features) and lengths.
        # Feature decomposition
        hand_features = x[:, :, :self.hand_dim]  # Extracts hand features (first hand_dim dims).
        spatial_features = x[:, :, self.hand_dim:self.hand_dim + self.arm_dim]  # Extracts spatial features (next arm_dim dims).

        # Encode features
        gesture_feat = self.gesture_encoder(hand_features)  # Encodes hand features through gesture_encoder.
        spatial_feat = self.spatial_encoder(spatial_features)  # Encodes spatial features.

        # Fuse features
        fused = torch.cat([gesture_feat, spatial_feat], dim=-1)  # Concatenates along feature dimension.
        features = self.fusion(fused)  # Fuses through fusion module.

        # Add temporal encoding
        features = self.temporal_pe(features)  # Adds positional encoding.

        # BiGRU processing
        packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True,
                                     enforce_sorted=False)  # Packs the sequence for efficient RNN processing, moving lengths to CPU.
        packed_output, _ = self.gru(packed)  # Passes through GRU, ignoring hidden state.
        output, _ = pad_packed_sequence(packed_output, batch_first=True)  # Unpacks the output.

        # Output projection
        logits = self.output(output)  # Projects to logits.
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)  # Applies log softmax and transposes to (seq, batch, vocab) for CTC.

        return log_probs  # Returns log probabilities in CTC format.


# ============================================================================
# 🔧 OPTIMIZER CONFIGURATIONS
# ============================================================================

OPTIMIZER_CONFIGS = {  # Defines a global dictionary OPTIMIZER_CONFIGS mapping optimizer names to their configurations.
    'adam': {  # Configuration for 'adam'.
        'name': 'Adam',  # Display name.
        'class': torch.optim.Adam,  # Optimizer class.
        'params': {'lr': 0.001, 'betas': (0.9, 0.999)},  # Hyperparameters: learning rate and betas.
        'description': 'Standard Adam optimizer'  # Description.
    },
    'adamw': {  # Configuration for 'adamw'.
        'name': 'AdamW',  # Display name.
        'class': torch.optim.AdamW,  # Optimizer class.
        'params': {'lr': 0.001, 'weight_decay': 0.01, 'betas': (0.9, 0.999)},  # Includes weight decay.
        'description': 'AdamW with weight decay (L2 regularization)'  # Description.
    },
    'adamw_strong': {  # Configuration for stronger regularization.
        'name': 'AdamW_Strong',  # Display name.
        'class': torch.optim.AdamW,  # Class.
        'params': {'lr': 0.001, 'weight_decay': 0.05, 'betas': (0.9, 0.999)},  # Higher weight decay.
        'description': 'AdamW with strong regularization'  # Description.
    },
    'adamw_low_lr': {  # Configuration for lower learning rate.
        'name': 'AdamW_LowLR',  # Display name.
        'class': torch.optim.AdamW,  # Class.
        'params': {'lr': 0.0005, 'weight_decay': 0.01, 'betas': (0.9, 0.999)},  # Lower LR.
        'description': 'AdamW with lower learning rate (more stable)'  # Description.
    },
    'signbert_paper': {  # Configuration matching the SignBERT paper.
        'name': 'SignBERT_Paper',  # Display name.
        'class': torch.optim.AdamW,  # Class.
        'params': {'lr': 0.0001, 'weight_decay': 0.01, 'betas': (0.9, 0.999)},  # Paper-specific LR.
        'description': 'Exact config from SignBERT+ paper'  # Description.
    }
}


def get_optimizer(model, config_name: str):  # Defines a function to retrieve an optimizer instance, taking model and config_name.
    """Get optimizer from configuration"""  # Docstring.
    if config_name not in OPTIMIZER_CONFIGS:  # Checks if config_name is valid.
        raise ValueError(f"Unknown optimizer: {config_name}")  # Raises ValueError if invalid.

    config = OPTIMIZER_CONFIGS[config_name]  # Retrieves the config dict.
    return config['class'](model.parameters(), **config['params'])  # Instantiates the optimizer with model params and config params.


# ============================================================================
# 🎯 TRAINER WITH MLFLOW INTEGRATION
# ============================================================================

class MLflowTrainer:  # Defines a class MLflowTrainer for handling training with MLflow integration.
    """Trainer with full MLflow tracking"""  # Docstring.

    def __init__(self, model, optimizer_name: str, device='cuda'):  # Initializer taking model, optimizer_name, and optional device (default 'cuda').
        self.model = model.to(device)  # Moves the model to the specified device.
        self.device = device  # Stores the device.
        self.optimizer_name = optimizer_name  # Stores the optimizer name.
        self.history = defaultdict(list)  # Initializes history as defaultdict of lists for metrics.
        self.criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)  # Sets CTC loss with blank=0, mean reduction, handling inf/NaN.

        print(f"🚀 MLflowTrainer initialized")  # Prints initialization message.
        print(f"   Device: {device}")  # Prints device.
        print(f"   Optimizer: {optimizer_name}")  # Prints optimizer.
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")  # Prints total model parameters with comma formatting.

    def train_epoch(self, dataloader, optimizer, epoch):  # Defines method to train one epoch, taking dataloader, optimizer, and epoch number.
        """Train one epoch with MLflow logging"""  # Docstring.
        self.model.train()  # Sets model to training mode.
        total_loss = 0  # Initializes total loss accumulator.
        num_batches = 0  # Initializes batch counter.

        for batch_idx, (landmarks, glosses, lengths, gloss_lengths) in enumerate(dataloader):  # Loops over batches in dataloader.
            landmarks = landmarks.to(self.device)  # Moves landmarks to device.
            lengths = lengths.to(self.device)  # Moves lengths to device.
            gloss_lengths = gloss_lengths.to(self.device)  # Moves gloss_lengths to device.
            target = torch.cat(glosses).to(self.device)  # Concatenates glosses into target tensor and moves to device.

            # Forward
            log_probs = self.model(landmarks, lengths)  # Computes log probs through model.
            loss = self.criterion(log_probs, target, lengths, gloss_lengths)  # Computes CTC loss.

            if torch.isnan(loss) or torch.isinf(loss):  # Checks for NaN or inf loss.
                continue  # Skips the batch if invalid.

            # Backward
            optimizer.zero_grad()  # Clears gradients.
            loss.backward()  # Backpropagates.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)  # Clips gradients to max norm 5.0.
            optimizer.step()  # Updates parameters.

            total_loss += loss.item()  # Accumulates loss value.
            num_batches += 1  # Increments batch count.

            # Log batch metrics to MLflow
            if batch_idx % 10 == 0:  # Every 10 batches.
                step = epoch * len(dataloader) + batch_idx  # Computes global step.
                mlflow.log_metric("batch_loss", loss.item(), step=step)  # Logs batch loss to MLflow.

        avg_loss = total_loss / max(num_batches, 1)  # Computes average loss, avoiding div by zero.
        self.history['train_loss'].append(avg_loss)  # Appends to history.

        # Log epoch metrics
        mlflow.log_metric("train_loss", avg_loss, step=epoch)  # Logs epoch train loss.

        return avg_loss  # Returns average loss.

    def evaluate(self, dataloader, epoch):  # Defines evaluation method, taking dataloader and epoch.
        """Evaluate with MLflow logging"""  # Docstring.
        self.model.eval()  # Sets model to eval mode.
        total_loss = 0  # Initializes total loss.
        num_batches = 0  # Initializes batch counter.

        with torch.no_grad():  # Disables gradient computation.
            for landmarks, glosses, lengths, gloss_lengths in dataloader:  # Loops over batches.
                landmarks = landmarks.to(self.device)  # To device.
                lengths = lengths.to(self.device)  # To device.
                gloss_lengths = gloss_lengths.to(self.device)  # To device.
                target = torch.cat(glosses).to(self.device)  # Concat target.

                log_probs = self.model(landmarks, lengths)  # Forward pass.
                loss = self.criterion(log_probs, target, lengths, gloss_lengths)  # Compute loss.

                if not (torch.isnan(loss) or torch.isinf(loss)):  # Skip invalid losses.
                    total_loss += loss.item()  # Accumulate.
                    num_batches += 1  # Increment.

        avg_loss = total_loss / max(num_batches, 1)  # Average loss.
        self.history['val_loss'].append(avg_loss)  # Append to history.

        # Log validation metrics
        mlflow.log_metric("val_loss", avg_loss, step=epoch)  # Log to MLflow.

        return avg_loss  # Return average.


# ============================================================================
# 🚀 MAIN TRAINING FUNCTION
# ============================================================================

def train_signbert_with_mlflow(  # Defines the main training function.
    data_files: List[Path],  # List of Path objects for data files.
    optimizer_configs: List[str] = ['adamw'],  # Optional list of optimizer configs, default ['adamw'].
    num_epochs: int = 50,  # Default 50 epochs.
    batch_size: int = 32,  # Default batch size 32.
    use_augmentation: bool = True,  # Default augmentation enabled.
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'  # Auto-detects device.
):  # Function definition ends.
    """
    Train SignBERT model with MLflow tracking

    Args:
        data_files: List of NPZ files
        optimizer_configs: List of optimizer names to compare
        num_epochs: Training epochs
        batch_size: Batch size
        use_augmentation: Enable data augmentation
        device: Training device
    """  # Multi-line docstring with args.
    # Setup MLflow
    setup_mlflow()  # Calls setup_mlflow to configure tracking.

    # Create dataset
    dataset = PhoenixDataset(data_files, augment=use_augmentation)  # Instantiates dataset with augmentation flag.
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=collate_fn)  # Creates training DataLoader with shuffling.
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn)  # Creates validation DataLoader without shuffling (note: uses same dataset, so train/val split is random).

    vocab_size = len(dataset.gloss_to_idx)  # Gets vocabulary size.

    print("\n" + "="*70)  # Prints separator.
    print("🎯 SIGNBERT TRAINING WITH MLFLOW")  # Prints header.
    print("="*70)  # Separator.
    print(f"Dataset: {len(data_files)} samples")  # Prints sample count.
    print(f"Vocabulary: {vocab_size} glosses")  # Prints vocab size.
    print(f"Optimizers to test: {optimizer_configs}")  # Prints optimizers.
    print(f"Epochs: {num_epochs}")  # Prints epochs.
    print(f"Batch size: {batch_size}")  # Prints batch size.
    print(f"Augmentation: {use_augmentation}")  # Prints augmentation status.
    print(f"Device: {device}")  # Prints device.
    print("="*70)  # Separator.

    all_results = {}  # Initializes dict for results.

    # Train with each optimizer
    for opt_name in optimizer_configs:  # Loops over each optimizer config.
        print(f"\n{'='*70}")  # Separator.
        print(f"🔥 TRAINING WITH: {OPTIMIZER_CONFIGS[opt_name]['name']}")  # Prints optimizer name.
        print(f"{'='*70}")  # Separator.
        print(f"Description: {OPTIMIZER_CONFIGS[opt_name]['description']}")  # Prints description.

        # Start MLflow run
        with mlflow.start_run(run_name=f"SignBERT_{opt_name}"):  # Starts an MLflow run with name.

            # Log parameters
            mlflow.log_param("model", "SignBERT_BiGRU")  # Logs model type.
            mlflow.log_param("optimizer", opt_name)  # Logs optimizer key.
            mlflow.log_param("optimizer_name", OPTIMIZER_CONFIGS[opt_name]['name'])  # Logs display name.
            mlflow.log_param("batch_size", batch_size)  # Logs batch size.
            mlflow.log_param("epochs", num_epochs)  # Logs epochs.
            mlflow.log_param("num_samples", len(data_files))  # Logs sample count.
            mlflow.log_param("vocab_size", vocab_size)  # Logs vocab size.
            mlflow.log_param("augmentation", use_augmentation)  # Logs augmentation.
            mlflow.log_param("device", device)  # Logs device.

            # Log optimizer specific params
            for key, value in OPTIMIZER_CONFIGS[opt_name]['params'].items():  # Loops over optimizer params.
                mlflow.log_param(f"opt_{key}", value)  # Logs each as "opt_key": value.

            # Create model
            model = SignBERTBiGRU(vocab_size=vocab_size)  # Instantiates model with vocab_size.
            mlflow.log_param("model_params", sum(p.numel() for p in model.parameters()))  # Logs total params.

            # Create trainer
            trainer = MLflowTrainer(model, opt_name, device)  # Instantiates trainer.

            # Create optimizer
            optimizer = get_optimizer(model, opt_name)  # Gets optimizer instance.

            # Scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # Creates learning rate scheduler.
                optimizer, mode='min', factor=0.5, patience=3  # Reduces on plateau, halve LR after 3 epochs without improvement.
            )
            mlflow.log_param("scheduler", "ReduceLROnPlateau")  # Logs scheduler type.
            mlflow.log_param("scheduler_patience", 3)  # Logs patience.

            # Training loop
            best_val_loss = float('inf')  # Initializes best validation loss to infinity.
            start_time = time.time()  # Records start time.

            for epoch in range(1, num_epochs + 1):  # Loops over epochs from 1 to num_epochs.
                print(f"\n📅 Epoch {epoch}/{num_epochs}")  # Prints epoch progress.

                # Train
                train_loss = trainer.train_epoch(train_loader, optimizer, epoch)  # Trains one epoch.
                val_loss = trainer.evaluate(val_loader, epoch)  # Evaluates.

                print(f"   Train Loss: {train_loss:.4f}")  # Prints train loss.
                print(f"   Val Loss:   {val_loss:.4f}")  # Prints val loss.

                # Scheduler step
                old_lr = optimizer.param_groups[0]['lr']  # Gets current LR.
                scheduler.step(val_loss)  # Steps scheduler with val_loss.
                new_lr = optimizer.param_groups[0]['lr']  # Gets new LR.

                # Log learning rate
                mlflow.log_metric("learning_rate", new_lr, step=epoch)  # Logs LR.

                if new_lr != old_lr:  # If LR changed.
                    print(f"   📉 Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")  # Prints change.

                # Save best model
                if val_loss < best_val_loss:  # If new best.
                    best_val_loss = val_loss  # Updates best.
                    print(f"   ✅ New best! Val Loss: {val_loss:.4f}")  # Prints message.

                    # Save model to MLflow with signature
                    # Get a sample batch for signature inference
                    trainer.model.eval()  # Sets to eval.
                    with torch.no_grad():  # No grad.
                        for sample_batch in val_loader:  # Gets one batch.
                            sample_landmarks, _, sample_lengths, _ = sample_batch  # Unpacks.
                            sample_input = sample_landmarks[:1].cpu()  # Takes first sample, to CPU.
                            sample_len = sample_lengths[:1]  # Takes length.

                            # Generate sample output
                            sample_output = trainer.model(  # Forward on sample.
                                sample_input.to(trainer.device),
                                sample_len.to(trainer.device)
                            ).cpu().detach().numpy()  # To CPU, detach, to numpy.

                            # Log model with signature
                            mlflow.pytorch.log_model(  # Logs PyTorch model.
                                trainer.model,
                                artifact_path="model",
                                signature=mlflow.models.infer_signature(  # Infers signature from sample input/output.
                                    sample_input.numpy(),
                                    sample_output
                                )
                            )
                            break  # Breaks after one batch.
                    trainer.model.train()  # Back to train mode.

            train_time = time.time() - start_time  # Computes total time.

            # Log final metrics
            mlflow.log_metric("best_val_loss", best_val_loss)  # Logs best val loss.
            mlflow.log_metric("final_train_loss", trainer.history['train_loss'][-1])  # Logs final train loss.
            mlflow.log_metric("training_time_seconds", train_time)  # Logs time in seconds.
            mlflow.log_metric("training_time_minutes", train_time / 60)  # Logs in minutes.

            # Log training curves as artifact
            fig = plot_training_curve(trainer.history)  # Plots curve.
            mlflow.log_figure(fig, f"training_curve_{opt_name}.png")  # Logs figure as artifact.
            plt.close(fig)  # Closes plot.

            # Store results
            all_results[opt_name] = {  # Stores in all_results.
                'best_val_loss': best_val_loss,
                'train_time': train_time,
                'history': trainer.history
            }

            print(f"\n✅ {opt_name} completed:")  # Completion message.
            print(f"   Best Val Loss: {best_val_loss:.4f}")  # Best loss.
            print(f"   Training time: {train_time/60:.1f} minutes")  # Time.

    # Print comparison
    print("\n" + "="*70)  # Separator.
    print("📊 OPTIMIZER COMPARISON")  # Header.
    print("="*70)  # Separator.

    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['best_val_loss'])  # Sorts results by best val loss ascending.

    for i, (name, result) in enumerate(sorted_results, 1):  # Loops over sorted results with index.
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "  # Assigns medal emoji based on rank.
        print(f"{emoji} {i}. {OPTIMIZER_CONFIGS[name]['name']:<20} "
              f"Loss: {result['best_val_loss']:.4f}  "
              f"Time: {result['train_time']/60:.1f}min")  # Prints ranked comparison.

    print("="*70)  # Final separator.

    return all_results  # Returns all results dict.


def plot_training_curve(history):  # Defines a function to plot the training curve, taking history dict.
    """Plot training curve"""  # Docstring.
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))  # Creates figure and axis with size 10x6.

    epochs = range(1, len(history['train_loss']) + 1)  # Creates epoch range.
    ax.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)  # Plots train loss with blue line and markers.
    ax.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2)  # Plots val loss with red.

    ax.set_xlabel('Epoch', fontsize=12)  # Sets x-label.
    ax.set_ylabel('Loss', fontsize=12)  # Sets y-label.
    ax.set_title('Training Progress', fontsize=14, fontweight='bold')  # Sets title.
    ax.legend(fontsize=10)  # Adds legend.
    ax.grid(True, alpha=0.3)  # Adds grid.

    plt.tight_layout()  # Adjusts layout.
    return fig  # Returns figure.


# ============================================================================
# 🎯 ENTRY POINT
# ============================================================================

if __name__ == "__main__":  # Checks if the script is run directly (not imported).
    print("🎬 SignBERT Production Training with MLflow")  # Prints title.
    print("\nUsage:")  # Prints usage header.
    print("""
from pathlib import Path
from signbert_mlflow_training import train_signbert_with_mlflow

# Load data
BASE = Path("path/to/LandmarksPhoenixDataset")
train_files = list((BASE / "landmarks_train").glob("*.npz"))

# Train with optimizer comparison
results = train_signbert_with_mlflow(
    data_files=train_files[:500],  # Start with subset
    optimizer_configs=['adam', 'adamw', 'adamw_low_lr', 'signbert_paper'],
    num_epochs=20,
    batch_size=16,
    use_augmentation=True
)

# View results in MLflow UI:
# https://mlflow.schlaepfer.me
""")  # Prints a multi-line usage example as a string literal, showing how to import and use the training function, and view results.