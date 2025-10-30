import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import CTCLoss
import glob
import json
import math
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from torchinfo import summary
import platform
import psutil

# ==================== DATASET ====================

class LandmarkDataset(Dataset):
    """Dataset loader for preprocessed landmarks"""
    def __init__(self, landmarks_dir, vocab_file=None, build_vocab=False, 
                 augment=False, all_dirs_for_vocab=None, max_samples=None, random_subset=False):
        self.landmarks_dir = landmarks_dir
        self.samples = sorted(glob.glob(os.path.join(landmarks_dir, "*.npz")))
        self.augment = augment

        # Initialize augmentation
        if self.augment:
            self.temporal_aug = TemporalAugmentation(speed_range=(0.85, 1.15), prob=0.5)

        # Limit samples
        if max_samples is not None:
            if random_subset:
                # Random subset for better representation
                random.seed(seed)
                self.samples = random.sample(self.samples, 
                                           min(max_samples, len(self.samples)))
                print(f"Random subset: {len(self.samples)} samples (seed={seed})")
            else:
                # First N samples
                self.samples = self.samples[:max_samples]
                print(f"First {len(self.samples)} samples")
                
                
        if build_vocab:
            # Build vocab from all directories if provided
            if all_dirs_for_vocab:
                self.gloss_vocab = self._build_vocab_from_multiple_dirs(all_dirs_for_vocab)
            else:
                self.gloss_vocab = self._build_vocab()
            self._save_vocab(vocab_file)
        else:
            self.gloss_vocab = self._load_vocab(vocab_file)

        self.idx_to_gloss = {v: k for k, v in self.gloss_vocab.items()}
        
    def _build_vocab_from_multiple_dirs(self, directories):
        """Build vocabulary from multiple directories (train, dev, test)"""
        glosses = set()
        for directory in directories:
            samples = sorted(glob.glob(os.path.join(directory, "*.npz")))
            for sample_path in tqdm(samples, desc=f"Building vocab from {directory}"):
                data = np.load(sample_path, allow_pickle=True)
                glosses.update(data['glosses'])

        gloss_to_idx = {
            '<pad>': 0, 
            '<blank>': 1, 
            '<sos>': 2, 
            '<eos>': 3,
            '<unk>': 4
        }
        
        for idx, gloss in enumerate(sorted(glosses)):
            gloss_to_idx[gloss] = idx + 5

        print(f"Vocabulary size: {len(gloss_to_idx)} (from {len(directories)} directories)")
        return gloss_to_idx

    def _build_vocab(self):
        """Build vocabulary from all glosses in dataset"""
        glosses = set()
        for sample_path in tqdm(self.samples, desc="Building vocab"):
            data = np.load(sample_path, allow_pickle=True)
            glosses.update(data['glosses'])

        gloss_to_idx = {
            '<pad>': 0, 
            '<blank>': 1, 
            '<sos>': 2, 
            '<eos>': 3,
            '<unk>': 4
        }
        
        for idx, gloss in enumerate(sorted(glosses)):
            gloss_to_idx[gloss] = idx + 5

        print(f"Vocabulary size: {len(gloss_to_idx)}")
        return gloss_to_idx

    def _save_vocab(self, vocab_file):
        """Save vocabulary to JSON file"""
        if vocab_file:
            with open(vocab_file, 'w') as f:
                json.dump(self.gloss_vocab, f, indent=2)
            print(f"Vocabulary saved to {vocab_file}")

    def _load_vocab(self, vocab_file):
        """Load vocabulary from JSON file"""
        with open(vocab_file, 'r') as f:
            vocab = json.load(f)
        print(f"Vocabulary loaded from {vocab_file}, size: {len(vocab)}")
        return vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx], allow_pickle=True)
        landmarks = data['landmarks']

        # Apply augmentation during training
        if self.augment:
            # Temporal augmentation
            landmarks = self.temporal_aug(landmarks)

            # Spatial noise
            if np.random.random() > 0.5:
                landmarks = add_spatial_noise(landmarks, noise_std=0.005)

        landmarks = torch.FloatTensor(landmarks)
        
        # Handle unknown glosses safely
        glosses = []
        for g in data['glosses']:
            g_str = str(g)
            glosses.append(self.gloss_vocab.get(g_str, self.gloss_vocab['<unk>']))
        
        return landmarks, torch.LongTensor(glosses)


def collate_fn(batch):
    """Collate function for variable-length sequences"""
    landmarks, glosses = zip(*batch)

    # Pad landmarks to max length in batch
    max_len = max(lm.shape[0] for lm in landmarks)
    feature_dim = landmarks[0].shape[1]
    padded_landmarks = torch.zeros(len(landmarks), max_len, feature_dim)
    landmark_lengths = []

    for i, lm in enumerate(landmarks):
        padded_landmarks[i, :lm.shape[0]] = lm
        landmark_lengths.append(lm.shape[0])

    # Pad glosses
    max_gloss_len = max(len(g) for g in glosses)
    padded_glosses = torch.zeros(len(glosses), max_gloss_len).long()
    gloss_lengths = []

    for i, g in enumerate(glosses):
        padded_glosses[i, :len(g)] = g
        gloss_lengths.append(len(g))

    return (padded_landmarks, torch.LongTensor(landmark_lengths),
            padded_glosses, torch.LongTensor(gloss_lengths))


# ==================== MODEL ====================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Use register_buffer so pe moves with model.to(device)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # self.pe is now automatically on the correct device
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SignLanguageTranslator(nn.Module):
    def __init__(self, input_dim, num_glosses, d_model=768, nhead=12,
                 num_encoder_layers=12, dropout=0.4):  # Increased dropout
        super().__init__()

        # Input projection with BatchNorm
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.BatchNorm1d(d_model),  # Use BatchNorm instead of LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Temporal convolution with BatchNorm
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),  # Added
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ... rest remains same but use GELU instead of ReLU ...
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',  # Use GELU instead of relu
            batch_first=True,
            norm_first=True
        )
        
        self.ctc_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_glosses)
        )

        self.d_model = d_model

    def forward(self, src, src_lengths):
        # Input projection: [B, T, input_dim] -> [B, T, d_model]
        x = self.input_proj(src)

        # Temporal convolution
        x_conv = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv

        # Positional encoding
        x = self.pos_encoder(x)

        # Create padding mask
        src_mask = self._generate_padding_mask(src_lengths, src.size(1)).to(src.device)

        # Encode
        memory = self.encoder(x, src_key_padding_mask=src_mask)

        # CTC prediction
        ctc_logits = self.ctc_head(memory)

        return ctc_logits

    def _generate_padding_mask(self, lengths, max_len):
        """Generate padding mask for variable length sequences"""
        batch_size = len(lengths)
        # FIX: Explicitly specify device to match input tensor
        mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths.unsqueeze(1)
        return mask


# ==================== DATA AUGMENTATION ====================

class TemporalAugmentation:
    """Temporal data augmentation"""
    def __init__(self, speed_range=(0.85, 1.15), prob=0.5):
        self.speed_range = speed_range
        self.prob = prob

    def __call__(self, landmarks):
        if np.random.random() > self.prob:
            return landmarks

        # Random temporal scaling
        speed = np.random.uniform(*self.speed_range)
        num_frames = landmarks.shape[0]
        new_num_frames = max(1, int(num_frames / speed))

        indices = np.linspace(0, num_frames - 1, new_num_frames)
        augmented = np.array([landmarks[min(int(i), num_frames-1)] for i in indices])

        return augmented


def add_spatial_noise(landmarks, noise_std=0.005):
    """Add Gaussian noise to landmarks"""
    noise = np.random.normal(0, noise_std, landmarks.shape)
    return landmarks + noise


# ==================== TRAINING ====================

def decode_predictions(ctc_output, lengths, vocab, blank_id=1):
    """Decode CTC output using greedy decoding"""
    batch_size = ctc_output.size(0)
    predictions = torch.argmax(ctc_output, dim=-1)

    decoded = []
    for i in range(batch_size):
        pred = predictions[i, :lengths[i]]

        # Remove blanks and repeated tokens
        pred_seq = []
        prev = None
        for p in pred:
            p = p.item()
            if p != blank_id and p != prev:
                pred_seq.append(p)
            prev = p

        # Convert to glosses
        glosses = [vocab.get(p, '<unk>') for p in pred_seq]
        decoded.append(glosses)

    return decoded


def compute_wer(predictions, targets):
    """Compute Word Error Rate"""
    errors = 0
    total_words = 0

    for pred, tgt in zip(predictions, targets):
        # Simple edit distance calculation
        pred_words = pred if isinstance(pred, list) else pred.split()
        tgt_words = tgt if isinstance(tgt, list) else tgt.split()

        d = [[0] * (len(tgt_words) + 1) for _ in range(len(pred_words) + 1)]

        for i in range(len(pred_words) + 1):
            d[i][0] = i
        for j in range(len(tgt_words) + 1):
            d[0][j] = j

        for i in range(1, len(pred_words) + 1):
            for j in range(1, len(tgt_words) + 1):
                if pred_words[i-1] == tgt_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1

        errors += d[len(pred_words)][len(tgt_words)]
        total_words += len(tgt_words)

    return errors / max(total_words, 1)


def train_epoch(model, train_loader, optimizer, criterion, device, vocab, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (landmarks, landmark_lengths, glosses, gloss_lengths) in enumerate(pbar):
        landmarks = landmarks.to(device)
        glosses = glosses.to(device)
        landmark_lengths = landmark_lengths.to(device)
        gloss_lengths = gloss_lengths.to(device)

        optimizer.zero_grad()

        # Forward pass
        ctc_logits = model(landmarks, landmark_lengths)

        # CTC loss expects [T, B, C] format and log probabilities
        ctc_logits = ctc_logits.transpose(0, 1)
        log_probs = F.log_softmax(ctc_logits, dim=-1)

        # Calculate CTC loss
        loss = criterion(log_probs, glosses, landmark_lengths, gloss_lengths)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Log batch loss to MLflow every 50 batches
        if batch_idx % 50 == 0:
            step = epoch * len(train_loader) + batch_idx
            mlflow.log_metric("batch_loss", loss.item(), step=step)

    return total_loss / num_batches


def validate(model, val_loader, criterion, device, vocab, idx_to_gloss):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for landmarks, landmark_lengths, glosses, gloss_lengths in pbar:
            landmarks = landmarks.to(device)
            glosses = glosses.to(device)
            landmark_lengths = landmark_lengths.to(device)
            gloss_lengths = gloss_lengths.to(device)

            # Forward pass
            ctc_logits = model(landmarks, landmark_lengths)

            # Calculate loss
            ctc_logits_t = ctc_logits.transpose(0, 1)
            log_probs = F.log_softmax(ctc_logits_t, dim=-1)
            loss = criterion(log_probs, glosses, landmark_lengths, gloss_lengths)

            total_loss += loss.item()
            num_batches += 1

            # Decode predictions
            predictions = decode_predictions(ctc_logits, landmark_lengths, idx_to_gloss)

            # Convert targets to glosses
            for i in range(glosses.size(0)):
                target = glosses[i, :gloss_lengths[i]].cpu().numpy()
                target_glosses = [idx_to_gloss.get(int(t), '<unk>') for t in target]
                all_predictions.append(predictions[i])
                all_targets.append(target_glosses)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    wer = compute_wer(all_predictions, all_targets)

    return avg_loss, wer


def train_model(model, train_loader, val_loader, num_epochs, device, vocab, idx_to_gloss, save_dir='checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    
    criterion = CTCLoss(blank=1, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
    
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    best_wer = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, vocab, epoch)
        print(f"Train Loss: {train_loss:.4f}")

        val_loss, val_wer = validate(model, val_loader, criterion, device, vocab, idx_to_gloss)
        print(f"Val Loss: {val_loss:.4f}, Val WER: {val_wer:.4f}")

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_wer": val_wer,
        }, step=epoch)

        if val_wer < best_wer:
            best_wer = val_wer
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_wer': val_wer,
            }
            #torch.save(checkpoint, os.path.join(save_dir, 'best_model.pt'))
            #print(f"✓ Best model saved: WER {best_wer:.4f}")

        scheduler.step()

        # Early stopping
        if early_stopping(val_wer, epoch):
            break

    return best_wer


def generate_model_summary(model, input_dim, device, batch_size=8, seq_length=100):
    """Generate comprehensive model summary"""
    print("\n" + "="*70)
    print("MODEL SUMMARY")
    print("="*70)
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, seq_length, input_dim).to(device)
    dummy_lengths = torch.tensor([seq_length] * batch_size).to(device)
    
    # Get torchinfo summary
    model_stats = summary(
        model,
        input_data=(dummy_input, dummy_lengths),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
        verbose=0
    )
    
    # Print summary
    print(model_stats)
    
    # Additional statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    
    print("\n" + "="*70)
    print("DETAILED STATISTICS")
    print("="*70)
    print(f"Total Parameters:           {total_params:>15,}")
    print(f"Trainable Parameters:       {trainable_params:>15,}")
    print(f"Non-trainable Parameters:   {non_trainable_params:>15,}")
    print(f"Model Size:                 {size_mb:>15.2f} MB")
    print(f"Input Shape:                {tuple(dummy_input.shape)}")
    print(f"Output Shape:               {model(dummy_input, dummy_lengths).shape}")
    print("="*70 + "\n")
    
    # Return summary as string for logging
    summary_str = str(model_stats)
    
    # Create detailed summary dictionary
    summary_dict = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "model_size_mb": size_mb,
        "input_shape": list(dummy_input.shape),
        "output_shape": list(model(dummy_input, dummy_lengths).shape)
    }
    
    return summary_str, summary_dict

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_wer = float('inf')
        self.best_epoch = 0

    def __call__(self, val_wer, epoch):
        if val_wer < (self.best_val_wer - self.min_delta):
            self.best_val_wer = val_wer
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                print(f"Best WER was {self.best_val_wer:.4f} at epoch {self.best_epoch}")
                return True
        return False

# ==================== MAIN ====================

def main():
    # Configuration
    LANDMARKS_TRAIN = "./landmarks_train"
    LANDMARKS_DEV = "./landmarks_dev"
    VOCAB_FILE = "vocab.json"
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    INPUT_DIM = 1659  # 126 (hands) + 1434 (face) + 99 (pose)
    D_MODEL = 512      # Reduced from 768
    NHEAD = 8          # Reduced from 12
    NUM_LAYERS = 8     # Reduced from 12
    DROPOUT = 0.35     # But keep dropout high
    
    # MLflow configuration
    EXPERIMENT_NAME = "SignNetAdvanced++"
    RUN_NAME = "transformer_ctc_baseline_increased_model_size"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.environ['MLFLOW_TRACKING_USERNAME'] = 'roman'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'SignNet'
    mlflow.set_tracking_uri("https://mlflow.schlaepfer.me")

    # Set MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    
    # Start MLflow run
    with mlflow.start_run(log_system_metrics=True, run_name=RUN_NAME):
        # System information
        mlflow.log_param("python_version", platform.python_version())
        mlflow.log_param("pytorch_version", torch.__version__)
        mlflow.log_param("os", platform.system())
        mlflow.log_param("cpu_count", os.cpu_count())
        mlflow.log_param("total_ram_gb", round(psutil.virtual_memory().total / (1024**3), 2))

        # GPU details
        if torch.cuda.is_available():
            mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
            mlflow.log_param("gpu_count", torch.cuda.device_count())
            mlflow.log_param("cuda_version", torch.version.cuda)
            mlflow.log_param("cudnn_version", torch.backends.cudnn.version())
            mlflow.log_param("gpu_memory_gb", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
            
            
        # Log hyperparameters
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "input_dim": INPUT_DIM,
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "optimizer": "Adam",
            "learning_rate": 1e-4,
            "scheduler": "ReduceLROnPlateau",
            "augmentation": "temporal+spatial",
            "loss_function": "CTC"
        })
        
        # Log device info
        mlflow.log_param("device", str(device))
        mlflow.log_param("cuda_available", torch.cuda.is_available())
        if torch.cuda.is_available():
            mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))

        # Create datasets
        print("\n" + "="*50)
        print("Loading Datasets")
        print("="*50)

        # Create datasets with augmentation
        train_dataset = LandmarkDataset(
            LANDMARKS_TRAIN,
            vocab_file=VOCAB_FILE,
            build_vocab=True,
            augment=True,  # Enable augmentation for training
            # max_samples=500,
            # random_subset=False
        )

        val_dataset = LandmarkDataset(
            LANDMARKS_DEV,
            vocab_file=VOCAB_FILE,
            build_vocab=False,
            augment=False  # No augmentation for validation
        )
        
        # Log dataset info
        mlflow.log_params({
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "vocab_size": len(train_dataset.gloss_vocab)
        })
        
        # Log vocabulary file as artifact
        mlflow.log_artifact(VOCAB_FILE)

        use_pin_memory = torch.cuda.is_available()
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=use_pin_memory
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=use_pin_memory
        )

        # Create model
        print("\n" + "="*50)
        print("Creating Model")
        print("="*50)

        num_glosses = len(train_dataset.gloss_vocab)
        model = SignLanguageTranslator(
            input_dim=INPUT_DIM,
            num_glosses=num_glosses,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_encoder_layers=NUM_LAYERS,
            dropout=DROPOUT
        ).to(device)

        # Generate and log model summary
        summary_str, summary_dict = generate_model_summary(
            model, 
            INPUT_DIM, 
            device, 
            batch_size=BATCH_SIZE,
            seq_length=100
        )
        
        # Log model statistics to MLflow
        mlflow.log_params(summary_dict)
        
        # Save model summary to file and log as artifact
        model_summary_path = "model_summary.txt"
        with open(model_summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_str)
            f.write("\n\n" + "="*70 + "\n")
            f.write("DETAILED STATISTICS\n")
            f.write("="*70 + "\n")
            for key, value in summary_dict.items():
                f.write(f"{key}: {value}\n")
        mlflow.log_artifact(model_summary_path)

        # Train model
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)

        best_wer = train_model(
            model,
            train_loader,
            val_loader,
            NUM_EPOCHS,
            device,
            train_dataset.gloss_vocab,
            train_dataset.idx_to_gloss
        )

        print(f"\n{'='*50}")
        print(f"Training Complete! Best WER: {best_wer:.4f}")
        print(f"{'='*50}")
        
        # Log final best WER
        mlflow.log_metric("final_best_wer", best_wer)
        
        # Log the final model
        mlflow.pytorch.log_model(model, "model")
        
        # Set tags for easy filtering
        mlflow.set_tags({
            "model_type": "transformer",
            "task": "sign_language_translation",
            "dataset": "phoenix-2014",
            "status": "completed"
        })


if __name__ == "__main__":
    main()
