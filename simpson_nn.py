import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import Counter
import kagglehub
import subprocess
import argparse

try:
    import pynvml
    pynvml_available = True
except ImportError:
    pynvml_available = False

# Install pynvml if not already installed
if not pynvml_available:
    print("Installing pynvml for GPU utilization monitoring...")
    subprocess.run(["pip3", "install", "pynvml"])

# Verify PyTorch setup
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("Error: CUDA not available. Please ensure a GPU-enabled Vast.ai instance.")
    exit(1)

# Download latest version of the dataset
path = kagglehub.dataset_download("alexattia/the-simpsons-characters-dataset")
print("Path to dataset files:", path)

# Configuration
IMG_SIZE = 224  # For EfficientNet-B0
DATA_DIR = os.path.join(path, "simpsons_dataset")
MAX_IMAGES_PER_CLASS = 2000
NUM_EPOCHS_TUNING = 5
NUM_EPOCHS = 20
INITIAL_LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "simpsons_efficientnet_b0.pth")
BATCH_SIZES = [32, 64, 128, 256]

# Create model directory
os.makedirs(MODEL_DIR, exist_ok=True)

# Get GPU utilization using nvidia-smi
def get_gpu_utilization_nvidia_smi():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv"])
        utilization = output.decode().split('\n')[1].split()[0].replace('%', '')
        return int(utilization)
    except:
        return -1

# Get GPU utilization using pynvml
def get_gpu_utilization_pynvml():
    if not pynvml_available:
        return -1
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        pynvml.nvmlShutdown()
        return util.gpu
    except:
        return -1

def print_gpu_memory(device):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f'GPU Memory: Allocated {allocated:.2f} MB, Reserved {reserved:.2f} MB, Max Allocated {max_allocated:.2f} MB')
    else:
        print('GPU not available, running on CPU')

# Load data
def load_data():
    label_names = sorted([name for name in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, name))])
    label_map = {name: idx for idx, name in enumerate(label_names)}
    image_counts = {label: len(os.listdir(os.path.join(DATA_DIR, label))) for label in label_names}
    top_characters = list(image_counts.keys())

    images, labels = [], []
    for label in top_characters:
        folder_path = os.path.join(DATA_DIR, label)
        for i, img_name in enumerate(os.listdir(folder_path)):
            if i >= MAX_IMAGES_PER_CLASS:
                break
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(label_map[label])

    X = np.array(images) / 255.0
    labels = np.array(labels)
    return X, labels, label_map, top_characters

# Custom Dataset class
class SimpsonsDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray((image * 255).astype(np.uint8))
        if self.transform:
            image = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image, label_tensor

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Train and evaluate function
def train_and_evaluate(batch_size, initial_lr, num_epochs, label_map, train_dataset, test_dataset, save_path=None):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.IMAGENET1K_V1')
    num_classes = len(label_map)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, min_lr=1e-6)

    model.train()
    best_val_accuracy = 0
    best_model_state = None
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

            util_smi = get_gpu_utilization_nvidia_smi()
            util_nvml = get_gpu_utilization_pynvml()

        epoch_loss = running_loss / len(train_dataset)
        print(f'Batch Size {batch_size}, Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        print_gpu_memory(DEVICE)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss = val_loss / len(test_dataset)
        val_accuracy = 100 * correct / total
        print(f'Batch Size {batch_size}, Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, '
              f'Validation Accuracy: {val_accuracy:.2f}%')
        print_gpu_memory(DEVICE)

        scheduler.step(val_loss)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
            if save_path:
                torch.save(best_model_state, save_path)
                print(f'Saved best model for Batch Size {batch_size} to {save_path}')

    return best_val_accuracy, best_model_state

def main():
    parser = argparse.ArgumentParser(description="Train or tune a model on the Simpsons dataset.")
    parser.add_argument('--mode', choices=['tune', 'train'], required=True,
                        help="Mode to run: 'tune' for batch size tuning, 'train' for training with a specific batch size.")
    parser.add_argument('--batch-size', type=int, default=None,
                        help="Batch size for training (required if mode is 'train').")
    args = parser.parse_args()

    # Load data
    X, labels, label_map, top_characters = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, stratify=labels, random_state=42
    )
    train_dataset = SimpsonsDataset(X_train, y_train, transform=transform)
    test_dataset = SimpsonsDataset(X_test, y_test, transform=transform)

    if args.mode == 'tune':
        results = {}
        best_accuracy = 0
        best_model_path = None
        best_model_state = None
        for batch_size in BATCH_SIZES:
            print(f'\nTuning with batch size {batch_size}, initial learning rate {INITIAL_LR:.6f}')
            temp_model_path = os.path.join(MODEL_DIR, f"temp_model_bs{batch_size}.pth")
            try:
                accuracy, model_state = train_and_evaluate(
                    batch_size, INITIAL_LR, NUM_EPOCHS_TUNING, label_map, train_dataset, test_dataset, temp_model_path
                )
                results[batch_size] = accuracy
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_path = temp_model_path
                    best_model_state = model_state
            except RuntimeError as e:
                print(f'Batch Size {batch_size} failed: {e} (likely out of memory)')
                results[batch_size] = None

        best_batch_size = max(results, key=lambda k: results[k] if results[k] is not None else -float('inf'))
        best_accuracy = results[best_batch_size]
        print(f'\nBest Batch Size: {best_batch_size}, Validation Accuracy: {best_accuracy:.2f}%')

        if best_model_path:
            model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.IMAGENET1K_V1')
            num_classes = len(label_map)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            model = model.to(DEVICE)
            model.load_state_dict(torch.load(best_model_path))
            print(f'Loaded best model from tuning at {best_model_path}')

    elif args.mode == 'train':
        if args.batch_size is None:
            print("Error: --batch-size is required when mode is 'train'.")
            exit(1)
        if args.batch_size not in BATCH_SIZES:
            print(f"Error: Batch size must be one of {BATCH_SIZES}.")
            exit(1)

        print(f'\nTraining final model with batch size {args.batch_size}, initial learning rate {INITIAL_LR:.6f}')
        accuracy, model_state = train_and_evaluate(
            args.batch_size, INITIAL_LR, NUM_EPOCHS, label_map, train_dataset, test_dataset, MODEL_PATH
        )
        if model_state:
            model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.IMAGENET1K_V1')
            num_classes = len(label_map)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            model = model.to(DEVICE)
            model.load_state_dict(model_state)
            torch.save(model_state, MODEL_PATH)
            print(f'Saved final model to {MODEL_PATH}')

    print(f"Loaded {len(X)} images across {len(top_characters)} characters")

if __name__ == "__main__":
    main()