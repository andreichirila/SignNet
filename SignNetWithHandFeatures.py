import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import mlflow
import mlflow.pytorch
import torch.nn.functional as F
import numpy as np
import random
from torchinfo import summary
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from SignNetCNN import SignNetCNN


DATASET_PATH = 'dataset_handfeatures'  # Set this to your extracted dataset path
TEST_INDICES_PATH = 'test_indices.json'

class DGSLandmarkDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)

        # Extract labels (first column)
        raw_labels = self.dataframe.iloc[:, 0].values.astype(str)

        # Create label encoder to convert string labels to integers
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(raw_labels)

        # Extract features (all columns after label)
        self.features = self.dataframe.iloc[:, 1:].values.astype(float)

        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        # Apply augmentation only if transform is defined and indicates training mode
        if self.transform is not None:
            feature = self.transform(feature)
        else:
            # Apply custom landmark augmentation here if no other transform is given
            feature = feature.numpy() if isinstance(feature, torch.Tensor) else feature
            feature = augment_landmarks(feature)

        # Convert to tensor
        feature = torch.tensor(feature, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return feature, label

def augment_landmarks(landmarks, noise_std=0.01, max_rotation_deg=10):
    """
    landmarks: np.array shape (63,) = (21 points * 3 coords)
    Adds noise and small random rotation to landmarks in XY plane.
    """
    lm = landmarks.reshape(21, 3).copy()

    # Add Gaussian noise to all coordinates
    lm += np.random.normal(0, noise_std, lm.shape)

    # Random rotation in radians
    angle = np.deg2rad(np.random.uniform(-max_rotation_deg, max_rotation_deg))
    cos_val, sin_val = np.cos(angle), np.sin(angle)

    # Rotation matrix for XY plane
    R = np.array([[cos_val, -sin_val],
                  [sin_val,  cos_val]])

    # Apply rotation to x,y only
    xy = lm[:, :2]
    xy_rot = np.dot(xy, R.T)
    lm[:, :2] = xy_rot

    return lm.flatten()

def get_device():
    if torch.cuda.is_available():
        print(f"Running on CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print(f"Running on mps")
        return torch.device("mps")
    else:
        print(f"Running on cpu")
        return torch.device("cpu")


def get_datasets_with_fixed_test_split():
    csv_path = os.path.join(DATASET_PATH, 'dataset.csv')  # Your single CSV file
    full_dataset = DGSLandmarkDataset(csv_path)

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    if not os.path.exists(TEST_INDICES_PATH):
        indices = list(range(total_size))
        splits = random_split(indices, [train_size, val_size, test_size])
        train_indices, val_indices, test_indices = splits[0].indices, splits[1].indices, splits[2].indices
        with open(TEST_INDICES_PATH, 'w') as f:
            json.dump(test_indices, f)
        print(f"Saved test indices to {TEST_INDICES_PATH}")
    else:
        with open(TEST_INDICES_PATH, 'r') as f:
            test_indices = json.load(f)
        train_val_indices = list(set(range(total_size)) - set(test_indices))
        train_val_size = len(train_val_indices)
        train_size = int(0.8 * train_val_size)
        val_size = train_val_size - train_size
        splits = random_split(train_val_indices, [train_size, val_size])
        train_indices, val_indices = splits[0].indices, splits[1].indices
        print(f"Loaded test indices from {TEST_INDICES_PATH}")

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


def train(model, optimizer, criterion, scheduler, train_loader, val_loader, device, epochs=30):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        total_batches = len(train_loader)
        for batch_idx, (features, labels) in enumerate(train_loader, 1):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_idx % 10 == 0 or batch_idx == total_batches:
                print(f"Epoch {epoch+1} [{batch_idx}/{total_batches}] - Batch Loss: {loss.item():.4f} - {(batch_idx / total_batches) * 100:.1f}% complete")
        avg_train_loss = train_loss / total_batches

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        mlflow.log_metrics({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_acc,
        }, step=epoch)

        for param_group in optimizer.param_groups:
            mlflow.log_metric("learning_rate", param_group['lr'], step=epoch)

        scheduler.step(avg_val_loss)

def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    mlflow.log_metrics({
        "test_loss": avg_loss,
        "test_accuracy": accuracy,
    })
    return avg_loss, accuracy
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(epochs=50, learning_rate=1e-3, seed=42):
    set_seed(seed)

    mlflow.set_tracking_uri("https://mlflow.schlaepfer.me")
    mlflow.set_experiment("DGS SignNet Features")

    with mlflow.start_run(log_system_metrics=False):
        train_dataset, val_dataset, test_dataset = get_datasets_with_fixed_test_split()

        # Check for zero features - should be rare in pre-extracted dataset
        zero_count = 0
        total = len(train_dataset)
        for features, label in train_dataset:
            if torch.all(features == 0):
                zero_count += 1
        print(f'{zero_count} out of {total} train samples have zero landmarks (no hand detected).')

        print(f"Starting batch size tuning")
        best_batch_size = 64 # tune_batch_size(train_dataset)

        train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=best_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False)

        device = get_device()
        print(f"Using device: {device}")

        model = SignNetCNN(num_classes=24).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

        mlflow.log_param("num_classes", 24)
        mlflow.log_param("batch_size", best_batch_size)
        mlflow.log_param("initial_lr", learning_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("seed", seed)
        mlflow.log_param("loss_function", criterion.__class__.__name__)
        mlflow.log_param("optimizer", optimizer.__class__.__name__)
        mlflow.log_param("scheduler", scheduler.__class__.__name__)

        with open("model_summary.txt", "w", encoding="utf-8") as f:
            f.write(str(summary(model)))
            mlflow.log_artifact("model_summary.txt")

        print(f"Starting training with batch size {best_batch_size}")
        train(model, optimizer, criterion, scheduler, train_loader, val_loader, device, epochs=epochs)

        print("Evaluating model on test dataset...")
        evaluate(model, test_loader, criterion, device)

        mlflow.pytorch.log_model(model, "model")
        torch.save(model.state_dict(), 'model/model.pth')

if __name__ == "__main__":
    main()
