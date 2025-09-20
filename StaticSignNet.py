import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import mlflow
import mlflow.pytorch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torchinfo import summary

DATASET_PATH = 'dataset'
TEST_INDICES_PATH = 'test_indices.json'

# Your CNN model class CustomCNN should be defined here or imported
class SignNet(nn.Module):
    def __init__(self, num_classes):
        super(SignNet, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Block 4
        self.pool1 = nn.MaxPool2d(2)

        # Block 5
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2)

        # Block 6
        self.conv5 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(384)
        self.pool3 = nn.MaxPool2d(2)

        # Block 7
        self.conv6 = nn.Conv2d(384, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        # GAP Layer
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # FC Layers
        self.fc1 = nn.Linear(512, 84)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(84, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # Block 4
        x = self.pool1(x)

        # Block 5
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Block 6
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Block 7
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)

        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        # Fully Connected
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        x = self.softmax(x)
        return x

def get_datasets_with_fixed_test_split(transform_train, transform_val):
    dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform_train)
    total_size = len(dataset)
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
        temp_dataset = Subset(dataset, train_val_indices)
        train_val_size = len(temp_dataset)
        train_size = int(0.8 * train_val_size)
        val_size = train_val_size - train_size
        splits = random_split(train_val_indices, [train_size, val_size])
        train_indices, val_indices = splits[0].indices, splits[1].indices
        print(f"Loaded test indices from {TEST_INDICES_PATH}")

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Override transforms
    train_dataset.dataset.transform = transform_train
    val_dataset.dataset.transform = transform_val
    test_dataset.dataset.transform = transform_val

    return train_dataset, val_dataset, test_dataset

def tune_batch_size(dataset, batch_sizes=[16, 32, 64]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignNet(num_classes=25).to(device)
    criterion = nn.CrossEntropyLoss()
    best_batch = batch_sizes[0]
    best_loss = float('inf')
    print("Starting batch size tuning...")
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        total_loss = 0.0
        count = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
            if count > 5:
                print(f"  Stopping early after {count} batches for quick estimate")
                break
        avg_loss = total_loss / count
        print(f"Batch size {batch_size}: training loss {avg_loss:.4f}")
        mlflow.log_metric(f"batch_size_{batch_size}_loss", avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_batch = batch_size
            print(f"New best batch size found: {best_batch} with loss {best_loss:.4f}")
        else:
            print(f"No improvement with batch size {batch_size}")
    print(f"Selected batch size: {best_batch}")
    mlflow.log_param("selected_batch_size", best_batch)
    return best_batch


def train(model, optimizer, criterion, scheduler, train_loader, val_loader, device, epochs=20):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        total_batches = len(train_loader)
        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_idx % 10 == 0 or batch_idx == total_batches:
                print(f"Epoch {epoch+1} [{batch_idx}/{total_batches}] - Batch Loss: {loss.item():.4f} - {(batch_idx/total_batches)*100:.1f}% complete")
        avg_train_loss = train_loss / total_batches

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(epochs=3, learning_rate=1e-3, seed=42):
    set_seed(seed)

    mlflow.set_tracking_uri("https://mlflow.schlaepfer.me")
    mlflow.set_experiment("Static Sign Net")

    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
    ])
    val_transform = transforms.ToTensor()

    with mlflow.start_run(log_system_metrics=True):
        train_dataset, val_dataset, test_dataset = get_datasets_with_fixed_test_split(train_transform, val_transform)

        print(f"Starting with batch size tuning")
        best_batch_size = 16 # tune_batch_size(train_dataset)

        train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=best_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SignNet(num_classes=25).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

        mlflow.log_param("num_classes", 25)
        mlflow.log_param("initial_lr", learning_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("augmentation", "RandomAffine degrees=10, translate=0.1, scale=0.9-1.1")
        mlflow.log_param("seed", seed)
        mlflow.log_param("loss_function", criterion.__class__.__name__)
        mlflow.log_param("optimizer", optimizer.__class__.__name__)
        mlflow.log_param("scheduler", scheduler.__class__.__name__)

        # Log model summary.
        with open("model_summary.txt", "w", encoding="utf-8") as f:
            f.write(str(summary(model)))
            mlflow.log_artifact("model_summary.txt")

        print(f"Starting training with batch size {best_batch_size}")
        train(model, optimizer, criterion, scheduler, train_loader, val_loader, device, epochs=epochs)

        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    main()
