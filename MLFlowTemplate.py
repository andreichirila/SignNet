import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchmetrics import Accuracy
import mlflow
from mlflow.models import infer_signature
import numpy as np
import random


# --------------------------
# Set random seed function
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------
# Data preparation
transform = transforms.ToTensor()

train_data = datasets.GTSRB(root="data", split="train", download=True, transform=transform)
test_data = datasets.GTSRB(root="data", split="test", download=True, transform=transform)

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)


# --------------------------
# Define a simple CNN model
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(10)  # 10 classes for FashionMNIST
        )

    def forward(self, x):
        return self.model(x)


# --------------------------
# Training and evaluation functions

def train_one_epoch(dataloader, model, loss_fn, metric_fn, optimizer, device):
    model.train()
    total_loss, total_acc, count = 0, 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        acc = metric_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()
        count += 1

        if batch % 100 == 0:
            print(f"Batch {batch} - Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}")

    avg_loss = total_loss / count
    avg_acc = total_acc / count
    return avg_loss, avg_acc


def evaluate(dataloader, model, loss_fn, metric_fn, device):
    model.eval()
    total_loss, total_acc, count = 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)
            acc = metric_fn(pred, y)

            total_loss += loss.item()
            total_acc += acc.item()
            count += 1

    avg_loss = total_loss / count
    avg_acc = total_acc / count
    return avg_loss, avg_acc


# --------------------------
# Main training loop with MLflow logging
def main(epochs=10, learning_rate=1e-3, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(seed)

    model = ImageClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    mlflow.set_tracking_uri("https://mlflow.schlaepfer.me")

    mlflow.set_experiment("Traffic sign")

    with mlflow.start_run():
        # Log params
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("seed", seed)
        mlflow.log_param("loss_function", loss_fn.__class__.__name__)
        mlflow.log_param("metric_function", metric_fn.__class__.__name__)
        mlflow.log_param("optimizer", optimizer.__class__.__name__)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            train_loss, train_acc = train_one_epoch(train_loader, model, loss_fn, metric_fn, optimizer, device)
            val_loss, val_acc = evaluate(test_loader, model, loss_fn, metric_fn, device)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        # Log the final model with signature for inference
        example_input = next(iter(train_loader))[0][:1]  # One batch sample
        example_input_np = example_input.cpu().numpy()
        signature = infer_signature(example_input_np, model(example_input.to(device)).cpu().detach().numpy())

        mlflow.pytorch.log_model(model, "model", signature=signature)

    print("Training complete and model logged to MLflow.")


if __name__ == "__main__":
    main()
