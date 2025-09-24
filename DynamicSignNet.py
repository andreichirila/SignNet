import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json

class SignLanguageLSTM(nn.Module):
    def __init__(self, input_dim=1530, hidden_dim=128, num_classes=100):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])
        return output

def compute_accuracy(output, target):
    preds = output.argmax(dim=1)
    correct = (preds == target).sum().item()
    return correct, target.size(0)

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for landmarks, labels in dataloader:
        landmarks, labels = landmarks.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(landmarks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        c, t = compute_accuracy(outputs, labels)
        correct += c
        total += t
    print(f"Train Loss: {running_loss / len(dataloader):.4f}, Accuracy: {correct / total:.4f}")

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for landmarks, labels in dataloader:
            landmarks, labels = landmarks.to(device), labels.to(device)
            outputs = model(landmarks)
            c, t = compute_accuracy(outputs, labels)
            correct += c
            total += t
    print(f"Eval Accuracy: {correct / total:.4f}")

def load_vocab(vocab_file):
    import json
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Loaded vocabulary of size {len(vocab)} from {vocab_file}")
    return vocab


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your vocab here (same vocab used for preprocessing)
    vocab = load_vocab("vocab.json")

    train_ds = LandmarkDataset("./landmarks_train", vocab)
    dev_ds = LandmarkDataset("./landmarks_dev", vocab)
    test_ds = LandmarkDataset("./landmarks_test", vocab)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=4)
    test_loader = DataLoader(test_ds, batch_size=4)

    model = SignLanguageLSTM(input_dim=1530, hidden_dim=128, num_classes=len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        train(model, train_loader, criterion, optimizer, device)
        evaluate(model, dev_loader, device)

    # Final test run
    evaluate(model, test_loader, device)
