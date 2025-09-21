import torch.nn as nn

class SignLanguageMLP(nn.Module):  # Umbenannt für Klarheit
    def __init__(self, num_classes, input_size=63):  # Input: 63 Features
        super(SignLanguageMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),  # Erste FC Layer: 63 -> 256
            nn.BatchNorm1d(256),  # Norm für Stabilität
            nn.ReLU(),  # Activation
            nn.Dropout(0.3),  # Dropout gegen Overfitting

            nn.Linear(256, 128),  # Zweite: 256 -> 128
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),  # Dritte: 128 -> 64
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),  # Höheres Dropout am Ende

            nn.Linear(64, num_classes)  # Output: num_classes
        )

    def forward(self, x):  # x: [batch, 63]
        return self.fc(x)  # Direkt durch FC Layers