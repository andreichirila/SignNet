import torch.nn.functional as F
import torch.nn as nn

class SignNetCNN(nn.Module):
    def __init__(self, num_classes, input_size=63, dropout_prob=0.3):
        super(SignNetCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_prob),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_prob),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_prob),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(max(0.4, dropout_prob)),
        )
        self.fc_out = nn.Linear(64, num_classes)

        # Optional shortcut layers for residual connections
        self.shortcut1 = nn.Linear(input_size, 512, bias=False)
        self.shortcut2 = nn.Linear(512, 256, bias=False)
        self.shortcut3 = nn.Linear(256, 128, bias=False)

    def forward(self, x):
        # Layer 1 + residual
        out1 = self.layer1(x) + self.shortcut1(x)
        # Layer 2 + residual
        out2 = self.layer2(out1) + self.shortcut2(out1)
        # Layer 3 + residual
        out3 = self.layer3(out2) + self.shortcut3(out2)
        # Layer 4
        out4 = self.layer4(out3)
        return self.fc_out(out4)