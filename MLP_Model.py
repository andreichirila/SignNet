import torch.nn as nn

class SignLanguageMLP(nn.Module):
    def __init__(self, num_classes, input_size=63, dropout_prob=0.3):
        """
        MLP for Static Sign Language Recognition.
        :param num_classes: Number of output classes for classification.
        :param input_size: Number of input features (default: 63).
        :param dropout_prob: Dropout probability to prevent overfitting.
        """
        super(SignLanguageMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_prob),

            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_prob),

            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_prob),

            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(max(0.4, dropout_prob)),  # Higher dropout for smaller layers

            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Forward pass.
        :param x: Input tensor of shape [batch_size, input_size].
        :return: Output logits of shape [batch_size, num_classes].
        """
        if x.shape[1] != self.fc[0].in_features:
            raise ValueError(f"Unexpected input size {x.shape[1]}, expected {self.fc[0].in_features}")
        return self.fc(x)