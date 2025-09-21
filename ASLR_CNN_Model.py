import torch.nn as nn

# Define the model (custom CNN architecture for sign language classification)
class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes):  # Initialize model with specified number of output classes
        super(SignLanguageCNN, self).__init__()  # Inherit from nn.Module
        self.features = nn.Sequential(  # Define feature extraction layers
            nn.Conv2d(3, 32, kernel_size=5),  # Convolution layer (input: 3 channels, output: 32 channels)
            nn.BatchNorm2d(32),  # Batch normalization for stability
            nn.ReLU(),  # ReLU activation
            nn.Conv2d(32, 32, kernel_size=5),  # Second convolution layer
            nn.BatchNorm2d(32),  # Batch normalization
            nn.ReLU(),  # ReLU activation
            nn.Conv2d(32, 64, kernel_size=3),  # Third convolution layer (increasing depth to 64 channels)
            nn.BatchNorm2d(64),  # Batch normalization
            nn.ReLU(),  # ReLU activation
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling layer to downsample by 2x
            nn.Conv2d(64, 128, kernel_size=3),  # Fourth convolution layer (128 channels)
            nn.BatchNorm2d(128),  # Batch normalization
            nn.ReLU(),  # ReLU activation
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling layer
            nn.Conv2d(128, 256, kernel_size=3),  # Further deepening to 256 channels
            nn.BatchNorm2d(256),  # Batch normalization
            nn.ReLU(),  # ReLU activation
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling
            nn.Conv2d(256, 256, kernel_size=3),  # Final convolution layer
            nn.BatchNorm2d(256),  # Batch normalization
            nn.ReLU(),  # ReLU activation
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling to reduce spatial dimensions to 1x1
        )
        self.classifier = nn.Sequential(  # Define fully connected layers for classification
            nn.Linear(256, 128),  # Fully connected layer (input: 256, output: 128)
            nn.ReLU(),  # ReLU activation
            nn.Linear(128, 64),  # Fully connected layer (output: 64)
            nn.ReLU(),  # ReLU activation
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(64, num_classes),  # Final fully connected layer (output: number of classes)
        )

    def forward(self, x):  # Define forward pass of the model
        x = self.features(x)  # Pass input through feature extractor
        x = x.view(x.size(0), -1)  # Flatten spatial dimensions for fully connected layers
        x = self.classifier(x)  # Pass through classifier layers
        return x  # Return output