if __name__ == "__main__":  # Entry point of the script to ensure proper process handling (not run on import)
    import numpy as np  # Import numpy for numerical operations
    import torch  # Import PyTorch for tensor manipulation and deep learning
    import torch.nn as nn  # Import PyTorch's neural network module to define models
    import torch.optim as optim  # Import PyTorch's optimizer module for training
    import torchvision  # Import torchvision for datasets and transformations
    import torchvision.transforms as transforms  # To apply transformations to images
    from torch.utils.data import DataLoader, SubsetRandomSampler  # For creating data loaders and sampling datasets
    import json  # For working with JSON data formats (e.g., saving results)
    import time  # To measure the duration of the training
    from datetime import datetime  # To timestamp generated files
    import os  # For interacting with the filesystem (e.g., make directories)

    # Check for GPU availability and configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU
    print(f"Using device: {device}")  # Output the selected device
    print(f"Check CUDA version PyTorch was built with: {torch.version.cuda}")  # Output the selected device
    print(f"GPU available: {torch.cuda.is_available()}")  # Output the selected device

    # Set some params
    batch_size = 64  # Number of images in each training batch
    img_height = 64  # Height of input images
    img_width = 64  # Width of input images
    num_classes = None  # Place-holder for number of classes, to be set dynamically after loading dataset

    # Define data transforms (augmentation + preprocessing)
    train_transforms = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=0),  # Apply random affine transformations
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values to [-1, 1] range
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values to [-1, 1] range
    ])

    # Load dataset using ImageFolder
    dataset = torchvision.datasets.ImageFolder(
        root="./dataset_compressed",  # Path to the dataset folder
        transform=train_transforms  # Apply training transformations to dataset
    )

    # Extract class names from dataset
    class_names = dataset.classes  # List of sub-folder names in dataset (used as class labels)
    num_classes = len(class_names)  # Total number of classes
    print(f"Class names: {class_names}")  # Output the class names for verification

    # Split dataset into train, validation, and evaluation sets
    dataset_size = len(dataset)  # Total number of images in dataset
    indices = list(range(dataset_size))  # Create a list of indices for the dataset
    np.random.seed(118)  # Set random seed for reproducibility
    np.random.shuffle(indices)  # Shuffle indices for random dataset split

    # Define split sizes for validation and evaluation
    val_split = 0.3  # Proportion of data to use for validation and evaluation
    train_size = int((1 - val_split) * dataset_size)  # Remaining portion for training
    val_size = dataset_size - train_size  # Validation + evaluation size
    eval_size = (2 * val_size) // 3  # Two-thirds of validation split for evaluation
    val_size = val_size - eval_size  # Remaining split for validation

    # Slice indices into training, validation, and evaluation sets
    train_indices = indices[:train_size]  # First portion for training
    val_indices = indices[train_size:train_size + val_size]  # Next portion for validation
    eval_indices = indices[train_size + val_size:]  # Remaining portion for evaluation

    # Create samplers for selecting data subsets
    train_sampler = SubsetRandomSampler(train_indices)  # Random sampler for training data
    val_sampler = SubsetRandomSampler(val_indices)  # Random sampler for validation data
    eval_sampler = SubsetRandomSampler(eval_indices)  # Random sampler for evaluation data

    # Create DataLoaders for loading data in batches
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2)  # Training data
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=2)  # Validation data
    eval_loader = DataLoader(dataset, batch_size=batch_size, sampler=eval_sampler, num_workers=2)  # Evaluation data

    # Update dataset to use validation transforms (no augmentation)
    dataset.transform = val_transforms  # Apply validation transformations for validation and evaluation

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
                nn.Softmax(dim=1)  # Apply Softmax activation for class probabilities
            )

        def forward(self, x):  # Define forward pass of the model
            x = self.features(x)  # Pass input through feature extractor
            x = x.view(x.size(0), -1)  # Flatten spatial dimensions for fully connected layers
            x = self.classifier(x)  # Pass through classifier layers
            return x  # Return output

    # Initialize model, loss function, and optimizer
    model = SignLanguageCNN(num_classes).to(device)  # Instantiate and move model to selected device
    criterion = nn.CrossEntropyLoss()  # Define cross-entropy loss for classification
    optimizer = optim.RMSprop(model.parameters(), lr=0.0005)  # Use RMSprop optimizer with specified learning rate


    # Custom ReduceLROnPlateau class to reduce learning rate when validation loss plateaus
    class ReduceLROnPlateau:
        def __init__(self, optimizer, factor=0.5,
                     patience=6):  # Initialize with optimizer, reduction factor, and patience
            self.optimizer = optimizer  # Optimizer to apply learning rate changes to
            self.factor = factor  # Factor to reduce the learning rate (e.g., new_lr = old_lr * factor)
            self.patience = patience  # Number of epochs to wait before reducing learning rate
            self.best_loss = float('inf')  # Initialize best loss to infinity
            self.counter = 0  # Counter to track number of plateaued epochs

        def step(self, val_loss):  # Called at the end of each epoch with the current validation loss
            if val_loss < self.best_loss:  # If validation loss improves
                self.best_loss = val_loss  # Update best loss
                self.counter = 0  # Reset counter
            else:  # If validation loss does not improve
                self.counter += 1  # Increment counter
                if self.counter >= self.patience:  # If patience threshold is exceeded
                    for param_group in self.optimizer.param_groups:  # Loop through all parameter groups
                        param_group['lr'] *= self.factor  # Reduce the learning rate
                    self.counter = 0  # Reset counter
                    print(f"Reduced learning rate to {self.optimizer.param_groups[0]['lr']}")  # Log the change


    # Custom EarlyStopping class to stop training if validation loss stops improving
    class EarlyStopping:
        def __init__(self, patience=30):  # Initialize with patience value
            self.patience = patience  # Number of epochs to wait before stopping
            self.best_loss = float('inf')  # Initialize best loss to infinity
            self.counter = 0  # Counter to track number of plateaued epochs
            self.early_stop = False  # Flag to indicate if training should stop

        def step(self, val_loss):  # Called at the end of each epoch with the current validation loss
            if val_loss < self.best_loss:  # If validation loss improves
                self.best_loss = val_loss  # Update best loss
                self.counter = 0  # Reset counter
            else:  # If validation loss does not improve
                self.counter += 1  # Increment counter
                if self.counter >= self.patience:  # If patience threshold is exceeded
                    self.early_stop = True  # Set early stop flag
                    print("Early stopping triggered")  # Log the event


    # Training loop configuration
    epochs = 80  # Number of epochs to train the model
    reduce_lr = ReduceLROnPlateau(optimizer, factor=0.5, patience=6)  # Initialize ReduceLROnPlateau
    early_stop = EarlyStopping(patience=10)  # Initialize EarlyStopping

    history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}  # Dictionary to store training and validation metrics
    start_time = time.time()  # Record the start time of training

    # Training loop
    for epoch in range(epochs):  # Loop over the number of epochs
        model.train()  # Set model mode to training
        running_loss = 0.0  # Initialize running loss for the epoch
        correct = 0  # Initialize correct predictions counter
        total = 0  # Initialize total samples counter

        for images, labels in train_loader:  # Loop over training data batches
            images, labels = images.to(device), labels.to(device)  # Move images and labels to the selected device
            optimizer.zero_grad()  # Reset gradients from the previous step
            outputs = model(images)  # Forward pass through the model
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass to calculate gradients
            optimizer.step()  # Update model parameters

            running_loss += loss.item()  # Accumulate total loss for the epoch
            _, predicted = torch.max(outputs, 1)  # Get predicted class from model output
            total += labels.size(0)  # Update total number of samples
            correct += (predicted == labels).sum().item()  # Count correct predictions

        train_loss = running_loss / len(train_loader)  # Compute average training loss
        train_acc = correct / total  # Compute training accuracy
        history['acc'].append(train_acc)  # Save training accuracy to history
        history['loss'].append(train_loss)  # Save training loss to history

        # Validation phase
        model.eval()  # Set model mode to evaluation (no gradient updates)
        val_loss = 0.0  # Initialize validation loss for the epoch
        correct = 0  # Initialize correct predictions counter
        total = 0  # Initialize total samples counter

        with torch.no_grad():  # Disable gradients for validation (faster computation)
            for images, labels in val_loader:  # Loop over validation data batches
                images, labels = images.to(device), labels.to(device)  # Move images and labels to the selected device
                outputs = model(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute validation loss
                val_loss += loss.item()  # Accumulate total validation loss
                _, predicted = torch.max(outputs, 1)  # Get predicted class
                total += labels.size(0)  # Update total number of samples
                correct += (predicted == labels).sum().item()  # Count correct predictions

        val_loss = val_loss / len(val_loader)  # Compute average validation loss
        val_acc = correct / total  # Compute validation accuracy
        history['val_acc'].append(val_acc)  # Save validation accuracy to history
        history['val_loss'].append(val_loss)  # Save validation loss to history

        # Log training and validation results for the current epoch
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        reduce_lr.step(val_loss)  # Update learning rate based on validation loss
        early_stop.step(val_loss)  # Check if early stopping should be triggered
        if early_stop.early_stop:  # If training should stop early
            break  # Exit the training loop

    end_time = time.time()  # Record the end time of training
    duration = end_time - start_time  # Compute total training time

    # Evaluation phase
    model.eval()  # Set model mode to evaluation
    eval_loss = 0.0  # Initialize evaluation loss
    correct = 0  # Initialize correct predictions counter
    total = 0  # Initialize total samples counter
    y_true = []  # List to store ground truth labels
    y_pred = []  # List to store model predictions

    with torch.no_grad():  # Disable gradients for evaluation
        for images, labels in eval_loader:  # Loop over evaluation data batches
            images, labels = images.to(device), labels.to(device)  # Move images and labels to the selected device
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute evaluation loss
            eval_loss += loss.item()  # Accumulate total evaluation loss
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            total += labels.size(0)  # Update total number of samples
            correct += (predicted == labels).sum().item()  # Count correct predictions
            y_true.extend(labels.cpu().numpy())  # Append ground truth labels to list
            y_pred.extend(predicted.cpu().numpy())  # Append predictions to list

    eval_loss = eval_loss / len(eval_loader)  # Compute average evaluation loss
    eval_acc = correct / total  # Compute evaluation accuracy
    print(f"Test Loss: {eval_loss:.4f}, Test Acc: {eval_acc:.4f}")  # Log evaluation results

    # Save evaluation results to a JSON file
    evaluation_result_data = {  # Create a dictionary with ground truth and predicted labels
        "y": y_true,  # Ground truth labels
        "pred": y_pred  # Predicted labels
    }
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')  # Format current time for file naming
    evaluation_result_json = json.dumps(evaluation_result_data, indent=4)  # Convert results to JSON format

    os.makedirs("../results", exist_ok=True)  # Ensure output folder exists
    with open(f"../results/evaluationResults-{current_time}.json", "w") as outfile:  # Open output file
        outfile.write(evaluation_result_json)  # Write evaluation results to file

    # Save training history to a JSON file
    history_json = json.dumps(history, indent=4)  # Convert training history to JSON format
    with open(f"../results/results-{current_time}.json", "w") as outfile:  # Open output file
        outfile.write(history_json)  # Write training history to file

    # Optionally save the trained model
    # os.makedirs("../models", exist_ok=True)  # Ensure model folder exists
    # Save model weights and metadata for reproducibility
    # torch.save(model.state_dict(), f"../models/trainedModel-{current_time}-eval_loss-{round(eval_loss, 3)}"
    #           f"-eval_acc-{round(eval_acc, 3)}-train_time-{round(duration, 3)}.pth")
