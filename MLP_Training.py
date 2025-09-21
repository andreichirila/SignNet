from LandmarkDataset import LandmarkDataset

if __name__ == "__main__":  # Check if the script is being executed directly
    import pandas as pd  # Import pandas for data processing
    from sklearn.model_selection import train_test_split  # Import function to split data into train/test sets
    from sklearn.preprocessing import LabelEncoder  # Import label encoder to convert labels to numeric values

    import torch  # Import PyTorch for deep learning and tensor manipulation
    import torch.optim as optim  # Import PyTorch optimizers for model training
    from torch.nn import CrossEntropyLoss  # Import cross-entropy loss function for classification tasks
    from torch.optim.lr_scheduler import ReduceLROnPlateau  # Scheduler to reduce learning rate on validation loss plateau
    from torch.utils.data import Dataset, DataLoader  # Utilities for creating datasets and handling mini-batches
    import numpy as np  # NumPy for efficient array operations

    import os  # OS for interacting with the filesystem
    import time  # Time module to measure execution duration
    from datetime import datetime  # Module for generating timestamps

    import mlflow  # MLflow for tracking model experiments and metrics
    import mlflow.pytorch  # MLflow plugin to handle PyTorch models
    from mlflow.models import ModelSignature
    from mlflow.types import Schema, TensorSpec

    from MLP_Model import SignLanguageMLP  # Import custom MLP model from the Landmarks module

    # Configure MLflow to use a specific tracking server
    mlflow.set_tracking_uri("https://mlflow.schlaepfer.me")  # Set URI for MLflow
    mlflow.set_experiment("Static Sign Net")  # Select/initialize experiment name in MLflow

    # Check device availability (GPU or CPU) for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    print(f"Using device: {device}")  # Display device information

    # Load the dataset containing hand-landmark coordinates
    df = pd.read_csv('landmark_datasets/german_sign_language.csv')  # Load the CSV dataset

    # Drop rows with missing values to ensure data consistency
    df = df.dropna()  # This ensures no incomplete rows are used during training

    # Logge Klassenverteilung
    print("Label distribution:\n", df['label'].value_counts().to_dict())

    # Encode categorical labels (e.g., 'a', 'b', etc.) into integers
    label_encoder = LabelEncoder()  # Initialize label encoder
    df['label_encoded'] = label_encoder.fit_transform(df['label'])  # Create numeric labels and add to dataframe
    num_classes = len(label_encoder.classes_)  # Count total number of unique classes
    print(f"Class names: {label_encoder.classes_}")  # Display the classes
    print(f"Number of classes: {num_classes}")  # Display the number of unique classes

    # Extract feature columns from the dataset
    feature_cols = [col for col in df.columns if 'coordinate' in col]  # Get all columns containing "coordinate"
    X = df[feature_cols].values.astype(np.float32)  # Extract features and convert to NumPy float32
    y = df['label_encoded'].values.astype(np.int64)  # Extract encoded class labels as int64

    # Normalize features relative to the wrist landmark (first three coordinates: x, y, z)
    for i in range(1, 21):  # Loop through all 21 landmarks excluding the wrist
        start_idx = i * 3  # Starting index of x, y, z coordinates for the current landmark
        X[:, start_idx:start_idx + 3] -= X[:, 0:3]  # Subtract wrist coordinates (x, y, z) for each data sample

    # Split the dataset into train, validation, and test sets (70-15-15 split)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)  # Train-temp split
    X_val, X_eval, y_val, y_eval = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)  # Temp split into validation and evaluation

    # Create PyTorch datasets for each data split
    batch_size = 32
    train_dataset = LandmarkDataset(X_train, y_train, augment=True)
    val_dataset = LandmarkDataset(X_val, y_val, augment=False)  # Keine Augmentation für Validation
    eval_dataset = LandmarkDataset(X_eval, y_eval, augment=False)

    # Create data loaders for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # Training data loader
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  # Validation data loader
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  # Test data loader
    epochs = 35  # Number of epochs to train the model
    learnin_rate = 0.0005

    # MLflow experiment tracking
    with mlflow.start_run():  # Start an MLflow run to log data and metrics
        model = SignLanguageMLP(num_classes).to(device)  # Create an instance of the model and move it to device
        criterion = CrossEntropyLoss()  # Define loss function
        optimizer = optim.RMSprop(model.parameters(), lr=learnin_rate)  # Use RMSprop optimizer with a learning rate of 0.0005

        # Log parameters to MLflow for transparency and reproducibility
        mlflow.log_param("batch_size", batch_size)  # Log batch size
        mlflow.log_param("epochs", epochs)  # Log total number of epochs
        mlflow.log_param("learning_rate", learnin_rate)  # Log learning rate
        mlflow.log_param("optimizer", "RMSprop")  # Log optimizer used
        mlflow.log_param("num_classes", num_classes)  # Log number of output classes

        # Learning rate scheduler to reduce LR on validation loss plateau
        reduce_lr = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6)  # Reduce learning rate when validation loss plateaus
        start_time = time.time()  # Track the start time of training

        # Training loop
        for epoch in range(epochs):  # Train for the specified number of epochs
            train_dataset.set_training(True)
            model.train()  # Set model to training mode
            running_loss = 0.0  # Initialize training loss for the current epoch
            correct = 0  # Count correct classifications in the epoch
            total = 0  # Count total samples in the epoch

            for inputs, labels in train_loader:  # Loop through training data batches
                inputs, labels = inputs.to(device), labels.to(device)  # Move images and labels to the same device
                optimizer.zero_grad()  # Clear gradients from the previous step
                outputs = model(inputs)  # Perform forward pass
                loss = criterion(outputs, labels)  # Compute the loss
                loss.backward()  # Backward pass to compute gradients
                optimizer.step()  # Update model weights

                running_loss += loss.item()  # Accumulate the epoch loss
                _, predicted = torch.max(outputs, 1)  # Get predicted classes
                total += labels.size(0)  # Count total labels in batch
                correct += (predicted == labels).sum().item()  # Count correctly predicted labels

            # Calculate average training loss and accuracy
            train_loss = running_loss / len(train_loader)  # Average training loss
            train_acc = correct / total  # Training accuracy

            # Validation phase
            val_dataset.set_training(False)
            model.eval()  # Set model mode to evaluation (no gradient updates)
            val_loss = 0.0  # Initialize validation loss for the epoch
            correct = 0  # Initialize correct predictions counter
            total = 0  # Initialize total samples counter

            with torch.no_grad():  # Disable gradients for validation (faster computation)
                for inputs, labels in val_loader:  # Loop over validation data batches
                    inputs, labels = inputs.to(device), labels.to(device)  # Move images and labels to the selected device
                    outputs = model(inputs)  # Forward pass
                    loss = criterion(outputs, labels)  # Compute validation loss
                    val_loss += loss.item()  # Accumulate total validation loss
                    _, predicted = torch.max(outputs, 1)  # Get predicted class
                    total += labels.size(0)  # Update total number of samples
                    correct += (predicted == labels).sum().item()  # Count correct predictions

            val_loss = val_loss / len(val_loader)  # Compute average validation loss
            val_acc = correct / total  # Compute validation accuracy

            # Log training and validation results for the current epoch
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_accuracy": train_acc,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, step=epoch)

            reduce_lr.step(val_loss)  # Update learning rate based on validation loss

    end_time = time.time()  # Record the end time of training
    duration = end_time - start_time  # Compute total training time
    print(f"Training completed in {duration:.2f} seconds")

    # Evaluation phase
    model.eval()  # Set model mode to evaluation
    eval_loss = 0.0  # Initialize evaluation loss
    correct = 0  # Initialize correct predictions counter
    total = 0  # Initialize total samples counter
    y_true = []  # List to store ground truth labels
    y_pred = []  # List to store model predictions

    with torch.no_grad():  # Disable gradients for evaluation
        for inputs, labels in eval_loader:  # Loop over evaluation data batches
            inputs, labels = inputs.to(device), labels.to(device)  # Move images and labels to the selected device
            outputs = model(inputs)  # Forward pass
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

    # Definiere die Signature
    signature = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype(np.float32), (-1, 63))]),  # Input: [batch_size, 63]
        outputs=Schema([TensorSpec(np.dtype(np.float32), (-1, num_classes))])  # Output: [batch_size, num_classes]
    )

    mlflow.pytorch.log_model(
        pytorch_model=model,
        name="MLP_Augmentation",
        signature=signature,
    )
    # Ensure current_time is properly formatted as a string using datetime
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # save the trained model
    os.makedirs("models", exist_ok=True)  # Ensure model folder exists
    #Save model weights and metadata for reproducibility

    torch.save(model.state_dict(), f"./models/MLP_trainedModel-{current_time}-eval_loss-{round(eval_loss, 3)}"
                                   f"-eval_acc-{round(eval_acc, 3)}-train_time-{round(duration, 3)}.pth")