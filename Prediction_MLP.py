import cv2  # Import OpenCV for image processing and webcam access
import torch  # Import PyTorch for model loading and inference

from MLP_Model import SignLanguageMLP  # Import the custom Multi-Layer Perceptron (MLP) model for predictions
from HandTrackingOpenCV import HandDetector  # Import a pre-built hand detector based on MediaPipe
import numpy as np  # Import NumPy for array operations and numerical computations
import os  # Import OS module for interacting with the file system
import glob  # Import glob module for finding files that match a pattern

# Configuration settings for predictions
CONFIG = {
    'num_classes': 24,  # Number of output classes (24 letters mapped to model outputs)
    'class_names': [  # List of class names corresponding to the output classes
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
        'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
        'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
    ],
    'confidence_threshold': 0.7,  # Minimum confidence threshold for considering a prediction valid
    'model_dir': './models',  # Directory path to saved models
}

def load_latest_model(model_dir, num_classes, device):
    """
    Loads the most recently saved MLP model for predicting sign language.
    """
    model_files = glob.glob(os.path.join(model_dir, "*.pth"))  # Find all model files in the directory
    if not model_files:  # Check if no model files are found
        raise ValueError(f"No trained model found in {model_dir}. Train a model first.")  # Raise error if no files

    latest_model = max(model_files, key=os.path.getctime)  # Get the model file with the latest creation/modification time
    print(f"Loading latest model: {latest_model}")  # Inform user about the model being loaded

    model = SignLanguageMLP(num_classes).to(device)  # Initialize the model and move it to the desired computation device (CPU/GPU)
    model.load_state_dict(torch.load(latest_model, map_location=device))  # Load the weights into the model
    model.eval()  # Set the model to evaluation mode (disables gradient computations and certain layers like dropout)
    return model  # Return the prepared model object

def predict_letter(model, input_tensor, class_names, threshold):
    """
    Predicts a sign language letter and its confidence given the input tensor.
    """
    with torch.no_grad():  # Disable gradient calculations for inference to save memory and computation
        outputs = model(input_tensor)  # Perform a forward pass to get model predictions
        probs = torch.softmax(outputs, dim=1)  # Convert raw model outputs (logits) to probabilities
        conf, predicted = torch.max(probs, 1)  # Get the highest probability and the associated class index
        confidence = conf.item()  # Extract the confidence value as a plain Python float

        if confidence > threshold:  # If confidence exceeds the threshold, classify it
            letter = class_names[predicted.item()]  # Map the predicted index to the corresponding class name
        else:  # If confidence is below the threshold, classify as "Unknown"
            letter = "Unknown"
            confidence = 0.0

    return letter, confidence  # Return the predicted letter and its confidence score

def main():
    # Determine which device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # Notify which device is being used

    # Load the most recent trained MLP model for predictions
    model = load_latest_model(CONFIG['model_dir'], CONFIG['num_classes'], device)

    # Initialize the hand detector with specified settings
    detector = HandDetector(detectionCon=0.8, maxHands=1)  # High detection confidence and detecting only 1 hand

    # Configure the webcam feed for collecting video frames
    capture = cv2.VideoCapture(0)  # Open default webcam (camera ID 0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set video stream width to 1920 pixels
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set video stream height to 1080 pixels

    try:
        while True:  # Continuously process video frames
            ret, img = capture.read()  # Capture a single frame from the webcam
            if not ret:  # If capturing fails, exit the loop
                print("Camera error – exiting.")  # Notify user
                break  # Exit the loop

            # Initialize annotated_text with a default value
            annotated_text = "No Hands Detected"  # Default message if no hands are detected
            color = (0, 0, 255)  # Default color (red)

            # Detect hands in the current frame
            hands, img_annotated, bbox = detector.findHands(img, draw=True)  # Annotate hands if detected
            if hands:  # If hand landmarks are detected
                # Extract the landmark points for the first detected hand
                hand_landmarks = hands[0]['lmList']  # Get list of (x, y, z) tuples for the hand landmarks

                # Ensure the detected landmarks are valid
                if hand_landmarks and len(hand_landmarks) > 0:  # Check if any landmarks exist
                    inner_len = len(hand_landmarks[0])  # Determine if the points provide 2D or 3D data

                # Flatten the 21 (x, y, z) landmarks into a single feature vector (1D array of 63 elements)
                input_features = np.array(hand_landmarks).flatten().astype(np.float32)

                # Normalize the landmarks relative to the wrist position (landmark 0)
                wrist = input_features[0:3]  # Extract the x, y, z coordinates of the wrist
                for i in range(1, 21):  # Loop through all other landmarks
                    start_idx = i * 3  # Index to start extracting x, y, z coordinates of the current landmark
                    input_features[start_idx:start_idx + 3] -= wrist  # Subtract wrist coordinates to normalize

                # Check that the feature vector contains exactly 63 elements
                if len(input_features) != 63:  # If the size is incorrect, raise an error
                    raise ValueError(f"Input features must have 63 elements, but got {len(input_features)}")

                # Convert the feature array into a PyTorch tensor and add a batch dimension
                input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).to(device)

                # Feed the prepared tensor into the model to get predictions
                letter, confidence = predict_letter(model, input_tensor, CONFIG['class_names'], CONFIG['confidence_threshold'])

                # Annotate the frame with the model's prediction and confidence
                if confidence > 0.7:  # Only display predictions if confidence exceeds 0.7
                    color = (0, 255, 0)  # Green for valid predictions above the threshold
                    annotated_text = f"{letter}: {confidence:.2f}"  # Display exact confidence up to 2 decimal places
                else:
                    color = (0, 0, 255)  # Red for invalid or low-confidence predictions
                    annotated_text = "Unknown"  # Show "Unknown" for low-confidence predictions

            # Ensure bbox is not None before using it
            if bbox is not None:
                cv2.putText(img_annotated, annotated_text, (bbox[0], bbox[1] - 10),
                            # Place the text near the detected hand
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Display the annotated frame in a window
            cv2.imshow("Live Sign Language Prediction", img_annotated)

            # Check if the ESC key has been pressed to exit the loop
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key (ASCII 27)
                break

    except KeyboardInterrupt:  # Handle manual interruption (Ctrl+C)
        print("\nInterrupted by user.")
    finally:
        # Release the webcam resource and close all OpenCV windows
        capture.release()  # Release the webcam
        cv2.destroyAllWindows()  # Close all OpenCV windows
        print("Resources released. Exiting.")  # Notify user about resource cleanup

# Entry point for the program
if __name__ == "__main__":
    main()  # Run the main function