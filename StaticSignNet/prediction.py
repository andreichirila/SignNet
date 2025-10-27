import cv2
import torch

import mediapipe as mp
import numpy as np
import torch.serialization
import torch.nn.functional as F

import mlflow

# Load the pretrained model weights
# model = ImprovedSignLanguageMLP(num_classes=24)  # initialize model class
# model.load_state_dict(torch.load('model/model.pth', map_location='cpu'))

mlflow.set_tracking_uri("https://mlflow.schlaepfer.me")
model_uri = "models:/StaticHandSignCNN/4"
model = mlflow.pytorch.load_model(model_uri)
model.eval()

# model = torch.load('model/model.pth', map_location=torch.device('cpu'), weights_only=False)
# model.eval()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Gesture labels A-Z
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm',
                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

def hand_scale(hand_lms):
    # hand_lms: (21, 3)
    # landmarks: shape (63,) -> reshape (21,3)
    hand_lms = np.array(hand_lms).reshape(21, 3)
    wrist = hand_lms[0]
    tips = hand_lms[[4, 8, 12, 16, 20]]  # Thumb, index, middle, ring, pinky tips
    tip_dists = [np.linalg.norm(tip[:2] - wrist[:2]) for tip in tips]
    avg_dist = np.mean(tip_dists)
    # Also consider box scale:
    min_xy = hand_lms[:, :2].min(axis=0)
    max_xy = hand_lms[:, :2].max(axis=0)
    box_diag = np.linalg.norm(max_xy - min_xy)
    scale = max(avg_dist, box_diag)
    if scale == 0:
        scale = 1
    normed = (hand_lms - wrist) / scale
    return normed.flatten()

def predict_gesture(landmarks, model):
    # landmarks: list of 63 floats (21 points * 3)
    landmarks = hand_scale(landmarks)
    with torch.no_grad():
        input_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)  # batch size 1
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
        pred_idx = torch.argmax(probabilities, dim=1).item()
        pred_prob = probabilities[0, pred_idx].item()
        return pred_idx, pred_prob

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    # frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract normalized landmarks x,y,z
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            # Predict gesture class and probability from landmarks
            predicted_index, predicted_prob = predict_gesture(landmark_list, model)
            predicted_label = labels[predicted_index]

            # Display prediction and probability on the frame (show probability as percentage)
            cv2.putText(frame, f'Predicted: {predicted_label} ({predicted_prob*100:.1f}%)', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('DGS Alphabet Recognition', frame)

    # Break on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
