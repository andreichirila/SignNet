import cv2
import torch
from MLP_Model import SignLanguageMLP
from HandTrackingOpenCV import HandDetector
import numpy as np
import os
import glob
import csv
import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk

CONFIG = {
    'num_classes': 24,
    'class_names': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm',
                    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y'],
    'confidence_threshold': 0.7,
    'model_dir': './models',
    'output_csv': './landmark_datasets/new_samples_german_sign_language.csv',
    'frame_width': 640,
    'frame_height': 480
}

def load_latest_model(model_dir, num_classes, device):
    model_files = glob.glob(os.path.join(model_dir, "*.pth"))
    if not model_files:
        raise ValueError(f"No trained model found in {model_dir}.")
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Loading latest model: {latest_model}")
    model = SignLanguageMLP(num_classes).to(device)
    model.load_state_dict(torch.load(latest_model, map_location=device))
    model.eval()
    return model

def predict_letter(model, input_tensor, class_names, threshold):
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)
        confidence = conf.item()
        letter = class_names[predicted.item()] if confidence > threshold else "unknown"
        confidence = confidence if confidence > threshold else 0.0
    return letter, confidence

class SignLanguageGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Data Collection")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = load_latest_model(CONFIG['model_dir'], CONFIG['num_classes'], self.device)
        self.detector = HandDetector(detectionCon=0.8, maxHands=1)
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['frame_width'])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['frame_height'])

        if not os.path.exists(CONFIG['output_csv']):
            os.makedirs(os.path.dirname(CONFIG['output_csv']), exist_ok=True)
            with open(CONFIG['output_csv'], 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['label'] + [f'coordinate {i}' for i in range(63)]
                writer.writerow(header)

        self.canvas = tk.Canvas(root, width=CONFIG['frame_width'], height=CONFIG['frame_height'])
        self.canvas.pack()

        self.label_text = tk.StringVar(value="No Hands Detected")
        self.label_display = tk.Label(root, textvariable=self.label_text, font=("Arial", 14))
        self.label_display.pack()

        self.letter_var = tk.StringVar(value=CONFIG['class_names'][0])
        self.letter_menu = tk.OptionMenu(root, self.letter_var, *CONFIG['class_names'])
        self.letter_menu.pack()

        self.record_button = tk.Button(root, text="Record Sample", command=self.record_sample)
        self.record_button.pack()

        self.root.bind('r', lambda event: self.record_sample())

        self.input_features = None
        self.predicted_letter = None
        self.confidence = 0.0

        self.update_webcam()

    def update_webcam(self):
        ret, img = self.capture.read()
        if not ret:
            print("Camera error – exiting.")
            self.root.quit()
            return

        hands, img_annotated, bbox = self.detector.findHands(img, draw=True)
        self.label_text.set("No Hands Detected")
        self.input_features = None
        self.predicted_letter = None
        self.confidence = 0.0

        if hands:
            hand_landmarks = hands[0]['lmList']
            if hand_landmarks and len(hand_landmarks) > 0:
                normalized_landmarks = []
                for lm in hand_landmarks:
                    x, y, z = lm
                    x_norm = x / CONFIG['frame_width']
                    y_norm = y / CONFIG['frame_height']
                    normalized_landmarks.extend([x_norm, y_norm, z])

                self.input_features = np.array(normalized_landmarks).flatten().astype(np.float32)

                wrist = self.input_features[0:3].copy()  # Copy to avoid modifying during prediction
                for i in range(1, 21):
                    start_idx = i * 3
                    self.input_features[start_idx:start_idx + 3] -= wrist

                if len(self.input_features) != 63:
                    raise ValueError(f"Input features must have 63 elements, but got {len(self.input_features)}")

                input_tensor = torch.tensor(self.input_features, dtype=torch.float32).unsqueeze(0).to(self.device)
                self.predicted_letter, self.confidence = predict_letter(
                    self.model, input_tensor, CONFIG['class_names'], CONFIG['confidence_threshold']
                )

                if self.confidence > 0.7:
                    self.label_text.set(f"Predicted: {self.predicted_letter} ({self.confidence:.2f})")
                    self.letter_var.set(self.predicted_letter)
                else:
                    self.label_text.set("unknown")

        img_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.photo = ImageTk.PhotoImage(image=img_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.root.after(33, self.update_webcam)

    def record_sample(self):
        if self.input_features is not None:
            selected_letter = self.letter_var.get().lower()
            wrist_normalized_features = self.input_features.copy()
            wrist_normalized_features[0:3] = 0  # Set wrist coordinates to 0
            with open(CONFIG['output_csv'], 'a', newline='') as f:
                writer = csv.writer(f)
                row = [selected_letter] + wrist_normalized_features.tolist()
                writer.writerow(row)
            print(f"Recorded sample for letter '{selected_letter}' to {CONFIG['output_csv']}")

            try:
                df = pd.read_csv(CONFIG['output_csv'])
                df = df.sort_values(by='label')
                df.to_csv(CONFIG['output_csv'], index=False)
                print(f"Sorted {CONFIG['output_csv']} by label")
            except Exception as e:
                print(f"Error sorting CSV: {e}")
        else:
            print("Cannot record: No valid prediction or confidence too low.")

    def run(self):
        self.root.mainloop()

    def __del__(self):
        self.capture.release()
        print("Resources released. Exiting.")

def main():
    root = tk.Tk()
    app = SignLanguageGUI(root)
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        del app

if __name__ == "__main__":
    main()