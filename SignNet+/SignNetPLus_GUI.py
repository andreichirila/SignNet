import cv2
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import os
import glob
import csv
import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk
import time
from collections import deque
import mediapipe as mp  # Für Holistic
from SignNetPlusModel import SignBERTBiGRU

# CTC-Dekodierung
def ctc_decode(log_probs, blank=0):
    argmax = torch.argmax(log_probs, dim=2)
    prev = argmax[0]
    sequence = [prev.item()]
    for t in range(1, log_probs.size(0)):
        current = argmax[t].item()
        if current != blank and current != prev:
            sequence.append(current)
        prev = current
    return sequence


CONFIG = {
    'num_classes': 100,  # Passe an dein Gloss-Vokabular an
    'class_names': ['<BLANK>', '<PAD>'] + ['gehen', 'essen', 'trinken'],  # Erweitere mit DGS-Glosses
    'confidence_threshold': 0.5,
    'model_dir': './models_dynamic',
    'output_csv': './landmark_datasets/dynamic_samples_dgs.csv',
    'frame_width': 640,
    'frame_height': 480,
    'sequence_length': 32,
    'hold_time_threshold': 2.0,
    'fps': 30
}


def load_latest_dynamic_model(model_dir, num_classes, device):
    model_files = glob.glob(os.path.join(model_dir, "*.pth"))
    if not model_files:
        raise ValueError(f"No trained dynamic model found in {model_dir}.")
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Loading latest dynamic model: {latest_model}")
    model = SignBERTBiGRU(vocab_size=num_classes).to(device)
    model.load_state_dict(torch.load(latest_model, map_location=device))
    model.eval()
    return model


def predict_gloss_sequence(model, sequence_tensor, class_names, lengths, threshold):
    with torch.no_grad():
        log_probs = model(sequence_tensor, lengths)
        decoded = ctc_decode(log_probs)
        probs = F.softmax(log_probs.transpose(0, 1), dim=-1)
        conf = torch.max(probs).mean().item()
        gloss_seq = [class_names[i] for i in decoded if i != 0]
        gloss = ' '.join(gloss_seq) if gloss_seq else "unknown"
        return gloss, conf if conf > threshold else 0.0


class DynamicSignLanguageGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic DGS Recognition with Holistic Detection")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = load_latest_dynamic_model(CONFIG['model_dir'], CONFIG['num_classes'], self.device)

        # MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['frame_width'])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['frame_height'])

        # CSV-Init
        if not os.path.exists(CONFIG['output_csv']):
            os.makedirs(os.path.dirname(CONFIG['output_csv']), exist_ok=True)
            with open(CONFIG['output_csv'], 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['label'] + [f'frame_{t}_coord_{i}' for t in range(CONFIG['sequence_length']) for i in
                                      range(1659)]
                writer.writerow(header)

        # GUI-Elemente
        self.canvas = tk.Canvas(root, width=CONFIG['frame_width'], height=CONFIG['frame_height'])
        self.canvas.pack()
        self.label_text = tk.StringVar(value="No Signs Detected")
        self.label_display = tk.Label(root, textvariable=self.label_text, font=("Arial", 14))
        self.label_display.pack()
        self.sentence_text = tk.StringVar(value="Sentence: ")
        self.sentence_display = tk.Label(root, textvariable=self.sentence_text, font=("Arial", 14))
        self.sentence_display.pack()
        self.record_button = tk.Button(root, text="Record Dynamic Sample", command=self.record_sample)
        self.record_button.pack()
        self.root.bind('r', lambda event: self.record_sample())

        self.landmark_buffer = deque(maxlen=CONFIG['sequence_length'])
        self.predicted_gloss = None
        self.confidence = 0.0
        self.sentence = []
        self.last_gloss = None
        self.gloss_start_time = None

        self.update_webcam()

    def extract_landmarks(self, results, img_h, img_w):
        """Extrahiert kombinierte Landmarken: Face + Pose + Left/Right Hands → 1659D"""
        landmarks = []

        # Face Mesh (468 * 3 = 1404D)
        if results.face_landmarks:
            for lm in results.face_landmarks.landmark:
                landmarks.extend([lm.x * img_w, lm.y * img_h, lm.z * img_w])
        else:
            landmarks.extend([0.0] * 1404)

        # Pose (33 * 3 = 99D)
        if results.pose_landmarks:
            pose_lms = []
            for lm in results.pose_landmarks.landmark:
                pose_lms.extend([lm.x * img_w, lm.y * img_h, lm.z * img_w])
            # Normalisierung: Schulter-Mitte subtrahieren (Punkte 11+12)
            shoulder_center_x = (pose_lms[11 * 3] + pose_lms[12 * 3]) / 2
            shoulder_center_y = (pose_lms[11 * 3 + 1] + pose_lms[12 * 3 + 1]) / 2
            for i in range(99):
                if i % 3 == 0:  # x
                    pose_lms[i] -= shoulder_center_x
                elif i % 3 == 1:  # y
                    pose_lms[i] -= shoulder_center_y
            landmarks.extend(pose_lms)
        else:
            landmarks.extend([0.0] * 99)

        # Hands: Left + Right (je 21*3=63D, total 126D)
        left_hand = [0.0] * 63
        right_hand = [0.0] * 63
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                idx = i * 3
                left_hand[idx] = lm.x * img_w
                left_hand[idx + 1] = lm.y * img_h
                left_hand[idx + 2] = lm.z * img_w
            # Wrist-Normalisierung (Punkt 0)
            wrist = left_hand[0:3]
            for i in range(63):
                left_hand[i] -= wrist[i % 3]
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                idx = i * 3
                right_hand[idx] = lm.x * img_w
                right_hand[idx + 1] = lm.y * img_h
                right_hand[idx + 2] = lm.z * img_w
            wrist = right_hand[0:3]
            for i in range(63):
                right_hand[i] -= wrist[i % 3]
        landmarks.extend(left_hand)
        landmarks.extend(right_hand)

        # Padding/Schneiden auf 1659D
        while len(landmarks) < 1659:
            landmarks.append(0.0)
        landmarks = landmarks[:1659]
        return np.array(landmarks, dtype=np.float32)

    def normalize_sequence(self, sequence):
        return sequence  # Bereits pro-Frame normalisiert

    def update_webcam(self):
        ret, img = self.capture.read()
        if not ret:
            print("Camera error – exiting.")
            self.root.quit()
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)

        # Zeichne Landmarken
        self.mp_draw.draw_landmarks(img, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS)
        self.mp_draw.draw_landmarks(img, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        self.mp_draw.draw_landmarks(img, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        self.mp_draw.draw_landmarks(img, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

        self.label_text.set("No Signs Detected")
        self.predicted_gloss = None
        self.confidence = 0.0

        current_landmarks = self.extract_landmarks(results, CONFIG['frame_height'], CONFIG['frame_width'])
        if current_landmarks is not None:
            self.landmark_buffer.append(current_landmarks)

        if len(self.landmark_buffer) == CONFIG['sequence_length']:
            sequence = np.stack(list(self.landmark_buffer))
            sequence = self.normalize_sequence(sequence)

            input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            lengths = torch.tensor([CONFIG['sequence_length']], dtype=torch.long).to(self.device)

            self.predicted_gloss, self.confidence = predict_gloss_sequence(
                self.model, input_tensor, CONFIG['class_names'], lengths, CONFIG['confidence_threshold']
            )

            if self.confidence > CONFIG['confidence_threshold']:
                self.label_text.set(f"Predicted: {self.predicted_gloss} ({self.confidence:.2f})")

                if self.predicted_gloss == self.last_gloss:
                    if self.gloss_start_time is not None:
                        elapsed = time.time() - self.gloss_start_time
                        if elapsed >= CONFIG['hold_time_threshold']:
                            if self.predicted_gloss not in self.sentence:
                                self.sentence.append(self.predicted_gloss)
                                self.sentence_text.set(f"Sentence: {' '.join(self.sentence)}")
                                print(f"Added '{self.predicted_gloss}' to sentence")
                                self.gloss_start_time = time.time()
                else:
                    self.last_gloss = self.predicted_gloss
                    self.gloss_start_time = time.time()
            else:
                self.label_text.set("Low confidence")
                self.last_gloss = None
                self.gloss_start_time = None

        img_rgb_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb_display)
        self.photo = ImageTk.PhotoImage(image=img_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.root.after(1000 // CONFIG['fps'], self.update_webcam)

    def record_sample(self):
        if len(self.landmark_buffer) == CONFIG['sequence_length']:
            selected_gloss = self.predicted_gloss or input("Enter gloss label: ")
            flattened_list = [item for frame in self.landmark_buffer for item in frame]
            with open(CONFIG['output_csv'], 'a', newline='') as f:
                writer = csv.writer(f)
                row = [selected_gloss] + flattened_list
                writer.writerow(row)
            print(f"Recorded full holistic sample for '{selected_gloss}'")

            try:
                df = pd.read_csv(CONFIG['output_csv'])
                df = df.sort_values(by='label')
                df.to_csv(CONFIG['output_csv'], index=False)
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Need full sequence (32 frames).")

    def run(self):
        self.root.mainloop()

    def __del__(self):
        if hasattr(self, 'holistic'):
            self.holistic.close()
        if hasattr(self, 'capture'):
            self.capture.release()
        print("Resources released.")


def main():
    root = tk.Tk()
    app = DynamicSignLanguageGUI(root)
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        del app


if __name__ == "__main__":
    main()