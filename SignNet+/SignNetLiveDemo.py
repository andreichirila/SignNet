"""
🎬 SignNet Live Demo GUI
Real-time Sign Language Recognition with MediaPipe + Transformer

Author: Roman Schläpfer, Andrei Chirila
Date: 2025-12-01
Updated for: SignLanguageTransformer (Top-200)

Fixes:
- Filter __ON__ / __OFF__ tokens
- Suppress REGEN spam with confidence threshold
- Higher prediction threshold
"""

import cv2
import numpy as np
import torch
import mediapipe as mp
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import sys

# Add Architecture path
sys.path.append(str(Path(__file__).parent / "Architecture"))


# ============================================================================
# 🔧 CONFIGURATION
# ============================================================================

@dataclass
class DemoConfig:
    """Configuration for the demo."""
    # Model
    model_path: str = "D:/OST/SignNet/SignNet+/HistoryModels/2025/12/01/Models/best_model.pt"
    vocab_path: str = "D:/OST/SignNet/SignNet+/data_analysis_comprehensive/top200_glosses.csv"

    # Model architecture (must match training)
    num_landmarks: int = 543
    landmark_dim: int = 2
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    gcn_hidden_dims: List[int] = None

    # Inference
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    buffer_size: int = 60  # Frames to accumulate before prediction
    min_frames: int = 20   # Minimum frames for prediction
    prediction_threshold: float = 0.4  # Erhöht von 0.3

    # Spam suppression
    filter_tokens: List[str] = None  # Tokens to always filter out
    suppress_classes: Dict[str, float] = None  # Classes with confidence thresholds

    # Display
    window_width: int = 1400
    window_height: int = 800
    camera_width: int = 1280
    camera_height: int = 720

    def __post_init__(self):
        if self.gcn_hidden_dims is None:
            self.gcn_hidden_dims = [64, 128, 256]

        # Tokens to always filter out
        if self.filter_tokens is None:
            self.filter_tokens = [
                '<PAD>', '<BLANK>', '<UNK>',
                '__ON__', '__OFF__'  # Satz-Anfang/Ende - nicht relevant für Demo
            ]

        # Classes that need higher confidence to show
        if self.suppress_classes is None:
            self.suppress_classes = {
                'REGEN': 0.5,      # Nur bei >50% Confidence zeigen
                'IX': 0.5,         # Index/Pointer - oft falsch
                'KOENNEN': 0.45,
                'KOMMEN': 0.45,
            }


# ============================================================================
# 📖 VOCABULARY
# ============================================================================

class Vocabulary:
    """Vocabulary for gloss encoding/decoding."""

    def __init__(self):
        self.gloss2idx: Dict[str, int] = {}
        self.idx2gloss: Dict[int, str] = {}

        # Special tokens
        self.pad_token = "<PAD>"
        self.blank_token = "<BLANK>"
        self.unk_token = "<UNK>"

        # Add special tokens
        self.gloss2idx[self.pad_token] = 0
        self.gloss2idx[self.blank_token] = 1
        self.gloss2idx[self.unk_token] = 2

        self.idx2gloss[0] = self.pad_token
        self.idx2gloss[1] = self.blank_token
        self.idx2gloss[2] = self.unk_token

    @classmethod
    def from_csv(cls, csv_path: str) -> 'Vocabulary':
        """Load vocabulary from top-k glosses CSV."""
        vocab = cls()

        with open(csv_path, 'r', encoding='utf-8') as f:
            # Skip header
            next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 1:
                    gloss = parts[0].strip()
                    if gloss and gloss not in vocab.gloss2idx:
                        idx = len(vocab.gloss2idx)
                        vocab.gloss2idx[gloss] = idx
                        vocab.idx2gloss[idx] = gloss

        return vocab

    def decode(self, indices: List[int]) -> List[str]:
        """Decode indices to glosses."""
        return [self.idx2gloss.get(idx, self.unk_token) for idx in indices]

    @property
    def vocab_size(self) -> int:
        return len(self.gloss2idx)

    @property
    def blank_idx(self) -> int:
        return 1


# ============================================================================
# 🧠 MODEL WRAPPER
# ============================================================================

class SignNetModel:
    """Load and run SignLanguageTransformer for inference."""

    def __init__(self, config: DemoConfig):
        self.config = config
        self.device = torch.device(config.device)

        print(f"📦 Loading model from: {config.model_path}")
        print(f"📖 Loading vocabulary from: {config.vocab_path}")

        # Load vocabulary
        self.vocab = Vocabulary.from_csv(config.vocab_path)
        print(f"   Vocabulary size: {self.vocab.vocab_size}")

        # Build model
        self.model = self._build_model()

        # Load checkpoint
        checkpoint = torch.load(
            config.model_path,
            map_location=self.device,
            weights_only=False
        )

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Get training info
        self.best_wer = checkpoint.get('wer', 'N/A')
        self.epoch = checkpoint.get('epoch', 'N/A')

        print(f"✅ Model loaded!")
        print(f"   Device: {self.device}")
        print(f"   Best WER: {self.best_wer}")
        print(f"   Epoch: {self.epoch}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Filter tokens: {config.filter_tokens}")
        print(f"   Suppress classes: {config.suppress_classes}")

    def _build_model(self):
        """Build the model architecture."""
        from Model import SignLanguageTransformer
        from Config import ModelConfig

        # Create config matching training
        model_config = ModelConfig(
            num_landmarks=self.config.num_landmarks,
            landmark_dim=self.config.landmark_dim,
            num_classes=self.vocab.vocab_size,
            gcn_input_dim=self.config.landmark_dim,
            gcn_hidden_dims=self.config.gcn_hidden_dims,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            d_ff=self.config.d_ff,
            max_seq_length=214,
            device=self.config.device,
        )

        model = SignLanguageTransformer(model_config)
        model = model.to(self.device)

        return model

    def predict(self, landmarks: np.ndarray) -> Tuple[List[str], List[float], float]:
        """
        Predict glosses from landmarks.

        Args:
            landmarks: Array of shape (T, 543, 2) - sequence of landmarks

        Returns:
            glosses: List of predicted glosses (filtered)
            confidences: List of confidence scores per gloss
            avg_confidence: Average confidence score
        """
        if len(landmarks) < self.config.min_frames:
            return [], [], 0.0

        with torch.no_grad():
            # Prepare input: (T, 543, 2) -> (1, T, 543, 2)
            x = torch.FloatTensor(landmarks).unsqueeze(0).to(self.device)
            lengths = torch.LongTensor([len(landmarks)]).to(self.device)

            # Forward pass
            log_probs, output_lengths = self.model(x, lengths)

            # Get probabilities
            probs = torch.exp(log_probs)  # (1, T, vocab_size)

            # Greedy CTC decoding
            predictions = log_probs.argmax(dim=-1).squeeze(0)  # (T,)

            # Remove blanks and consecutive duplicates
            decoded = []
            decoded_confidences = []
            prev_token = -1

            for t, token in enumerate(predictions.tolist()):
                if token != prev_token and token != self.vocab.blank_idx:
                    if token != 0:  # Skip PAD
                        decoded.append(token)
                        decoded_confidences.append(probs[0, t, token].item())
                prev_token = token

            # Decode to glosses
            glosses = self.vocab.decode(decoded)

            # ============================================================
            # 🔧 FILTERING & SUPPRESSION
            # ============================================================

            filtered_glosses = []
            filtered_confidences = []

            for gloss, conf in zip(glosses, decoded_confidences):
                # 1. Skip always-filtered tokens
                if gloss in self.config.filter_tokens:
                    continue

                # 2. Check suppressed classes (need higher confidence)
                if gloss in self.config.suppress_classes:
                    min_conf = self.config.suppress_classes[gloss]
                    if conf < min_conf:
                        continue  # Skip if confidence too low

                # 3. General threshold
                if conf < self.config.prediction_threshold:
                    continue

                filtered_glosses.append(gloss)
                filtered_confidences.append(conf)

            # Average confidence
            avg_confidence = np.mean(filtered_confidences) if filtered_confidences else 0.0

            return filtered_glosses, filtered_confidences, avg_confidence


# ============================================================================
# 🎥 MEDIAPIPE TRACKER
# ============================================================================

class MediaPipeTracker:
    """Track face, pose, and hands with MediaPipe."""

    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize holistic model
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Landmark counts
        self.num_hand_landmarks = 21
        self.num_pose_landmarks = 33
        self.num_face_landmarks = 468

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray, any]:
        """
        Process frame and extract landmarks.

        Returns:
            landmarks: Array of shape (543, 2) or None if no detection
            annotated_frame: Frame with drawn landmarks
            results: MediaPipe results object
        """
        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Process
        results = self.holistic.process(image_rgb)

        # Convert back to BGR for OpenCV
        image_rgb.flags.writeable = True
        annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        self._draw_landmarks(annotated, results)

        # Extract landmarks (543, 2) - only X, Y
        landmarks = self._extract_landmarks(results)

        return landmarks, annotated, results

    def _draw_landmarks(self, frame: np.ndarray, results):
        """Draw all landmarks on frame."""
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )

        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )

    def _extract_landmarks(self, results) -> Optional[np.ndarray]:
        """Extract landmarks: (543, 2)"""
        landmarks = np.zeros((543, 2), dtype=np.float32)

        # ⚠️ NUR wenn Hand erkannt wird!
        has_hand = False

        # Left hand: indices 0-20
        if results.left_hand_landmarks:
            has_hand = True  # ← Geändert
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                landmarks[i, 0] = lm.x
                landmarks[i, 1] = lm.y

        # Right hand: indices 21-41
        if results.right_hand_landmarks:
            has_hand = True  # ← Geändert
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                landmarks[21 + i, 0] = lm.x
                landmarks[21 + i, 1] = lm.y

        # Pose: indices 42-74
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                landmarks[42 + i, 0] = lm.x
                landmarks[42 + i, 1] = lm.y

        # Face: indices 75-542
        if results.face_landmarks:
            for i, lm in enumerate(results.face_landmarks.landmark):
                landmarks[75 + i, 0] = lm.x
                landmarks[75 + i, 1] = lm.y

        # Nur returnen wenn Hand erkannt!
        return landmarks if has_hand else None


# ============================================================================
# 🖼️ GUI APPLICATION
# ============================================================================

class SignNetGUI:
    """Main GUI application for real-time sign language recognition."""

    def __init__(self, config: DemoConfig):
        self.config = config

        # Initialize window
        self.root = tk.Tk()
        self.root.title("SignNet Live Demo - Top-200 Transformer")
        self.root.geometry(f"{config.window_width}x{config.window_height}")
        self.root.configure(bg='#1e1e1e')

        # Load model
        print("\n" + "="*60)
        print("🎬 SignNet Live Demo")
        print("="*60)
        self.model = SignNetModel(config)

        # Initialize tracker
        self.tracker = MediaPipeTracker()

        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)

        # State
        self.running = False
        self.landmark_buffer = deque(maxlen=config.buffer_size)
        self.current_glosses: List[str] = []
        self.current_confidences: List[float] = []
        self.current_avg_confidence = 0.0
        self.fps = 0.0
        self.frame_times = deque(maxlen=30)
        self.prediction_history = deque(maxlen=10)  # Keep last 10 predictions
        self.gloss_history: List[str] = []  # Accumulated glosses

        # Setup UI
        self._setup_ui()

        # Start video thread
        self.running = True
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()

    def _setup_ui(self):
        """Setup user interface."""

        # Title
        title = tk.Label(
            self.root,
            text="🎬 SignNet Live Recognition (Top-200 Transformer)",
            font=('Arial', 20, 'bold'),
            bg='#1e1e1e',
            fg='#ffffff'
        )
        title.pack(pady=10)

        # Main container
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left panel - Video
        video_frame = tk.Frame(main_frame, bg='#2d2d2d', relief=tk.RAISED, borderwidth=2)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.video_label = tk.Label(video_frame, bg='#000000')
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right panel - Info
        info_frame = tk.Frame(main_frame, bg='#2d2d2d', relief=tk.RAISED, borderwidth=2, width=400)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        info_frame.pack_propagate(False)

        # Info title
        tk.Label(
            info_frame,
            text="📊 Recognition Results",
            font=('Arial', 16, 'bold'),
            bg='#2d2d2d',
            fg='#ffffff'
        ).pack(pady=10)

        # Predicted sequence (accumulated)
        pred_frame = tk.Frame(info_frame, bg='#3d3d3d', relief=tk.SUNKEN, borderwidth=2)
        pred_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            pred_frame,
            text="Accumulated Glosses:",
            font=('Arial', 12),
            bg='#3d3d3d',
            fg='#cccccc'
        ).pack(pady=5)

        self.pred_label = tk.Label(
            pred_frame,
            text="---",
            font=('Arial', 14, 'bold'),
            bg='#3d3d3d',
            fg='#00ff00',
            wraplength=350,
            justify='center'
        )
        self.pred_label.pack(pady=10)

        # Clear button
        self.clear_btn = tk.Button(
            pred_frame,
            text="🗑️ Clear History",
            font=('Arial', 10),
            command=self._clear_history,
            bg='#555555',
            fg='#ffffff'
        )
        self.clear_btn.pack(pady=5)

        # Current gloss (large)
        current_frame = tk.Frame(info_frame, bg='#3d3d3d', relief=tk.SUNKEN, borderwidth=2)
        current_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            current_frame,
            text="Current Sign:",
            font=('Arial', 12),
            bg='#3d3d3d',
            fg='#cccccc'
        ).pack(pady=5)

        self.current_label = tk.Label(
            current_frame,
            text="---",
            font=('Arial', 36, 'bold'),
            bg='#3d3d3d',
            fg='#00aaff'
        )
        self.current_label.pack(pady=10)

        # Confidence bar
        conf_frame = tk.Frame(info_frame, bg='#3d3d3d', relief=tk.SUNKEN, borderwidth=2)
        conf_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            conf_frame,
            text="Confidence:",
            font=('Arial', 12),
            bg='#3d3d3d',
            fg='#cccccc'
        ).pack(pady=5)

        self.conf_label = tk.Label(
            conf_frame,
            text="0%",
            font=('Arial', 20, 'bold'),
            bg='#3d3d3d',
            fg='#00aaff'
        )
        self.conf_label.pack(pady=5)

        self.conf_bar = ttk.Progressbar(
            conf_frame,
            length=350,
            mode='determinate',
            maximum=100
        )
        self.conf_bar.pack(pady=10)

        # Stats
        stats_frame = tk.Frame(info_frame, bg='#3d3d3d', relief=tk.SUNKEN, borderwidth=2)
        stats_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            stats_frame,
            text="Statistics:",
            font=('Arial', 12, 'bold'),
            bg='#3d3d3d',
            fg='#cccccc'
        ).pack(pady=5)

        self.fps_label = tk.Label(
            stats_frame,
            text="FPS: 0",
            font=('Arial', 11),
            bg='#3d3d3d',
            fg='#ffffff'
        )
        self.fps_label.pack(pady=2)

        self.buffer_label = tk.Label(
            stats_frame,
            text=f"Buffer: 0/{self.config.buffer_size} frames",
            font=('Arial', 11),
            bg='#3d3d3d',
            fg='#ffffff'
        )
        self.buffer_label.pack(pady=2)

        self.device_label = tk.Label(
            stats_frame,
            text=f"Device: {self.config.device}",
            font=('Arial', 11),
            bg='#3d3d3d',
            fg='#ffffff'
        )
        self.device_label.pack(pady=2)

        wer_text = f"WER: {self.model.best_wer:.1%}" if isinstance(self.model.best_wer, float) else "WER: N/A"
        self.model_label = tk.Label(
            stats_frame,
            text=f"Model: Top-200 | {wer_text}",
            font=('Arial', 11),
            bg='#3d3d3d',
            fg='#ffffff'
        )
        self.model_label.pack(pady=2)

        self.filter_label = tk.Label(
            stats_frame,
            text=f"Threshold: {self.config.prediction_threshold:.0%}",
            font=('Arial', 11),
            bg='#3d3d3d',
            fg='#ffffff'
        )
        self.filter_label.pack(pady=2)

        # Instructions
        inst_frame = tk.Frame(info_frame, bg='#3d3d3d', relief=tk.SUNKEN, borderwidth=2)
        inst_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(
            inst_frame,
            text="📝 Instructions:",
            font=('Arial', 12, 'bold'),
            bg='#3d3d3d',
            fg='#cccccc'
        ).pack(pady=5)

        instructions = [
            "✅ Position yourself in front of camera",
            "✅ Ensure good lighting",
            "✅ Perform German sign language gestures",
            "✅ Wait for buffer to fill (60 frames)",
            "✅ Green text = recognized!",
            "",
            "🔧 Filtered: __ON__, __OFF__",
            "🔧 REGEN needs >50% confidence",
            "",
            "Press Q or close window to quit"
        ]

        for inst in instructions:
            tk.Label(
                inst_frame,
                text=inst,
                font=('Arial', 10),
                bg='#3d3d3d',
                fg='#ffffff',
                anchor='w',
                justify='left'
            ).pack(anchor='w', padx=10, pady=2)

        # Status bar
        self.status_label = tk.Label(
            self.root,
            text="🟢 Live Recognition Active",
            font=('Arial', 12),
            bg='#1e1e1e',
            fg='#00ff00'
        )
        self.status_label.pack(pady=5)

    def _clear_history(self):
        """Clear accumulated gloss history."""
        self.gloss_history = []
        self.prediction_history.clear()
        self.current_glosses = []
        self.current_confidences = []
        self.current_avg_confidence = 0.0
        self._update_info()

    def _video_loop(self):
        """Main video processing loop."""
        last_prediction = ""
        repeat_count = 0

        while self.running:
            start_time = time.time()

            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Process with MediaPipe
            landmarks, annotated, results = self.tracker.process_frame(frame)

            # Add to buffer if we have hand detection
            if landmarks is not None:
                self.landmark_buffer.append(landmarks)

            # Predict if buffer has enough frames
            if len(self.landmark_buffer) >= self.config.min_frames:
                # Stack landmarks: (T, 543, 2)
                sequence = np.stack(list(self.landmark_buffer))

                # Predict
                glosses, confidences, avg_conf = self.model.predict(sequence)

                if glosses:
                    self.current_glosses = glosses
                    self.current_confidences = confidences
                    self.current_avg_confidence = avg_conf

                    # Add to history (avoid duplicates)
                    current = glosses[-1] if glosses else ""

                    if current and current != last_prediction:
                        self.gloss_history.append(current)
                        # Keep only last 20 glosses in history
                        if len(self.gloss_history) > 20:
                            self.gloss_history = self.gloss_history[-20:]
                        last_prediction = current
                        repeat_count = 0
                    elif current == last_prediction:
                        repeat_count += 1
                        # After many repeats, allow adding again
                        if repeat_count > 30:  # ~1 second of repeats
                            repeat_count = 0

            # Add overlay
            annotated = self._add_overlay(annotated)

            # Convert for Tkinter
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_pil = frame_pil.resize((960, 540), Image.Resampling.LANCZOS)
            frame_tk = ImageTk.PhotoImage(frame_pil)

            # Update video
            self.video_label.configure(image=frame_tk)
            self.video_label.image = frame_tk

            # Calculate FPS
            elapsed = time.time() - start_time
            self.frame_times.append(elapsed)
            if len(self.frame_times) > 0:
                self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))

            # Update info
            self._update_info()

            # Small delay
            time.sleep(0.001)

        # Cleanup
        self.cap.release()
        self.tracker.close()

    def _add_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add text overlay to frame."""
        h, w = frame.shape[:2]

        # Top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        cv2.putText(frame, "SignNet Live (Top-200)", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w - 150, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Prediction overlay at bottom
        if self.current_glosses:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h-120), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

            # Show sequence (last 5)
            gloss_text = " → ".join(self.current_glosses[-5:])
            cv2.putText(frame, gloss_text, (20, h-70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Current gloss (large)
            if self.current_glosses:
                current = self.current_glosses[-1]
                # Color based on confidence
                conf = self.current_confidences[-1] if self.current_confidences else 0
                if conf > 0.6:
                    color = (0, 255, 0)  # Green - high confidence
                elif conf > 0.4:
                    color = (0, 200, 255)  # Orange - medium
                else:
                    color = (0, 100, 255)  # Red - low

                cv2.putText(frame, current, (20, h-25),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # Confidence
            conf_percent = int(self.current_avg_confidence * 100)
            cv2.putText(frame, f"{conf_percent}%", (w - 100, h-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        return frame

    def _update_info(self):
        """Update info panel."""
        # Accumulated glosses
        if self.gloss_history:
            display_history = self.gloss_history[-10:]  # Last 10
            self.pred_label.config(text=" → ".join(display_history))
        else:
            self.pred_label.config(text="---")

        # Current gloss
        if self.current_glosses:
            self.current_label.config(text=self.current_glosses[-1])

            # Confidence
            conf_percent = int(self.current_avg_confidence * 100)
            self.conf_label.config(text=f"{conf_percent}%")
            self.conf_bar['value'] = conf_percent

            # Color based on confidence
            if conf_percent > 60:
                self.current_label.config(fg='#00ff00')  # Green
            elif conf_percent > 40:
                self.current_label.config(fg='#00aaff')  # Blue
            else:
                self.current_label.config(fg='#ffaa00')  # Orange
        else:
            self.current_label.config(text="---", fg='#00aaff')
            self.conf_label.config(text="0%")
            self.conf_bar['value'] = 0

        # Stats
        self.fps_label.config(text=f"FPS: {self.fps:.1f}")
        self.buffer_label.config(text=f"Buffer: {len(self.landmark_buffer)}/{self.config.buffer_size} frames")

    def run(self):
        """Run the application."""
        print("\n🎥 Starting live demo...")
        print("   Press Q or close window to quit\n")

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Keyboard binding
        self.root.bind('<q>', lambda e: self._on_close())
        self.root.bind('<Q>', lambda e: self._on_close())
        self.root.bind('<c>', lambda e: self._clear_history())
        self.root.bind('<C>', lambda e: self._clear_history())

        self.root.mainloop()

    def _on_close(self):
        """Handle window close."""
        self.running = False
        time.sleep(0.5)
        self.root.destroy()


# ============================================================================
# 🚀 MAIN
# ============================================================================

def main():
    """Main entry point."""
    print("="*60)
    print("🎬 SignNet Live Demo")
    print("   Model: SignLanguageTransformer (Top-200)")
    print("   Dataset: RWTH-PHOENIX-2014")
    print("="*60)

    # Configuration
    config = DemoConfig()

    # Check files exist
    if not Path(config.model_path).exists():
        print(f"\n❌ Model not found: {config.model_path}")
        print("   Please check the path!")
        return

    if not Path(config.vocab_path).exists():
        print(f"\n❌ Vocabulary not found: {config.vocab_path}")
        print("   Please check the path!")
        return

    # Run demo
    try:
        app = SignNetGUI(config)
        app.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 Demo closed!")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()