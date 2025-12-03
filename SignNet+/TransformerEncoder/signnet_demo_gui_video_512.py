#!/usr/bin/env python3
"""
SignNet Live Demo GUI - WITH VIDEO MODE
Real-time Sign Language Recognition with MediaPipe + Transformer

Author: Roman Schläpfer, Andrei Chirila
Date: 2025-12-03
Updated: Added Video Mode for MP4 file prediction

Features:
- Live webcam recognition (original)
- Video file mode: Load MP4, predict frame-by-frame
- Side-by-side comparison with ground truth
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
import json
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import sys
import os


# ============================================================================
# 🔧 CONFIGURATION
# ============================================================================

@dataclass
class DemoConfig:
    """Configuration for the demo."""
    # Model paths - UPDATE THESE TO YOUR LOCAL PATHS
    model_path: str = "./models_balanced/sign_classifier_final_enhanced_r2.pth"
    vocab_path: str = "./models_balanced/main_vocab.json"

    # Alternative paths to try (will search in order)
    model_search_paths: List[str] = field(default_factory=lambda: [
        "./models_balanced/sign_classifier_final_enhanced_r2.pth",
    ])
    vocab_search_paths: List[str] = field(default_factory=lambda: [
        "./models_balanced/main_vocab.json"
    ])

    # Model architecture - ORIGINAL LARGER MODEL (512h/6L/8heads)
    # Use this for models trained with the original config
    input_size: int = 1659  # MediaPipe landmarks: hands(126) + face(1434) + pose(99)
    hidden_size: int = 512  # Original size
    num_layers: int = 6  # Original layers
    num_heads: int = 8  # Original heads
    dim_feedforward: int = 2048  # 4 * hidden_size
    dropout_rate: float = 0.0  # No dropout during inference
    attention_dropout: float = 0.0

    # Inference
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    buffer_size: int = 60  # Frames to accumulate before prediction (increased for video)
    min_frames: int = 8  # Minimum frames for prediction (reduced)
    prediction_threshold: float = 0.30  # Minimum confidence to show (reduced for testing)

    # Temporal smoothing
    smoothing_window: int = 3  # Average predictions over this many frames (reduced)
    stability_threshold: int = 2  # Need this many consecutive same predictions (reduced)

    # Spam suppression - classes that need higher confidence
    suppress_classes: Dict[str, float] = field(default_factory=lambda: {
        'REGEN': 0.50,
        'REGEN-PLUSPLUS': 0.50,
        'KOMMEN': 0.45,
        'HABEN': 0.45,
        'NOCH': 0.50,
        'DANN': 0.45,
    })

    # Display
    window_width: int = 1600  # Increased
    window_height: int = 950  # Increased
    camera_width: int = 1280
    camera_height: int = 720
    camera_id: int = 0

    # Video mode settings
    video_playback_speed: float = 0.5  # 0.5 = half speed (gives model more time)
    video_loop: bool = True  # Loop video playback

    # Filter tokens (internal markers + dropped classes)
    filter_tokens: List[str] = field(default_factory=lambda: [
        '<PAD>', '<BLANK>', '<UNK>', 'UNKNOWN',
        'SUEDRAUM',
        'HAUPTSAECHLICH',
    ])


# ============================================================================
# 🧠 MODEL DEFINITION
# ============================================================================

class TransformerSignClassifierWithHandedness(nn.Module):
    """Transformer encoder model with multi-task learning (sign + handedness)."""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int,
                 num_layers: int = 6, num_heads: int = 8, dim_feedforward: int = 2048,
                 dropout_rate: float = 0.0, attention_dropout: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.input_proj = nn.Linear(input_size, hidden_size)

        self.pos_embedding = nn.Parameter(torch.zeros(1, 2048, hidden_size))
        nn.init.normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=attention_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.fc_sign = nn.Linear(hidden_size, num_classes)
        self.fc_handedness = nn.Linear(hidden_size, 4)

    def forward(self, landmarks, src_key_padding_mask=None):
        B, T, D = landmarks.shape

        x = self.input_proj(landmarks)

        if T > self.pos_embedding.size(1):
            raise ValueError(f"Sequence length {T} exceeds max positional length")
        pos_emb = self.pos_embedding[:, :T, :]
        x = x + pos_emb

        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).float().unsqueeze(-1)
            x_masked = x * mask
            lengths = mask.sum(dim=1).clamp(min=1.0)
            pooled = x_masked.sum(dim=1) / lengths
        else:
            pooled = x.mean(dim=1)

        pooled = self.dropout(pooled)

        sign_logits = self.fc_sign(pooled)
        handedness_logits = self.fc_handedness(pooled)

        return sign_logits, handedness_logits


# ============================================================================
# 📖 VOCABULARY
# ============================================================================

class Vocabulary:
    """Vocabulary for gloss encoding/decoding."""

    def __init__(self):
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.num_classes: int = 0

    @classmethod
    def from_json(cls, json_path: str) -> 'Vocabulary':
        """Load vocabulary from JSON file."""
        vocab = cls()

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vocab.word_to_idx = data.get('word_to_idx', {})
        idx_to_word_raw = data.get('idx_to_word', {})
        vocab.idx_to_word = {int(k): v for k, v in idx_to_word_raw.items()}
        vocab.num_classes = data.get('num_classes', len(vocab.word_to_idx))

        return vocab

    def decode(self, idx: int) -> str:
        """Decode index to gloss."""
        return self.idx_to_word.get(idx, f"UNKNOWN_{idx}")

    def decode_top_k(self, probs: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Get top-k predictions with probabilities."""
        top_indices = np.argsort(probs)[::-1][:k]
        results = []
        for idx in top_indices:
            gloss = self.decode(idx)
            prob = probs[idx]
            if not gloss.startswith("UNKNOWN_"):
                results.append((gloss, prob))
        if len(results) < k:
            for idx in np.argsort(probs)[::-1][k:k + 5]:
                gloss = self.decode(idx)
                prob = probs[idx]
                if not gloss.startswith("UNKNOWN_"):
                    results.append((gloss, prob))
                if len(results) >= k:
                    break
        return results[:k]


# ============================================================================
# 🧠 MODEL WRAPPER
# ============================================================================

class SignNetModel:
    """Load and run TransformerSignClassifierWithHandedness for inference."""

    def __init__(self, config: DemoConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.vocab = None
        self.loaded = False

    def load(self, model_path: str, vocab_path: str) -> bool:
        """Load model and vocabulary."""
        try:
            print(f"📦 Loading model from: {model_path}")
            print(f"📖 Loading vocabulary from: {vocab_path}")

            self.vocab = Vocabulary.from_json(vocab_path)
            print(f"   Vocabulary size (from JSON): {self.vocab.num_classes}")

            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            state_dict = checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("_orig_mod."):
                    new_state_dict[k[10:]] = v
                else:
                    new_state_dict[k] = v

            fc_sign_weight = new_state_dict.get('fc_sign.weight')
            if fc_sign_weight is not None:
                num_classes_from_model = fc_sign_weight.shape[0]
                print(f"   Num classes (from model): {num_classes_from_model}")
            else:
                num_classes_from_model = self.vocab.num_classes
                print(f"   ⚠️ Could not detect num_classes from model, using vocab: {num_classes_from_model}")

            self.model = TransformerSignClassifierWithHandedness(
                input_size=self.config.input_size,
                hidden_size=self.config.hidden_size,
                num_classes=num_classes_from_model,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                dim_feedforward=self.config.dim_feedforward,
                dropout_rate=0.0,
                attention_dropout=0.0
            ).to(self.device)

            self.model.load_state_dict(new_state_dict)
            self.model.eval()

            self.vocab.num_classes = num_classes_from_model

            print(f"✅ Model loaded successfully!")
            print(f"   Device: {self.device}")
            print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"   Classes: {num_classes_from_model}")

            self.loaded = True
            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict(self, landmarks: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]], str]:
        """Predict gloss from landmarks."""
        if not self.loaded or len(landmarks) < self.config.min_frames:
            return "", 0.0, [], "NONE"

        with torch.no_grad():
            x = torch.FloatTensor(landmarks).unsqueeze(0).to(self.device)

            sign_logits, handedness_logits = self.model(x)

            sign_probs = F.softmax(sign_logits, dim=-1).squeeze(0).cpu().numpy()
            hand_probs = F.softmax(handedness_logits, dim=-1).squeeze(0).cpu().numpy()

            top_idx = np.argmax(sign_probs)
            confidence = sign_probs[top_idx]
            predicted_gloss = self.vocab.decode(top_idx)

            top_k = self.vocab.decode_top_k(sign_probs, k=10)

            hand_idx = np.argmax(hand_probs)
            handedness_map = {0: "LEFT", 1: "RIGHT", 2: "BOTH", 3: "NONE"}
            handedness = handedness_map.get(hand_idx, "NONE")

            # Filtering
            if predicted_gloss.startswith("UNKNOWN"):
                return "", 0.0, top_k, handedness

            if predicted_gloss in self.config.filter_tokens:
                return "", 0.0, top_k, handedness

            if predicted_gloss in self.config.suppress_classes:
                min_conf = self.config.suppress_classes[predicted_gloss]
                if confidence < min_conf:
                    return "", confidence, top_k, handedness

            if confidence < self.config.prediction_threshold:
                return "", confidence, top_k, handedness

            return predicted_gloss, confidence, top_k, handedness


# ============================================================================
# 🎥 MEDIAPIPE TRACKER
# ============================================================================

class MediaPipeTracker:
    """Track face, pose, and hands with MediaPipe Holistic."""

    NUM_HAND_LANDMARKS = 21
    NUM_POSE_LANDMARKS = 33
    NUM_FACE_LANDMARKS = 478

    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray, bool]:
        """Process frame and extract landmarks."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        results = self.holistic.process(image_rgb)

        image_rgb.flags.writeable = True
        annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        self._draw_landmarks(annotated, results)

        landmarks, has_hands = self._extract_landmarks(results)

        return landmarks, annotated, has_hands

    def _draw_landmarks(self, frame: np.ndarray, results):
        """Draw all landmarks on frame."""
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
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
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

    def _extract_landmarks(self, results) -> Tuple[Optional[np.ndarray], bool]:
        """Extract landmarks: [hands (126), face (1434), pose (99)] = 1659 features"""
        landmarks = np.zeros(1659, dtype=np.float32)
        has_hands = False

        if results.left_hand_landmarks:
            has_hands = True
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                landmarks[i * 3] = lm.x
                landmarks[i * 3 + 1] = lm.y
                landmarks[i * 3 + 2] = lm.z

        if results.right_hand_landmarks:
            has_hands = True
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                landmarks[63 + i * 3] = lm.x
                landmarks[63 + i * 3 + 1] = lm.y
                landmarks[63 + i * 3 + 2] = lm.z

        if results.face_landmarks:
            for i, lm in enumerate(results.face_landmarks.landmark):
                landmarks[126 + i * 3] = lm.x
                landmarks[126 + i * 3 + 1] = lm.y
                landmarks[126 + i * 3 + 2] = lm.z

        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                landmarks[1560 + i * 3] = lm.x
                landmarks[1560 + i * 3 + 1] = lm.y
                landmarks[1560 + i * 3 + 2] = lm.z

        return landmarks if has_hands else None, has_hands

    def close(self):
        """Release resources."""
        self.holistic.close()


# ============================================================================
# 🖼️ GUI APPLICATION WITH VIDEO MODE
# ============================================================================

class SignNetGUI:
    """Main GUI application with webcam AND video file support."""

    def __init__(self, config: DemoConfig):
        self.config = config

        # Initialize window
        self.root = tk.Tk()
        self.root.title("SignNet Demo - Webcam & Video Mode")
        self.root.geometry(f"{config.window_width}x{config.window_height}")
        self.root.configure(bg='#1a1a2e')

        # Model
        self.model = SignNetModel(config)

        # Tracker
        self.tracker = None

        # Video/Camera
        self.cap = None
        self.video_path = None
        self.video_fps = 25.0
        self.video_total_frames = 0
        self.video_current_frame = 0

        # Mode: "webcam" or "video"
        self.mode = "webcam"

        # State
        self.running = False
        self.paused = False
        self.landmark_buffer = deque(maxlen=config.buffer_size)
        self.prediction_history = deque(maxlen=config.smoothing_window)
        self.stable_prediction = ""
        self.stable_count = 0
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.current_top_k = []
        self.current_handedness = "NONE"
        self.fps = 0.0
        self.frame_times = deque(maxlen=30)
        self.gloss_history: List[str] = []

        # Ground truth for video mode
        self.ground_truth = ""

        # Setup UI
        self._setup_ui()

        # Try to auto-load model
        self.root.after(100, self._try_auto_load)

    def _find_file(self, search_paths: List[str]) -> Optional[str]:
        """Search for file in multiple paths."""
        for path in search_paths:
            if Path(path).exists():
                return path
        return None

    def _try_auto_load(self):
        """Try to automatically load model from default paths."""
        model_path = self._find_file(self.config.model_search_paths)
        vocab_path = self._find_file(self.config.vocab_search_paths)

        if model_path and vocab_path:
            print(f"🔍 Found model: {model_path}")
            print(f"🔍 Found vocab: {vocab_path}")
            success = self.model.load(model_path, vocab_path)

            if success:
                self.model_status.config(
                    text=f"✅ Loaded: {self.model.vocab.num_classes} classes",
                    fg='#4ecca3'
                )
                self.webcam_btn.config(state=tk.NORMAL)
                self.video_btn.config(state=tk.NORMAL)
                self.analyze_btn.config(state=tk.NORMAL)
                self.status_label.config(text="[OK] Model loaded - Choose mode", fg='#4ecca3')
            else:
                self.model_status.config(text="❌ Auto-load failed", fg='#ff6b6b')
        else:
            print("⚠️ Model files not found in default paths")
            self.model_status.config(text="⚠️ Click 'Load Model'", fg='#ffd93d')

    def _setup_ui(self):
        """Setup user interface."""

        # Title bar
        title_frame = tk.Frame(self.root, bg='#16213e', height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        title = tk.Label(
            title_frame,
            text="SignNet - Webcam & Video Mode",
            font=('Arial', 22, 'bold'),
            bg='#16213e',
            fg='#e94560'
        )
        title.pack(pady=15)

        # Main container
        main_frame = tk.Frame(self.root, bg='#1a1a2e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Left panel - Video
        video_frame = tk.Frame(main_frame, bg='#0f3460', relief=tk.RAISED, borderwidth=2)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.video_label = tk.Label(video_frame, bg='#000000')
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Video controls (for video mode)
        self.video_controls_frame = tk.Frame(video_frame, bg='#16213e', height=50)
        self.video_controls_frame.pack(fill=tk.X, padx=5, pady=5)

        self.play_pause_btn = tk.Button(
            self.video_controls_frame,
            text="Pause",
            font=('Arial', 10),
            command=self._toggle_pause,
            bg='#e94560',
            fg='#ffffff',
            state=tk.DISABLED
        )
        self.play_pause_btn.pack(side=tk.LEFT, padx=5)

        self.restart_btn = tk.Button(
            self.video_controls_frame,
            text="🔄 Restart",
            font=('Arial', 10),
            command=self._restart_video,
            bg='#4ecca3',
            fg='#1a1a2e',
            state=tk.DISABLED
        )
        self.restart_btn.pack(side=tk.LEFT, padx=5)

        self.progress_label = tk.Label(
            self.video_controls_frame,
            text="Frame: 0 / 0",
            font=('Arial', 10),
            bg='#16213e',
            fg='#ffffff'
        )
        self.progress_label.pack(side=tk.LEFT, padx=20)

        # Speed control
        tk.Label(
            self.video_controls_frame,
            text="Speed:",
            font=('Arial', 10),
            bg='#16213e',
            fg='#ffffff'
        ).pack(side=tk.LEFT, padx=(10, 2))

        self.speed_var = tk.DoubleVar(value=0.5)
        self.speed_scale = tk.Scale(
            self.video_controls_frame,
            from_=0.1,
            to=2.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.speed_var,
            command=self._update_speed,
            bg='#16213e',
            fg='#ffffff',
            highlightthickness=0,
            length=100
        )
        self.speed_scale.pack(side=tk.LEFT, padx=2)

        self.speed_label = tk.Label(
            self.video_controls_frame,
            text="0.5x",
            font=('Arial', 10),
            bg='#16213e',
            fg='#4ecca3'
        )
        self.speed_label.pack(side=tk.LEFT, padx=2)

        # Mirror toggle for video
        self.mirror_var = tk.BooleanVar(value=False)
        self.mirror_check = tk.Checkbutton(
            self.video_controls_frame,
            text="🔄 Mirror",
            variable=self.mirror_var,
            font=('Arial', 10),
            bg='#16213e',
            fg='#ffffff',
            selectcolor='#0f3460',
            activebackground='#16213e'
        )
        self.mirror_check.pack(side=tk.LEFT, padx=10)

        # Ground truth input
        tk.Label(
            self.video_controls_frame,
            text="Ground Truth:",
            font=('Arial', 10),
            bg='#16213e',
            fg='#ffffff'
        ).pack(side=tk.LEFT, padx=(20, 5))

        self.gt_entry = tk.Entry(
            self.video_controls_frame,
            font=('Arial', 12, 'bold'),
            width=15,
            bg='#0f3460',
            fg='#4ecca3',
            insertbackground='#ffffff'
        )
        self.gt_entry.pack(side=tk.LEFT, padx=5)
        self.gt_entry.bind('<Return>', lambda e: self._update_ground_truth())

        # Right panel - Info
        info_frame = tk.Frame(main_frame, bg='#0f3460', relief=tk.RAISED, borderwidth=2, width=420)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        info_frame.pack_propagate(False)

        # Model loading section
        load_frame = tk.Frame(info_frame, bg='#16213e', relief=tk.SUNKEN, borderwidth=2)
        load_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            load_frame,
            text="📦 Model",
            font=('Arial', 12, 'bold'),
            bg='#16213e',
            fg='#ffffff'
        ).pack(pady=5)

        self.model_status = tk.Label(
            load_frame,
            text="Not loaded",
            font=('Arial', 10),
            bg='#16213e',
            fg='#ff6b6b'
        )
        self.model_status.pack(pady=2)

        btn_frame = tk.Frame(load_frame, bg='#16213e')
        btn_frame.pack(pady=5)

        self.load_btn = tk.Button(
            btn_frame,
            text="📁 Load Model",
            font=('Arial', 10),
            command=self._load_model_dialog,
            bg='#e94560',
            fg='#ffffff'
        )
        self.load_btn.pack(side=tk.LEFT, padx=5)

        # Mode selection
        mode_frame = tk.Frame(info_frame, bg='#16213e', relief=tk.SUNKEN, borderwidth=2)
        mode_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            mode_frame,
            text="🎮 Mode Selection",
            font=('Arial', 12, 'bold'),
            bg='#16213e',
            fg='#ffffff'
        ).pack(pady=5)

        mode_btn_frame = tk.Frame(mode_frame, bg='#16213e')
        mode_btn_frame.pack(pady=5)

        self.webcam_btn = tk.Button(
            mode_btn_frame,
            text="Webcam",
            font=('Arial', 11),
            command=self._start_webcam,
            bg='#4ecca3',
            fg='#1a1a2e',
            width=12,
            state=tk.DISABLED
        )
        self.webcam_btn.pack(side=tk.LEFT, padx=5)

        self.video_btn = tk.Button(
            mode_btn_frame,
            text="Load Video",
            font=('Arial', 11),
            command=self._load_video,
            bg='#ffd93d',
            fg='#1a1a2e',
            width=12,
            state=tk.DISABLED
        )
        self.video_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(
            mode_btn_frame,
            text="⏹️ Stop",
            font=('Arial', 11),
            command=self._stop,
            bg='#ff6b6b',
            fg='#ffffff',
            width=8,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Second row for analyze button
        mode_btn_frame2 = tk.Frame(mode_frame, bg='#16213e')
        mode_btn_frame2.pack(pady=5)

        self.analyze_btn = tk.Button(
            mode_btn_frame2,
            text="🔬 Analyze Video (Full)",
            font=('Arial', 11),
            command=self._analyze_full_video,
            bg='#9b59b6',
            fg='#ffffff',
            width=25,
            state=tk.DISABLED
        )
        self.analyze_btn.pack(pady=2)

        self.mode_label = tk.Label(
            mode_frame,
            text="Mode: None",
            font=('Arial', 10),
            bg='#16213e',
            fg='#a0a0a0'
        )
        self.mode_label.pack(pady=3)

        # Current prediction (compact)
        pred_frame = tk.Frame(info_frame, bg='#16213e', relief=tk.SUNKEN, borderwidth=2)
        pred_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(
            pred_frame,
            text="Prediction",
            font=('Arial', 10, 'bold'),
            bg='#16213e',
            fg='#ffffff'
        ).pack(pady=2)

        self.current_label = tk.Label(
            pred_frame,
            text="---",
            font=('Arial', 28, 'bold'),
            bg='#16213e',
            fg='#4ecca3'
        )
        self.current_label.pack(pady=2)

        # Ground truth display with score (for video mode)
        self.gt_display_frame = tk.Frame(pred_frame, bg='#16213e')
        self.gt_display_frame.pack(fill=tk.X, pady=3)

        tk.Label(
            self.gt_display_frame,
            text="Expected:",
            font=('Arial', 9),
            bg='#16213e',
            fg='#a0a0a0'
        ).pack(side=tk.LEFT, padx=5)

        self.gt_label = tk.Label(
            self.gt_display_frame,
            text="---",
            font=('Arial', 12, 'bold'),
            bg='#16213e',
            fg='#ffd93d'
        )
        self.gt_label.pack(side=tk.LEFT)

        # GT Score label (shows probability of ground truth class)
        self.gt_score_label = tk.Label(
            self.gt_display_frame,
            text="",
            font=('Arial', 10),
            bg='#16213e',
            fg='#ff6b6b'
        )
        self.gt_score_label.pack(side=tk.LEFT, padx=5)

        self.match_label = tk.Label(
            self.gt_display_frame,
            text="",
            font=('Arial', 12, 'bold'),
            bg='#16213e',
            fg='#4ecca3'
        )
        self.match_label.pack(side=tk.LEFT, padx=5)

        # Handedness (smaller)
        self.hand_label = tk.Label(
            pred_frame,
            text="---",
            font=('Arial', 9),
            bg='#16213e',
            fg='#a0a0a0'
        )
        self.hand_label.pack(pady=2)

        # Confidence bar (compact)
        conf_frame = tk.Frame(info_frame, bg='#16213e', relief=tk.SUNKEN, borderwidth=2)
        conf_frame.pack(fill=tk.X, padx=10, pady=5)

        conf_header = tk.Frame(conf_frame, bg='#16213e')
        conf_header.pack(fill=tk.X, pady=2)

        tk.Label(
            conf_header,
            text="Confidence:",
            font=('Arial', 10),
            bg='#16213e',
            fg='#a0a0a0'
        ).pack(side=tk.LEFT, padx=10)

        self.conf_label = tk.Label(
            conf_header,
            text="0%",
            font=('Arial', 14, 'bold'),
            bg='#16213e',
            fg='#4ecca3'
        )
        self.conf_label.pack(side=tk.LEFT)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Custom.Horizontal.TProgressbar",
                        background='#4ecca3',
                        troughcolor='#0f3460')

        self.conf_bar = ttk.Progressbar(
            conf_frame,
            length=380,
            mode='determinate',
            maximum=100,
            style="Custom.Horizontal.TProgressbar"
        )
        self.conf_bar.pack(pady=5)

        # Top-10 predictions with scrollbar
        top_frame = tk.Frame(info_frame, bg='#16213e', relief=tk.SUNKEN, borderwidth=2)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(
            top_frame,
            text="Top-10 Predictions",
            font=('Arial', 10, 'bold'),
            bg='#16213e',
            fg='#ffffff'
        ).pack(pady=3)

        # Scrollable listbox for predictions
        top_list_frame = tk.Frame(top_frame, bg='#16213e')
        top_list_frame.pack(fill=tk.X, padx=10, pady=3)

        self.top_listbox = tk.Listbox(
            top_list_frame,
            height=6,
            font=('Consolas', 9),
            bg='#0f3460',
            fg='#ffffff',
            selectbackground='#4ecca3',
            selectforeground='#1a1a2e',
            borderwidth=0,
            highlightthickness=0
        )
        self.top_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)

        top_scrollbar = tk.Scrollbar(top_list_frame, orient=tk.VERTICAL)
        top_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.top_listbox.config(yscrollcommand=top_scrollbar.set)
        top_scrollbar.config(command=self.top_listbox.yview)

        # History
        hist_frame = tk.Frame(info_frame, bg='#16213e', relief=tk.SUNKEN, borderwidth=2)
        hist_frame.pack(fill=tk.X, padx=10, pady=10)

        hist_header = tk.Frame(hist_frame, bg='#16213e')
        hist_header.pack(fill=tk.X, pady=5)

        tk.Label(
            hist_header,
            text="History",
            font=('Arial', 12, 'bold'),
            bg='#16213e',
            fg='#ffffff'
        ).pack(side=tk.LEFT, padx=10)

        self.clear_btn = tk.Button(
            hist_header,
            text="🗑️ Clear",
            font=('Arial', 9),
            command=self._clear_history,
            bg='#e94560',
            fg='#ffffff'
        )
        self.clear_btn.pack(side=tk.RIGHT, padx=10)

        self.history_label = tk.Label(
            hist_frame,
            text="---",
            font=('Arial', 11),
            bg='#16213e',
            fg='#4ecca3',
            wraplength=380,
            justify='center'
        )
        self.history_label.pack(pady=10)

        # Stats
        stats_frame = tk.Frame(info_frame, bg='#16213e', relief=tk.SUNKEN, borderwidth=2)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(
            stats_frame,
            text="📈 Statistics",
            font=('Arial', 12, 'bold'),
            bg='#16213e',
            fg='#ffffff'
        ).pack(pady=5)

        self.fps_label = tk.Label(
            stats_frame,
            text="FPS: --",
            font=('Arial', 11),
            bg='#16213e',
            fg='#ffffff'
        )
        self.fps_label.pack(pady=2)

        self.buffer_label = tk.Label(
            stats_frame,
            text=f"Buffer: 0/{self.config.buffer_size}",
            font=('Arial', 11),
            bg='#16213e',
            fg='#ffffff'
        )
        self.buffer_label.pack(pady=2)

        self.device_label = tk.Label(
            stats_frame,
            text=f"Device: {self.config.device.upper()}",
            font=('Arial', 11),
            bg='#16213e',
            fg='#4ecca3' if 'cuda' in self.config.device else '#ff6b6b'
        )
        self.device_label.pack(pady=2)

        # Status bar
        self.status_label = tk.Label(
            self.root,
            text="⚪ Load model to start",
            font=('Arial', 12),
            bg='#1a1a2e',
            fg='#a0a0a0'
        )
        self.status_label.pack(pady=5)

    def _load_model_dialog(self):
        """Open file dialog to load model."""
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Model", "*.pth *.pt"), ("All Files", "*.*")],
            initialdir="."
        )

        if not model_path:
            return

        vocab_path = filedialog.askopenfilename(
            title="Select Vocabulary File",
            filetypes=[("JSON", "*.json"), ("All Files", "*.*")],
            initialdir=str(Path(model_path).parent)
        )

        if not vocab_path:
            return

        success = self.model.load(model_path, vocab_path)

        if success:
            self.model_status.config(
                text=f"✅ Loaded: {self.model.vocab.num_classes} classes",
                fg='#4ecca3'
            )
            self.webcam_btn.config(state=tk.NORMAL)
            self.video_btn.config(state=tk.NORMAL)
            self.analyze_btn.config(state=tk.NORMAL)
            self.status_label.config(text="[OK] Model loaded - Choose mode", fg='#4ecca3')
        else:
            self.model_status.config(text="❌ Load failed", fg='#ff6b6b')
            messagebox.showerror("Error", "Failed to load model.")

    def _start_webcam(self):
        """Start webcam mode."""
        self._stop()

        self.cap = cv2.VideoCapture(self.config.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)

        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera!")
            return

        self.mode = "webcam"
        self.tracker = MediaPipeTracker()
        self.running = True
        self.paused = False

        self.mode_label.config(text="Mode: Webcam", fg='#4ecca3')
        self.stop_btn.config(state=tk.NORMAL)
        self.play_pause_btn.config(state=tk.DISABLED)
        self.restart_btn.config(state=tk.DISABLED)
        self.status_label.config(text="[OK] Webcam Active", fg='#4ecca3')

        self.video_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.video_thread.start()

    def _load_video(self):
        """Load video file for prediction."""
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
                ("MP4", "*.mp4"),
                ("All Files", "*.*")
            ],
            initialdir="."
        )

        if not video_path:
            return

        self._stop()
        self._reset_all_state()  # Reset all predictions and buffers

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open video: {video_path}")
            return

        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.video_total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_current_frame = 0

        # Try to extract ground truth from filename
        filename = Path(video_path).stem.upper()
        self.ground_truth = filename
        self.gt_entry.delete(0, tk.END)
        self.gt_entry.insert(0, filename)
        self.gt_label.config(text=filename)
        self.gt_score_label.config(text="")  # Reset GT score

        self.mode = "video"
        self.tracker = MediaPipeTracker()
        self.running = True
        self.paused = False

        self.mode_label.config(text=f"Mode: Video ({Path(video_path).name})", fg='#ffd93d')
        self.stop_btn.config(state=tk.NORMAL)
        self.play_pause_btn.config(state=tk.NORMAL, text="Pause")
        self.restart_btn.config(state=tk.NORMAL)
        self.status_label.config(text=f"Video: {Path(video_path).name}", fg='#4ecca3')

        self._clear_history()

        self.video_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.video_thread.start()

    def _toggle_pause(self):
        """Toggle pause state."""
        self.paused = not self.paused
        if self.paused:
            self.play_pause_btn.config(text="Play")
            self.status_label.config(text="Paused", fg='#ffd93d')
        else:
            self.play_pause_btn.config(text="Pause")
            self.status_label.config(text="[OK] Playing", fg='#4ecca3')

    def _restart_video(self):
        """Restart video from beginning."""
        if self.cap and self.mode == "video":
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.video_current_frame = 0
            self._clear_history()
            self.paused = False
            self.play_pause_btn.config(text="Pause")

    def _update_speed(self, value):
        """Update playback speed."""
        self.config.video_playback_speed = float(value)
        self.speed_label.config(text=f"{float(value):.1f}x")

    def _update_ground_truth(self):
        """Update ground truth from entry."""
        self.ground_truth = self.gt_entry.get().upper()
        self.gt_label.config(text=self.ground_truth)

    def _analyze_full_video(self):
        """Analyze entire video at once - extract all frames, then predict."""
        video_path = filedialog.askopenfilename(
            title="Select Video File to Analyze",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
                ("MP4", "*.mp4"),
                ("All Files", "*.*")
            ],
            initialdir="."
        )

        if not video_path:
            return

        self._stop()

        # Extract ground truth from filename
        filename = Path(video_path).stem.upper()
        self.ground_truth = filename
        self.gt_entry.delete(0, tk.END)
        self.gt_entry.insert(0, filename)
        self.gt_label.config(text=filename)

        self.status_label.config(text=f"🔬 Analyzing: {Path(video_path).name}...", fg='#9b59b6')
        self.root.update()

        # Run analysis in thread
        def analyze():
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Could not open: {video_path}"))
                    return

                tracker = MediaPipeTracker()
                all_landmarks = []
                frame_count = 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Extract landmarks from ALL frames
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Option to mirror
                    if self.mirror_var.get():
                        frame = cv2.flip(frame, 1)

                    landmarks, annotated, has_hands = tracker.process_frame(frame)

                    if landmarks is not None:
                        all_landmarks.append(landmarks)

                    frame_count += 1

                    # Update progress
                    if frame_count % 5 == 0:
                        self.root.after(0, lambda fc=frame_count, tf=total_frames:
                        self.status_label.config(text=f"🔬 Extracting: {fc}/{tf} frames...", fg='#9b59b6'))

                    # Show last frame
                    if frame_count == total_frames:
                        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        frame_pil = Image.fromarray(frame_rgb)
                        frame_pil = frame_pil.resize((900, 506), Image.Resampling.LANCZOS)
                        frame_tk = ImageTk.PhotoImage(frame_pil)
                        self.root.after(0, lambda ft=frame_tk: self._update_video_display(ft))

                cap.release()
                tracker.close()

                if len(all_landmarks) < 5:
                    self.root.after(0, lambda: self.status_label.config(
                        text=f"❌ Not enough frames with hands detected ({len(all_landmarks)})", fg='#ff6b6b'))
                    return

                # Stack all landmarks and predict
                sequence = np.stack(all_landmarks)
                print(f"Analyzing {len(all_landmarks)} frames from {total_frames} total")

                gloss, conf, top_k, handedness = self.model.predict(sequence)

                # Update display
                self.current_prediction = gloss
                self.current_confidence = conf
                self.current_top_k = top_k
                self.current_handedness = handedness

                # Check if correct
                is_correct = gloss == self.ground_truth
                status_text = f"✅ CORRECT: {gloss}" if is_correct else f"❌ Predicted: {gloss} (Expected: {self.ground_truth})"
                status_color = '#4ecca3' if is_correct else '#ff6b6b'

                self.root.after(0, lambda: self._update_display())
                self.root.after(0, lambda: self.status_label.config(
                    text=f"🔬 {status_text} ({conf * 100:.1f}%) - {len(all_landmarks)} frames", fg=status_color))

                # Print detailed results
                print(f"\n{'=' * 50}")
                print(f"📁 Video: {Path(video_path).name}")
                print(f"🎯 Ground Truth: {self.ground_truth}")
                print(f"🔮 Prediction: {gloss} ({conf * 100:.1f}%)")
                print(f"✋ Handedness: {handedness}")
                print(f"Frames analyzed: {len(all_landmarks)}/{total_frames}")
                print(f"Top-5:")
                for i, (g, p) in enumerate(top_k):
                    marker = "⭐" if g == self.ground_truth else ""
                    print(f"   {i + 1}. {g}: {p * 100:.1f}% {marker}")
                print(f"{'=' * 50}\n")

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self.status_label.config(text=f"❌ Error: {e}", fg='#ff6b6b'))

        threading.Thread(target=analyze, daemon=True).start()

    def _update_video_display(self, frame_tk):
        """Update video display (thread-safe)."""
        self.video_label.configure(image=frame_tk)
        self.video_label.image = frame_tk

    def _stop(self):
        """Stop current mode."""
        self.running = False
        time.sleep(0.3)

        if self.cap:
            self.cap.release()
            self.cap = None

        if self.tracker:
            self.tracker.close()
            self.tracker = None

        self.mode = "none"
        self.paused = False
        self.stop_btn.config(state=tk.DISABLED)
        self.play_pause_btn.config(state=tk.DISABLED)
        self.restart_btn.config(state=tk.DISABLED)
        self.mode_label.config(text="Mode: None", fg='#a0a0a0')
        self.status_label.config(text="⚪ Stopped", fg='#a0a0a0')
        self.video_label.config(image='')

    def _clear_history(self):
        """Clear prediction history."""
        self.gloss_history = []
        self.landmark_buffer.clear()
        self.prediction_history.clear()
        self.stable_prediction = ""
        self.stable_count = 0
        self._update_display()

    def _reset_all_state(self):
        """Reset all state when loading new video."""
        # Clear buffers
        self.landmark_buffer.clear()
        self.prediction_history.clear()
        self.gloss_history = []
        self.stable_prediction = ""
        self.stable_count = 0

        # Reset predictions
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.current_top_k = []
        self.current_handedness = "NONE"

        # Reset frame times
        self.frame_times.clear()
        self.fps = 0.0

        # Reset display
        self.current_label.config(text="---", fg='#4ecca3')
        self.conf_label.config(text="0%")
        self.conf_bar['value'] = 0
        self.hand_label.config(text="---")
        self.gt_score_label.config(text="")
        self.match_label.config(text="")
        self.history_label.config(text="---")
        self.top_listbox.delete(0, tk.END)

    def _process_loop(self):
        """Main processing loop for both webcam and video."""
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            start_time = time.time()

            ret, frame = self.cap.read()

            # Handle video end
            if not ret:
                if self.mode == "video" and self.config.video_loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.video_current_frame = 0
                    continue
                elif self.mode == "video":
                    self.paused = True
                    self.root.after(0, lambda: self.play_pause_btn.config(text="Play"))
                    self.root.after(0, lambda: self.status_label.config(text="⏹️ Video ended", fg='#ffd93d'))
                    continue
                else:
                    continue

            # Update frame counter for video mode
            if self.mode == "video":
                self.video_current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Mirror for webcam only (or if mirror checkbox is checked for video)
            if self.mode == "webcam":
                frame = cv2.flip(frame, 1)
            elif self.mode == "video" and self.mirror_var.get():
                frame = cv2.flip(frame, 1)

            # Process with MediaPipe
            landmarks, annotated, has_hands = self.tracker.process_frame(frame)

            # Add to buffer if hands detected
            if landmarks is not None:
                self.landmark_buffer.append(landmarks)

            # Predict
            if len(self.landmark_buffer) >= self.config.min_frames:
                sequence = np.stack(list(self.landmark_buffer))
                gloss, conf, top_k, handedness = self.model.predict(sequence)

                self.current_prediction = gloss
                self.current_confidence = conf
                self.current_top_k = top_k
                self.current_handedness = handedness

                # Temporal smoothing
                if gloss:
                    self.prediction_history.append(gloss)

                    if gloss == self.stable_prediction:
                        self.stable_count += 1
                    else:
                        self.stable_prediction = gloss
                        self.stable_count = 1

                    if self.stable_count == self.config.stability_threshold:
                        if not self.gloss_history or self.gloss_history[-1] != gloss:
                            self.gloss_history.append(gloss)
                            if len(self.gloss_history) > 15:
                                self.gloss_history = self.gloss_history[-15:]

            # Add overlay
            annotated = self._add_overlay(annotated, has_hands)

            # Convert for Tkinter
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_pil = frame_pil.resize((1024, 576), Image.Resampling.LANCZOS)  # Larger: 16:9
            frame_tk = ImageTk.PhotoImage(frame_pil)

            self.video_label.configure(image=frame_tk)
            self.video_label.image = frame_tk

            # FPS
            elapsed = time.time() - start_time
            self.frame_times.append(elapsed)
            if len(self.frame_times) > 0:
                self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))

            # Update display
            self._update_display()

            # Control playback speed for video mode
            if self.mode == "video":
                target_delay = (1.0 / self.video_fps) / self.config.video_playback_speed
                actual_delay = max(0.001, target_delay - elapsed)
                time.sleep(actual_delay)
            else:
                time.sleep(0.001)

    def _add_overlay(self, frame: np.ndarray, has_hands: bool) -> np.ndarray:
        """Add overlay to frame."""
        h, w = frame.shape[:2]

        # Top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (22, 33, 62), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        mode_text = "Webcam" if self.mode == "webcam" else "Video"
        cv2.putText(frame, f"SignNet - {mode_text}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (233, 69, 96), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w - 120, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (78, 204, 163), 2)

        # Hand indicator with frame count
        hand_color = (78, 204, 163) if has_hands else (107, 107, 255)
        hand_text = f"Hands OK ({len(self.landmark_buffer)})" if has_hands else "No Hands"
        cv2.putText(frame, hand_text, (w - 300, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)

        # Bottom bar with prediction
        if self.current_prediction:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 100), (w, h), (22, 33, 62), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

            # Prediction
            cv2.putText(frame, f"Pred: {self.current_prediction}", (20, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (78, 204, 163), 2)

            # Ground truth (video mode)
            if self.mode == "video" and self.ground_truth:
                gt_color = (78, 204, 163) if self.current_prediction == self.ground_truth else (107, 107, 255)
                cv2.putText(frame, f"GT: {self.ground_truth}", (20, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, gt_color, 2)

            # Confidence
            conf_text = f"{self.current_confidence * 100:.1f}%"
            cv2.putText(frame, conf_text, (w - 150, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (233, 69, 96), 2)

        return frame

    def _update_display(self):
        """Update info panel."""
        # Current prediction
        if self.current_prediction:
            self.current_label.config(text=self.current_prediction)

            # Check match with ground truth and show GT score
            if self.mode == "video" and self.ground_truth:
                # Find GT score in top_k
                gt_score = 0.0
                for gloss, prob in self.current_top_k:
                    if gloss == self.ground_truth:
                        gt_score = prob
                        break

                # Show GT score
                if gt_score > 0:
                    self.gt_score_label.config(text=f"({gt_score * 100:.1f}%)")
                    if gt_score >= 0.5:
                        self.gt_score_label.config(fg='#4ecca3')  # Green
                    elif gt_score >= 0.2:
                        self.gt_score_label.config(fg='#ffd93d')  # Yellow
                    else:
                        self.gt_score_label.config(fg='#ff6b6b')  # Red
                else:
                    self.gt_score_label.config(text="(not in top-10)", fg='#ff6b6b')

                if self.current_prediction == self.ground_truth:
                    self.match_label.config(text="MATCH!", fg='#4ecca3')
                    self.current_label.config(fg='#4ecca3')
                else:
                    self.match_label.config(text="X", fg='#ff6b6b')
                    self.current_label.config(fg='#ff6b6b')
            else:
                self.match_label.config(text="")
                self.gt_score_label.config(text="")

            # Handedness (compact)
            self.hand_label.config(text=self.current_handedness)

            # Confidence
            conf_pct = int(self.current_confidence * 100)
            self.conf_label.config(text=f"{conf_pct}%")
            self.conf_bar['value'] = conf_pct

            if conf_pct >= 70:
                self.conf_label.config(fg='#4ecca3')
            elif conf_pct >= 50:
                self.conf_label.config(fg='#ffd93d')
            else:
                self.conf_label.config(fg='#ff6b6b')
        else:
            self.current_label.config(text="---", fg='#a0a0a0')
            self.match_label.config(text="")
            self.gt_score_label.config(text="")
            self.hand_label.config(text="---")
            self.conf_label.config(text="0%")
            self.conf_bar['value'] = 0

        # Top-10 listbox
        self.top_listbox.delete(0, tk.END)
        for i, (gloss, prob) in enumerate(self.current_top_k):
            marker = " *GT*" if (self.mode == "video" and gloss == self.ground_truth) else ""
            entry = f"{i + 1:2}. {gloss:20} {prob * 100:5.1f}%{marker}"
            self.top_listbox.insert(tk.END, entry)

            # Color coding
            if self.mode == "video" and gloss == self.ground_truth:
                self.top_listbox.itemconfig(i, fg='#ffd93d')  # Gold for GT match
            elif i == 0:
                self.top_listbox.itemconfig(i, fg='#4ecca3')  # Green for top-1
            else:
                self.top_listbox.itemconfig(i, fg='#ffffff')  # White for rest

        # History
        if self.gloss_history:
            self.history_label.config(text=" → ".join(self.gloss_history[-8:]))
        else:
            self.history_label.config(text="---")

        # Stats
        self.fps_label.config(text=f"FPS: {self.fps:.1f}")
        self.buffer_label.config(text=f"Buffer: {len(self.landmark_buffer)}/{self.config.buffer_size}")

        # Video progress
        if self.mode == "video":
            self.progress_label.config(text=f"Frame: {self.video_current_frame} / {self.video_total_frames}")

    def run(self):
        """Run the application."""
        print("\nStarting SignNet GUI with Video Mode...")
        print("   Load model, then choose Webcam or Video mode\n")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind('<q>', lambda e: self._on_close())
        self.root.bind('<Escape>', lambda e: self._on_close())
        self.root.bind('<c>', lambda e: self._clear_history())
        self.root.bind('<space>', lambda e: self._toggle_pause() if self.mode == "video" else None)
        self.root.bind('<r>', lambda e: self._restart_video() if self.mode == "video" else None)

        self.root.mainloop()

    def _on_close(self):
        """Handle window close."""
        self.running = False
        time.sleep(0.3)

        if self.cap:
            self.cap.release()
        if self.tracker:
            self.tracker.close()

        self.root.destroy()


# ============================================================================
# 🚀 MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("SignNet Demo - Webcam & Video Mode")
    print("   Model: TransformerSignClassifierWithHandedness")
    print("   Features: Live webcam + MP4 video prediction")
    print("=" * 60)

    config = DemoConfig()

    try:
        app = SignNetGUI(config)
        app.run()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 Demo closed!")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()