#!/usr/bin/env python3
"""
🎬 SignNet Live Demo GUI
Real-time Sign Language Recognition with MediaPipe + Transformer

Author: Roman Schläpfer, Andrei Chirila
Date: 2025-12-02
Updated for: TransformerSignClassifierWithHandedness (78.5% Accuracy)

Model: Isolated Gloss Classification (145 classes)
Input: MediaPipe Landmarks (1659 features per frame)
Architecture: Transformer Encoder + Multi-Task Learning
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


# ============================================================================
# 🔧 CONFIGURATION
# ============================================================================

@dataclass
class DemoConfig:
    """Configuration for the demo."""
    # Model paths - UPDATE THESE TO YOUR LOCAL PATHS
    model_path: str = "./models_balanced/sign_classifier_best_enhanced.pth"
    vocab_path: str = "./models_balanced/main_vocab.json"

    # Alternative paths to try (will search in order)
    model_search_paths: List[str] = field(default_factory=lambda: [
        "./models_balanced/sign_classifier_best_enhanced.pth",
    ])
    vocab_search_paths: List[str] = field(default_factory=lambda: [
        "./models_balanced/main_vocab.json"
    ])

    # Model architecture (must match training!)
    input_size: int = 1659  # MediaPipe landmarks: hands(126) + face(1434) + pose(99)
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dim_feedforward: int = 2048
    dropout_rate: float = 0.0  # No dropout during inference
    attention_dropout: float = 0.0

    # Inference
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    buffer_size: int = 30  # Frames to accumulate before prediction
    min_frames: int = 10  # Minimum frames for prediction
    prediction_threshold: float = 0.35  # Minimum confidence to show (erhöht)

    # Temporal smoothing
    smoothing_window: int = 5  # Average predictions over this many frames
    stability_threshold: int = 3  # Need this many consecutive same predictions

    # Spam suppression - classes that need higher confidence
    suppress_classes: Dict[str, float] = field(default_factory=lambda: {
        'REGEN': 0.50,  # Oft falsch positiv
        'REGEN-PLUSPLUS': 0.50,
        'KOMMEN': 0.45,
        'HABEN': 0.45,
        'NOCH': 0.50,
        'DANN': 0.45,
    })

    # Display
    window_width: int = 1400
    window_height: int = 850
    camera_width: int = 1280
    camera_height: int = 720
    camera_id: int = 0

    # Filter tokens (internal markers + dropped classes)
    filter_tokens: List[str] = field(default_factory=lambda: [
        '<PAD>', '<BLANK>', '<UNK>', 'UNKNOWN',
        'SUEDRAUM',  # Dropped during training (F1=0)
        'HAUPTSAECHLICH',  # Should be dropped (F1=0)
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
        # Convert string keys to int for idx_to_word
        idx_to_word_raw = data.get('idx_to_word', {})
        vocab.idx_to_word = {int(k): v for k, v in idx_to_word_raw.items()}
        vocab.num_classes = data.get('num_classes', len(vocab.word_to_idx))

        return vocab

    def decode(self, idx: int) -> str:
        """Decode index to gloss. Returns UNKNOWN for missing indices."""
        return self.idx_to_word.get(idx, f"UNKNOWN_{idx}")

    def decode_top_k(self, probs: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Get top-k predictions with probabilities."""
        top_indices = np.argsort(probs)[::-1][:k]
        results = []
        for idx in top_indices:
            gloss = self.decode(idx)
            prob = probs[idx]
            # Skip unknown indices in top-k display
            if not gloss.startswith("UNKNOWN_"):
                results.append((gloss, prob))
        # Ensure we have k results if possible
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

            # Load vocabulary
            self.vocab = Vocabulary.from_json(vocab_path)
            print(f"   Vocabulary size (from JSON): {self.vocab.num_classes}")

            # Load checkpoint FIRST to get actual num_classes
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

            # Handle potential _orig_mod. prefix from torch.compile
            state_dict = checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']

            # Remove _orig_mod. prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("_orig_mod."):
                    new_state_dict[k[10:]] = v
                else:
                    new_state_dict[k] = v

            # Get num_classes from the checkpoint's fc_sign layer
            fc_sign_weight = new_state_dict.get('fc_sign.weight')
            if fc_sign_weight is not None:
                num_classes_from_model = fc_sign_weight.shape[0]
                print(f"   Num classes (from model): {num_classes_from_model}")
            else:
                num_classes_from_model = self.vocab.num_classes
                print(f"   ⚠️ Could not detect num_classes from model, using vocab: {num_classes_from_model}")

            # Build model with correct num_classes from checkpoint
            self.model = TransformerSignClassifierWithHandedness(
                input_size=self.config.input_size,
                hidden_size=self.config.hidden_size,
                num_classes=num_classes_from_model,  # Use model's num_classes!
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                dim_feedforward=self.config.dim_feedforward,
                dropout_rate=0.0,  # No dropout for inference
                attention_dropout=0.0
            ).to(self.device)

            self.model.load_state_dict(new_state_dict)
            self.model.eval()

            # Update vocab num_classes to match model (for display purposes)
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
        """
        Predict gloss from landmarks.

        Args:
            landmarks: Array of shape (T, 1659) - sequence of flattened landmarks

        Returns:
            predicted_gloss: Top-1 prediction
            confidence: Confidence score
            top_k: List of (gloss, probability) tuples
            handedness: Predicted handedness ("LEFT", "RIGHT", "BOTH", "NONE")
        """
        if not self.loaded or len(landmarks) < self.config.min_frames:
            return "", 0.0, [], "NONE"

        with torch.no_grad():
            # Prepare input: (T, 1659) -> (1, T, 1659)
            x = torch.FloatTensor(landmarks).unsqueeze(0).to(self.device)

            # Forward pass
            sign_logits, handedness_logits = self.model(x)

            # Get probabilities
            sign_probs = F.softmax(sign_logits, dim=-1).squeeze(0).cpu().numpy()
            hand_probs = F.softmax(handedness_logits, dim=-1).squeeze(0).cpu().numpy()

            # Top-1 prediction
            top_idx = np.argmax(sign_probs)
            confidence = sign_probs[top_idx]
            predicted_gloss = self.vocab.decode(top_idx)

            # Top-5 predictions
            top_k = self.vocab.decode_top_k(sign_probs, k=5)

            # Handedness
            hand_idx = np.argmax(hand_probs)
            handedness_map = {0: "LEFT", 1: "RIGHT", 2: "BOTH", 3: "NONE"}
            handedness = handedness_map.get(hand_idx, "NONE")

            # ============================================================
            # 🔧 FILTERING & SUPPRESSION
            # ============================================================

            # 1. Filter unknown indices (from model/vocab mismatch)
            if predicted_gloss.startswith("UNKNOWN"):
                return "", 0.0, top_k, handedness

            # 2. Filter internal tokens and dropped classes
            if predicted_gloss in self.config.filter_tokens:
                return "", 0.0, top_k, handedness

            # 3. Check suppressed classes (need higher confidence)
            if predicted_gloss in self.config.suppress_classes:
                min_conf = self.config.suppress_classes[predicted_gloss]
                if confidence < min_conf:
                    return "", confidence, top_k, handedness

            # 4. General threshold
            if confidence < self.config.prediction_threshold:
                return "", confidence, top_k, handedness

            return predicted_gloss, confidence, top_k, handedness


# ============================================================================
# 🎥 MEDIAPIPE TRACKER
# ============================================================================

class MediaPipeTracker:
    """Track face, pose, and hands with MediaPipe Holistic."""

    # Landmark counts
    NUM_HAND_LANDMARKS = 21
    NUM_POSE_LANDMARKS = 33
    NUM_FACE_LANDMARKS = 478

    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize holistic model
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray, bool]:
        """
        Process frame and extract landmarks.

        Returns:
            landmarks: Array of shape (1659,) or None if no hand detection
            annotated_frame: Frame with drawn landmarks
            has_hands: Whether hands were detected
        """
        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Process
        results = self.holistic.process(image_rgb)

        # Convert back to BGR
        image_rgb.flags.writeable = True
        annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        self._draw_landmarks(annotated, results)

        # Extract landmarks (1659 features)
        landmarks, has_hands = self._extract_landmarks(results)

        return landmarks, annotated, has_hands

    def _draw_landmarks(self, frame: np.ndarray, results):
        """Draw all landmarks on frame."""
        # Face mesh
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )

        # Pose
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Hands
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
        """
        Extract landmarks in the same format as training:
        [hands (126), face (1434), pose (99)] = 1659 features

        Each landmark has (x, y, z) coordinates.
        """
        # Initialize with zeros
        landmarks = np.zeros(1659, dtype=np.float32)
        has_hands = False

        # Left hand: 21 landmarks × 3 = 63 features (indices 0-62)
        if results.left_hand_landmarks:
            has_hands = True
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                landmarks[i * 3] = lm.x
                landmarks[i * 3 + 1] = lm.y
                landmarks[i * 3 + 2] = lm.z

        # Right hand: 21 landmarks × 3 = 63 features (indices 63-125)
        if results.right_hand_landmarks:
            has_hands = True
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                landmarks[63 + i * 3] = lm.x
                landmarks[63 + i * 3 + 1] = lm.y
                landmarks[63 + i * 3 + 2] = lm.z

        # Face: 478 landmarks × 3 = 1434 features (indices 126-1559)
        if results.face_landmarks:
            for i, lm in enumerate(results.face_landmarks.landmark):
                landmarks[126 + i * 3] = lm.x
                landmarks[126 + i * 3 + 1] = lm.y
                landmarks[126 + i * 3 + 2] = lm.z

        # Pose: 33 landmarks × 3 = 99 features (indices 1560-1658)
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
# 🖼️ GUI APPLICATION
# ============================================================================

class SignNetGUI:
    """Main GUI application for real-time sign language recognition."""

    def __init__(self, config: DemoConfig):
        self.config = config

        # Initialize window
        self.root = tk.Tk()
        self.root.title("SignNet Live Demo - 78.5% Accuracy Model")
        self.root.geometry(f"{config.window_width}x{config.window_height}")
        self.root.configure(bg='#1a1a2e')

        # Model (will be loaded later)
        self.model = SignNetModel(config)

        # Tracker
        self.tracker = None

        # Video capture
        self.cap = None

        # State
        self.running = False
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
                    text=f"✅ Auto-loaded: {self.model.vocab.num_classes} classes",
                    fg='#4ecca3'
                )
                self.start_btn.config(state=tk.NORMAL)
                self.status_label.config(text="🟢 Model loaded - Click Start Camera", fg='#4ecca3')
            else:
                self.model_status.config(text="❌ Auto-load failed", fg='#ff6b6b')
        else:
            print("⚠️ Model files not found in default paths")
            print(f"   Searched: {self.config.model_search_paths}")
            self.model_status.config(text="⚠️ Click 'Load Model'", fg='#ffd93d')

    def _setup_ui(self):
        """Setup user interface."""

        # Title bar
        title_frame = tk.Frame(self.root, bg='#16213e', height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        title = tk.Label(
            title_frame,
            text="🎬 SignNet Live Recognition",
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
            fg='#ffffff',
            activebackground='#ff6b6b'
        )
        self.load_btn.pack(side=tk.LEFT, padx=5)

        self.start_btn = tk.Button(
            btn_frame,
            text="▶️ Start Camera",
            font=('Arial', 10),
            command=self._toggle_camera,
            bg='#4ecca3',
            fg='#1a1a2e',
            state=tk.DISABLED
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        # Current prediction (large)
        pred_frame = tk.Frame(info_frame, bg='#16213e', relief=tk.SUNKEN, borderwidth=2)
        pred_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            pred_frame,
            text="🎯 Current Sign",
            font=('Arial', 12, 'bold'),
            bg='#16213e',
            fg='#ffffff'
        ).pack(pady=5)

        self.current_label = tk.Label(
            pred_frame,
            text="---",
            font=('Arial', 42, 'bold'),
            bg='#16213e',
            fg='#4ecca3'
        )
        self.current_label.pack(pady=10)

        # Handedness
        self.hand_label = tk.Label(
            pred_frame,
            text="✋ ---",
            font=('Arial', 14),
            bg='#16213e',
            fg='#a0a0a0'
        )
        self.hand_label.pack(pady=5)

        # Confidence bar
        conf_frame = tk.Frame(info_frame, bg='#16213e', relief=tk.SUNKEN, borderwidth=2)
        conf_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            conf_frame,
            text="📊 Confidence",
            font=('Arial', 12, 'bold'),
            bg='#16213e',
            fg='#ffffff'
        ).pack(pady=5)

        self.conf_label = tk.Label(
            conf_frame,
            text="0%",
            font=('Arial', 24, 'bold'),
            bg='#16213e',
            fg='#4ecca3'
        )
        self.conf_label.pack(pady=5)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Custom.Horizontal.TProgressbar",
                        background='#4ecca3',
                        troughcolor='#0f3460',
                        borderwidth=0,
                        lightcolor='#4ecca3',
                        darkcolor='#4ecca3')

        self.conf_bar = ttk.Progressbar(
            conf_frame,
            length=380,
            mode='determinate',
            maximum=100,
            style="Custom.Horizontal.TProgressbar"
        )
        self.conf_bar.pack(pady=10)

        # Top-5 predictions
        top5_frame = tk.Frame(info_frame, bg='#16213e', relief=tk.SUNKEN, borderwidth=2)
        top5_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            top5_frame,
            text="🏆 Top-5 Predictions",
            font=('Arial', 12, 'bold'),
            bg='#16213e',
            fg='#ffffff'
        ).pack(pady=5)

        self.top5_labels = []
        for i in range(5):
            lbl = tk.Label(
                top5_frame,
                text=f"{i + 1}. ---",
                font=('Arial', 10),
                bg='#16213e',
                fg='#a0a0a0',
                anchor='w'
            )
            lbl.pack(fill=tk.X, padx=20, pady=1)
            self.top5_labels.append(lbl)

        # History
        hist_frame = tk.Frame(info_frame, bg='#16213e', relief=tk.SUNKEN, borderwidth=2)
        hist_frame.pack(fill=tk.X, padx=10, pady=10)

        hist_header = tk.Frame(hist_frame, bg='#16213e')
        hist_header.pack(fill=tk.X, pady=5)

        tk.Label(
            hist_header,
            text="📝 History",
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

        # Instructions panel
        inst_frame = tk.Frame(info_frame, bg='#16213e', relief=tk.SUNKEN, borderwidth=2)
        inst_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(
            inst_frame,
            text="📝 Anleitung",
            font=('Arial', 12, 'bold'),
            bg='#16213e',
            fg='#ffffff'
        ).pack(pady=5)

        instructions = [
            "✅ Kamera starten",
            "✅ Gute Beleuchtung sicherstellen",
            "✅ Hände ins Bild halten",
            "✅ Gebärde zeigen & halten",
            "",
            f"🔧 Threshold: {self.config.prediction_threshold:.0%}",
            f"🔧 Buffer: {self.config.buffer_size} Frames",
            "",
            "⌨️ Q = Beenden | C = History löschen"
        ]

        for inst in instructions:
            tk.Label(
                inst_frame,
                text=inst,
                font=('Arial', 9),
                bg='#16213e',
                fg='#a0a0a0' if inst.startswith("🔧") else '#ffffff',
                anchor='w',
                justify='left'
            ).pack(anchor='w', padx=15, pady=1)

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

        # Load model
        success = self.model.load(model_path, vocab_path)

        if success:
            self.model_status.config(
                text=f"✅ Loaded: {self.model.vocab.num_classes} classes",
                fg='#4ecca3'
            )
            self.start_btn.config(state=tk.NORMAL)
            self.status_label.config(text="🟢 Model loaded - Click Start Camera", fg='#4ecca3')
        else:
            self.model_status.config(text="❌ Load failed", fg='#ff6b6b')
            messagebox.showerror("Error", "Failed to load model. Check console for details.")

    def _toggle_camera(self):
        """Start or stop camera."""
        if self.running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        """Start camera and processing."""
        self.cap = cv2.VideoCapture(self.config.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)

        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera!")
            return

        self.tracker = MediaPipeTracker()
        self.running = True

        self.start_btn.config(text="⏹️ Stop Camera", bg='#e94560')
        self.status_label.config(text="🟢 Live Recognition Active", fg='#4ecca3')

        # Start video thread
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()

    def _stop_camera(self):
        """Stop camera."""
        self.running = False
        time.sleep(0.3)

        if self.cap:
            self.cap.release()
            self.cap = None

        if self.tracker:
            self.tracker.close()
            self.tracker = None

        self.start_btn.config(text="▶️ Start Camera", bg='#4ecca3')
        self.status_label.config(text="⚪ Camera stopped", fg='#a0a0a0')

        # Clear video
        self.video_label.config(image='')

    def _clear_history(self):
        """Clear prediction history."""
        self.gloss_history = []
        self.landmark_buffer.clear()
        self.prediction_history.clear()
        self.stable_prediction = ""
        self.stable_count = 0
        self._update_display()

    def _video_loop(self):
        """Main video processing loop."""
        while self.running:
            start_time = time.time()

            ret, frame = self.cap.read()
            if not ret:
                continue

            # Mirror
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

                    # Check for stable prediction
                    if gloss == self.stable_prediction:
                        self.stable_count += 1
                    else:
                        self.stable_prediction = gloss
                        self.stable_count = 1

                    # Add to history if stable enough
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
            frame_pil = frame_pil.resize((900, 506), Image.Resampling.LANCZOS)
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

            time.sleep(0.001)

    def _add_overlay(self, frame: np.ndarray, has_hands: bool) -> np.ndarray:
        """Add overlay to frame."""
        h, w = frame.shape[:2]

        # Top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (22, 33, 62), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        cv2.putText(frame, "SignNet Live (78.5% Acc)", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (233, 69, 96), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w - 120, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (78, 204, 163), 2)

        # Hand indicator
        hand_color = (78, 204, 163) if has_hands else (107, 107, 255)
        hand_text = "✓ Hands" if has_hands else "✗ No Hands"
        cv2.putText(frame, hand_text, (w - 280, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)

        # Bottom bar with prediction
        if self.current_prediction:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 80), (w, h), (22, 33, 62), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

            # Gloss
            cv2.putText(frame, self.current_prediction, (20, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (78, 204, 163), 3)

            # Confidence
            conf_text = f"{self.current_confidence * 100:.1f}%"
            cv2.putText(frame, conf_text, (w - 150, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (233, 69, 96), 2)

        return frame

    def _update_display(self):
        """Update info panel."""
        # Current prediction
        if self.current_prediction:
            self.current_label.config(text=self.current_prediction)

            # Handedness
            hand_emoji = {"LEFT": "🤚", "RIGHT": "✋", "BOTH": "👐", "NONE": "❓"}
            self.hand_label.config(text=f"{hand_emoji.get(self.current_handedness, '❓')} {self.current_handedness}")

            # Confidence
            conf_pct = int(self.current_confidence * 100)
            self.conf_label.config(text=f"{conf_pct}%")
            self.conf_bar['value'] = conf_pct

            # Color by confidence
            if conf_pct >= 70:
                self.current_label.config(fg='#4ecca3')
            elif conf_pct >= 50:
                self.current_label.config(fg='#ffd93d')
            else:
                self.current_label.config(fg='#ff6b6b')
        else:
            self.current_label.config(text="---", fg='#a0a0a0')
            self.hand_label.config(text="✋ ---")
            self.conf_label.config(text="0%")
            self.conf_bar['value'] = 0

        # Top-5
        for i, lbl in enumerate(self.top5_labels):
            if i < len(self.current_top_k):
                gloss, prob = self.current_top_k[i]
                lbl.config(text=f"{i + 1}. {gloss} ({prob * 100:.1f}%)")
                lbl.config(fg='#4ecca3' if i == 0 else '#a0a0a0')
            else:
                lbl.config(text=f"{i + 1}. ---", fg='#606060')

        # History
        if self.gloss_history:
            self.history_label.config(text=" → ".join(self.gloss_history[-8:]))
        else:
            self.history_label.config(text="---")

        # Stats
        self.fps_label.config(text=f"FPS: {self.fps:.1f}")
        self.buffer_label.config(text=f"Buffer: {len(self.landmark_buffer)}/{self.config.buffer_size}")

    def run(self):
        """Run the application."""
        print("\n🎥 Starting SignNet GUI...")
        print("   Load model and click 'Start Camera' to begin\n")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind('<q>', lambda e: self._on_close())
        self.root.bind('<Escape>', lambda e: self._on_close())
        self.root.bind('<c>', lambda e: self._clear_history())

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
    print("🎬 SignNet Live Demo")
    print("   Model: TransformerSignClassifierWithHandedness")
    print("   Accuracy: 78.5% (145 classes)")
    print("   Dataset: RWTH-PHOENIX-Weather 2014")
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