#!/usr/bin/env python3
"""
SignNet Modern GUI - Clean Design with CustomTkinter
Real-time Sign Language Recognition with MediaPipe + Transformer

Author: Roman Schläpfer, Andrei Chirila
Date: 2025-12-04

Features:
- Modern UI with rounded corners and dark theme
- Live webcam recognition
- Video file mode with thumbnail browser
- Side-by-side comparison with ground truth
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
from pathlib import Path
import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import json
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ============================================================================
# 🔧 CONFIGURATION
# ============================================================================

@dataclass
class DemoConfig:
    """Configuration for the demo."""
    # Model paths
    model_path: str = "./models_balanced/sign_classifier_final_enhanced_r.pth"
    vocab_path: str = "./models_balanced/main_vocab.json"

    model_search_paths: List[str] = field(default_factory=lambda: [
        "./models_balanced/sign_classifier_final_enhanced_r.pth",
    ])
    vocab_search_paths: List[str] = field(default_factory=lambda: [
        "./models_balanced/main_vocab.json"
    ])

    # Video folder for thumbnail bar
    video_folder: str = r"D:\OST\SignNet\SignNet+\TransformerEncoder\videos"

    # Thumbnail settings
    thumbnail_width: int = 140
    thumbnail_height: int = 79  # 16:9 ratio
    thumbnail_bar_height: int = 160

    # Model architecture
    input_size: int = 1659
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dim_feedforward: int = 2048
    dropout_rate: float = 0.0
    attention_dropout: float = 0.0

    # Inference
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    buffer_size: int = 60
    min_frames: int = 8
    prediction_threshold: float = 0.30

    # Temporal smoothing
    smoothing_window: int = 3
    stability_threshold: int = 2

    # Spam suppression
    suppress_classes: Dict[str, float] = field(default_factory=lambda: {
        'REGEN': 0.50, 'REGEN-PLUSPLUS': 0.50, 'KOMMEN': 0.45,
        'HABEN': 0.45, 'NOCH': 0.50, 'DANN': 0.45,
    })

    # Display
    window_width: int = 1500
    window_height: int = 900
    camera_width: int = 1280
    camera_height: int = 720
    camera_id: int = 0

    # Video mode settings
    video_playback_speed: float = 0.5
    video_loop: bool = True

    # Filter tokens
    filter_tokens: List[str] = field(default_factory=lambda: [
        '<PAD>', '<BLANK>', '<UNK>', 'UNKNOWN', 'SUEDRAUM', 'HAUPTSAECHLICH',
    ])


# ============================================================================
# 🎨 THEME COLORS
# ============================================================================

class Theme:
    """Modern dark theme colors."""
    BG_DARK = "#1a1b26"
    BG_CARD = "#24283b"
    BG_CARD_HOVER = "#2a2e42"
    BG_INPUT = "#1f2335"

    ACCENT_PRIMARY = "#7aa2f7"
    ACCENT_SUCCESS = "#9ece6a"
    ACCENT_WARNING = "#e0af68"
    ACCENT_DANGER = "#f7768e"
    ACCENT_PURPLE = "#bb9af7"

    TEXT_PRIMARY = "#c0caf5"
    TEXT_SECONDARY = "#565f89"
    TEXT_MUTED = "#414868"

    BORDER = "#3b4261"

    # Button specific
    BTN_WEBCAM = "#3d59a1"
    BTN_VIDEO = "#e0af68"
    BTN_STOP = "#f7768e"
    BTN_ANALYZE = "#bb9af7"


# ============================================================================
# 🧠 MODEL DEFINITION
# ============================================================================

class TransformerSignClassifierWithHandedness(nn.Module):
    """Transformer encoder model with multi-task learning."""

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
            d_model=hidden_size, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=attention_dropout, batch_first=True, activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_sign = nn.Linear(hidden_size, num_classes)
        self.fc_handedness = nn.Linear(hidden_size, 4)

    def forward(self, landmarks, src_key_padding_mask=None):
        B, T, D = landmarks.shape
        x = self.input_proj(landmarks)

        if T > self.pos_embedding.size(1):
            raise ValueError(f"Sequence length {T} exceeds max positional length")
        x = x + self.pos_embedding[:, :T, :]
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).float().unsqueeze(-1)
            x_masked = x * mask
            lengths = mask.sum(dim=1).clamp(min=1.0)
            pooled = x_masked.sum(dim=1) / lengths
        else:
            pooled = x.mean(dim=1)

        pooled = self.dropout(pooled)
        return self.fc_sign(pooled), self.fc_handedness(pooled)


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
        vocab = cls()
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        vocab.word_to_idx = data.get('word_to_idx', {})
        vocab.idx_to_word = {int(k): v for k, v in data.get('idx_to_word', {}).items()}
        vocab.num_classes = data.get('num_classes', len(vocab.word_to_idx))
        return vocab

    def decode(self, idx: int) -> str:
        return self.idx_to_word.get(idx, f"UNKNOWN_{idx}")

    def decode_top_k(self, probs: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        top_indices = np.argsort(probs)[::-1][:k]
        results = [(self.decode(idx), probs[idx]) for idx in top_indices
                   if not self.decode(idx).startswith("UNKNOWN_")]
        return results[:k]


# ============================================================================
# 🧠 MODEL WRAPPER
# ============================================================================

class SignNetModel:
    """Load and run model for inference."""

    def __init__(self, config: DemoConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.vocab = None
        self.loaded = False

    def load(self, model_path: str, vocab_path: str) -> bool:
        try:
            print(f"📦 Loading model from: {model_path}")
            self.vocab = Vocabulary.from_json(vocab_path)

            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint

            # Remove _orig_mod. prefix if present
            new_state_dict = {k[10:] if k.startswith("_orig_mod.") else k: v for k, v in state_dict.items()}

            num_classes = new_state_dict.get('fc_sign.weight', torch.zeros(self.vocab.num_classes, 1)).shape[0]

            self.model = TransformerSignClassifierWithHandedness(
                input_size=self.config.input_size, hidden_size=self.config.hidden_size,
                num_classes=num_classes, num_layers=self.config.num_layers,
                num_heads=self.config.num_heads, dim_feedforward=self.config.dim_feedforward,
            ).to(self.device)

            self.model.load_state_dict(new_state_dict)
            self.model.eval()
            self.vocab.num_classes = num_classes

            print(f"✅ Model loaded! Classes: {num_classes}, Device: {self.device}")
            self.loaded = True
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict(self, landmarks: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]], str]:
        if not self.loaded or len(landmarks) < self.config.min_frames:
            return "", 0.0, [], "NONE"

        with torch.no_grad():
            x = torch.FloatTensor(landmarks).unsqueeze(0).to(self.device)
            sign_logits, hand_logits = self.model(x)

            sign_probs = F.softmax(sign_logits, dim=-1).squeeze(0).cpu().numpy()
            hand_probs = F.softmax(hand_logits, dim=-1).squeeze(0).cpu().numpy()

            top_idx = np.argmax(sign_probs)
            confidence = sign_probs[top_idx]
            predicted_gloss = self.vocab.decode(top_idx)
            top_k = self.vocab.decode_top_k(sign_probs, k=10)

            handedness_map = {0: "LEFT", 1: "RIGHT", 2: "BOTH", 3: "NONE"}
            handedness = handedness_map.get(np.argmax(hand_probs), "NONE")

            # Filtering
            if (predicted_gloss.startswith("UNKNOWN") or
                    predicted_gloss in self.config.filter_tokens or
                    confidence < self.config.prediction_threshold):
                return "", confidence, top_k, handedness

            if predicted_gloss in self.config.suppress_classes:
                if confidence < self.config.suppress_classes[predicted_gloss]:
                    return "", confidence, top_k, handedness

            return predicted_gloss, confidence, top_k, handedness


# ============================================================================
# 🖐️ MEDIAPIPE TRACKER
# ============================================================================

class MediaPipeTracker:
    """Track face, pose, and hands with separate MediaPipe models."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2, model_complexity=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True)
        self.mp_pose = mp.solutions.pose.Pose(static_image_mode=False)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray, bool]:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results_hands = self.mp_hands.process(image_rgb)
        results_face = self.mp_face.process(image_rgb)
        results_pose = self.mp_pose.process(image_rgb)

        annotated = frame.copy()
        self._draw_landmarks(annotated, results_hands, results_face, results_pose)
        landmarks, has_hands = self._extract_landmarks(results_hands, results_face, results_pose)

        return landmarks, annotated, has_hands

    def _draw_landmarks(self, frame, results_hands, results_face, results_pose):
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
        if results_pose.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results_pose.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

    def _extract_landmarks(self, results_hands, results_face, results_pose) -> Tuple[Optional[np.ndarray], bool]:
        landmarks = np.zeros(1659, dtype=np.float32)
        has_hands = False

        if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
            has_hands = True
            for hand_landmarks, handedness in zip(
                    results_hands.multi_hand_landmarks[:2], results_hands.multi_handedness[:2]
            ):
                hand_idx = 0 if handedness.classification[0].label == "Left" else 1
                for j, lm in enumerate(hand_landmarks.landmark):
                    base = hand_idx * 63 + j * 3
                    landmarks[base:base + 3] = [lm.x, lm.y, lm.z]

        if results_face.multi_face_landmarks:
            for j, lm in enumerate(results_face.multi_face_landmarks[0].landmark):
                base = 126 + j * 3
                landmarks[base:base + 3] = [lm.x, lm.y, lm.z]

        if results_pose.pose_landmarks:
            for j, lm in enumerate(results_pose.pose_landmarks.landmark):
                base = 1560 + j * 3
                landmarks[base:base + 3] = [lm.x, lm.y, lm.z]

        return landmarks if has_hands else None, has_hands

    def close(self):
        self.mp_hands.close()
        self.mp_face.close()
        self.mp_pose.close()


# ============================================================================
# 🖼️ MODERN GUI APPLICATION
# ============================================================================

class SignNetModernGUI:
    """Modern GUI application with CustomTkinter."""

    def __init__(self, config: DemoConfig):
        self.config = config

        # Initialize window
        self.root = ctk.CTk()
        self.root.title("SignNet - Webcam & Video Mode")
        self.root.geometry(f"{config.window_width}x{config.window_height}")
        self.root.configure(fg_color=Theme.BG_DARK)

        # Model & Tracker
        self.model = SignNetModel(config)
        self.tracker = None

        # Video/Camera state
        self.cap = None
        self.video_path = None
        self.video_fps = 25.0
        self.video_total_frames = 0
        self.video_current_frame = 0
        self.mode = "none"  # "webcam", "video", "none"

        # Processing state
        self.running = False
        self.paused = False
        self.landmark_buffer = deque(maxlen=config.buffer_size)
        self.prediction_history = deque(maxlen=config.smoothing_window)
        self.stable_prediction = ""
        self.stable_count = 0

        # Current predictions
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.current_top_k = []
        self.current_handedness = "NONE"
        self.ground_truth = ""
        self.gloss_history: List[str] = []

        # FPS tracking
        self.fps = 0.0
        self.frame_times = deque(maxlen=30)

        # Thumbnail storage
        self.thumbnails: Dict[str, ctk.CTkImage] = {}
        self.video_files: List[Path] = []
        self.thumbnail_buttons: List[ctk.CTkButton] = []
        self.selected_thumbnail_idx = -1

        # Build UI
        self._setup_ui()

        # Auto-load model
        self.root.after(100, self._try_auto_load)
        self.root.after(200, self._load_thumbnails)

    def _setup_ui(self):
        """Setup the modern UI."""
        # Configure grid
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=0)  # Title
        self.root.grid_rowconfigure(1, weight=1)  # Main content
        self.root.grid_rowconfigure(2, weight=0)  # Thumbnail bar

        # Title bar
        self._setup_title_bar()

        # Left panel (Video + Controls)
        self._setup_left_panel()

        # Right panel (Info)
        self._setup_right_panel()

        # Bottom thumbnail bar
        self._setup_thumbnail_bar()

    def _setup_title_bar(self):
        """Setup title bar."""
        title_frame = ctk.CTkFrame(self.root, fg_color=Theme.BG_CARD, corner_radius=0, height=50)
        title_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=0, pady=0)
        title_frame.grid_propagate(False)

        title_label = ctk.CTkLabel(
            title_frame, text="SignNet - Webcam & Video Mode",
            font=ctk.CTkFont(size=20, weight="bold"), text_color=Theme.TEXT_PRIMARY
        )
        title_label.pack(pady=12)

    def _setup_left_panel(self):
        """Setup left panel with video display and controls."""
        left_frame = ctk.CTkFrame(self.root, fg_color=Theme.BG_CARD, corner_radius=15)
        left_frame.grid(row=1, column=0, sticky="nsew", padx=(15, 8), pady=15)
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_rowconfigure(0, weight=1)

        # Video display frame
        video_container = ctk.CTkFrame(left_frame, fg_color=Theme.BG_INPUT, corner_radius=12)
        video_container.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        video_container.grid_columnconfigure(0, weight=1)
        video_container.grid_rowconfigure(0, weight=1)

        # Video label
        self.video_label = ctk.CTkLabel(video_container, text="", fg_color=Theme.BG_INPUT)
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Overlay info (top of video)
        self.overlay_frame = ctk.CTkFrame(video_container, fg_color="transparent")
        self.overlay_frame.place(relx=0, rely=0, relwidth=1, anchor="nw")

        self.overlay_label = ctk.CTkLabel(
            self.overlay_frame, text="SignNet — Ready",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=Theme.TEXT_PRIMARY
        )
        self.overlay_label.pack(side="left", padx=15, pady=8)

        self.fps_overlay_label = ctk.CTkLabel(
            self.overlay_frame, text="FPS: --",
            font=ctk.CTkFont(size=12), text_color=Theme.TEXT_SECONDARY
        )
        self.fps_overlay_label.pack(side="right", padx=15, pady=8)

        self.hands_overlay_label = ctk.CTkLabel(
            self.overlay_frame, text="No Hands",
            font=ctk.CTkFont(size=12), text_color=Theme.TEXT_SECONDARY
        )
        self.hands_overlay_label.pack(side="right", padx=10, pady=8)

        # Video controls
        controls_frame = ctk.CTkFrame(left_frame, fg_color=Theme.BG_INPUT, corner_radius=10, height=50)
        controls_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 15))
        controls_frame.grid_propagate(False)

        # Pause/Play button
        self.play_pause_btn = ctk.CTkButton(
            controls_frame, text="⏸  Pause", width=90, height=32,
            fg_color=Theme.BTN_STOP, hover_color="#ff8fa3",
            command=self._toggle_pause, state="disabled"
        )
        self.play_pause_btn.pack(side="left", padx=10, pady=9)

        # Restart button
        self.restart_btn = ctk.CTkButton(
            controls_frame, text="↻  Restart", width=90, height=32,
            fg_color=Theme.ACCENT_SUCCESS, hover_color="#b5e48c",
            text_color=Theme.BG_DARK, command=self._restart_video, state="disabled"
        )
        self.restart_btn.pack(side="left", padx=5, pady=9)

        # Frame counter
        self.frame_label = ctk.CTkLabel(
            controls_frame, text="Frame: 0 / 0",
            font=ctk.CTkFont(size=11), text_color=Theme.TEXT_SECONDARY
        )
        self.frame_label.pack(side="left", padx=15, pady=9)

        # Speed control
        speed_label = ctk.CTkLabel(controls_frame, text="Speed:", font=ctk.CTkFont(size=11),
                                   text_color=Theme.TEXT_SECONDARY)
        speed_label.pack(side="left", padx=(10, 5), pady=9)

        self.speed_slider = ctk.CTkSlider(
            controls_frame, from_=0.1, to=2.0, number_of_steps=19,
            width=100, height=16, command=self._update_speed
        )
        self.speed_slider.set(0.5)
        self.speed_slider.pack(side="left", padx=5, pady=9)

        self.speed_value_label = ctk.CTkLabel(
            controls_frame, text="0.5x", font=ctk.CTkFont(size=11), text_color=Theme.ACCENT_SUCCESS
        )
        self.speed_value_label.pack(side="left", padx=5, pady=9)

        # Mirror checkbox
        self.mirror_var = ctk.BooleanVar(value=False)
        self.mirror_check = ctk.CTkCheckBox(
            controls_frame, text="Mirror", variable=self.mirror_var,
            font=ctk.CTkFont(size=11), width=70, height=24
        )
        self.mirror_check.pack(side="left", padx=15, pady=9)

        # Ground truth
        gt_label = ctk.CTkLabel(controls_frame, text="Ground Truth:", font=ctk.CTkFont(size=11),
                                text_color=Theme.TEXT_SECONDARY)
        gt_label.pack(side="left", padx=(10, 5), pady=9)

        self.gt_entry = ctk.CTkEntry(
            controls_frame, width=100, height=28, fg_color=Theme.BG_DARK,
            border_color=Theme.BORDER, text_color=Theme.ACCENT_SUCCESS
        )
        self.gt_entry.pack(side="left", padx=5, pady=9)
        self.gt_entry.bind('<Return>', lambda e: self._update_ground_truth())

    def _setup_right_panel(self):
        """Setup right panel with model info, controls, predictions."""
        right_frame = ctk.CTkFrame(self.root, fg_color=Theme.BG_CARD, corner_radius=15)
        right_frame.grid(row=1, column=1, sticky="nsew", padx=(8, 15), pady=15)
        right_frame.grid_columnconfigure(0, weight=1)

        # Model section
        self._create_section_header(right_frame, "📦  Model", row=0)
        model_frame = ctk.CTkFrame(right_frame, fg_color=Theme.BG_INPUT, corner_radius=10)
        model_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 15))

        self.model_status_label = ctk.CTkLabel(
            model_frame, text="⏳ Loading...",
            font=ctk.CTkFont(size=12), text_color=Theme.ACCENT_WARNING
        )
        self.model_status_label.pack(pady=8)

        self.load_model_btn = ctk.CTkButton(
            model_frame, text="📁  Load Model", width=200, height=36,
            fg_color=Theme.BTN_STOP, hover_color="#ff8fa3",
            command=self._load_model_dialog
        )
        self.load_model_btn.pack(pady=(0, 12))

        # Mode Selection section
        self._create_section_header(right_frame, "🎮  Mode Selection", row=2)
        mode_frame = ctk.CTkFrame(right_frame, fg_color=Theme.BG_INPUT, corner_radius=10)
        mode_frame.grid(row=3, column=0, sticky="ew", padx=15, pady=(0, 15))

        btn_row = ctk.CTkFrame(mode_frame, fg_color="transparent")
        btn_row.pack(pady=10)

        self.webcam_btn = ctk.CTkButton(
            btn_row, text="📹  Webcam", width=100, height=36,
            fg_color=Theme.BTN_WEBCAM, hover_color="#4c6eb5",
            command=self._start_webcam, state="disabled"
        )
        self.webcam_btn.pack(side="left", padx=5)

        self.video_btn = ctk.CTkButton(
            btn_row, text="🎬  Load Video", width=110, height=36,
            fg_color=Theme.BTN_VIDEO, hover_color="#ebc577",
            text_color=Theme.BG_DARK, command=self._load_video, state="disabled"
        )
        self.video_btn.pack(side="left", padx=5)

        self.stop_btn = ctk.CTkButton(
            btn_row, text="⏹  Stop", width=80, height=36,
            fg_color=Theme.BTN_STOP, hover_color="#ff8fa3",
            command=self._stop, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=5)

        self.analyze_btn = ctk.CTkButton(
            mode_frame, text="🔬  Analyze Video (Full)", width=280, height=36,
            fg_color=Theme.BTN_ANALYZE, hover_color="#cba6ff",
            command=self._analyze_full_video, state="disabled"
        )
        self.analyze_btn.pack(pady=(0, 10))

        self.mode_status_label = ctk.CTkLabel(
            mode_frame, text="Mode: None",
            font=ctk.CTkFont(size=11), text_color=Theme.TEXT_SECONDARY
        )
        self.mode_status_label.pack(pady=(0, 8))

        # Prediction section
        self._create_section_header(right_frame, "🎯  Prediction", row=4)
        pred_frame = ctk.CTkFrame(right_frame, fg_color=Theme.BG_INPUT, corner_radius=10)
        pred_frame.grid(row=5, column=0, sticky="ew", padx=15, pady=(0, 15))

        self.prediction_label = ctk.CTkLabel(
            pred_frame, text="---",
            font=ctk.CTkFont(size=28, weight="bold"), text_color=Theme.ACCENT_SUCCESS
        )
        self.prediction_label.pack(pady=(15, 5))

        # Expected + Match
        expected_row = ctk.CTkFrame(pred_frame, fg_color="transparent")
        expected_row.pack(pady=5)

        ctk.CTkLabel(expected_row, text="Expected:", font=ctk.CTkFont(size=11), text_color=Theme.TEXT_SECONDARY).pack(
            side="left")
        self.expected_label = ctk.CTkLabel(expected_row, text="---", font=ctk.CTkFont(size=12, weight="bold"),
                                           text_color=Theme.ACCENT_WARNING)
        self.expected_label.pack(side="left", padx=5)
        self.gt_score_label = ctk.CTkLabel(expected_row, text="", font=ctk.CTkFont(size=11),
                                           text_color=Theme.TEXT_SECONDARY)
        self.gt_score_label.pack(side="left", padx=5)
        self.match_label = ctk.CTkLabel(expected_row, text="", font=ctk.CTkFont(size=12, weight="bold"),
                                        text_color=Theme.ACCENT_SUCCESS)
        self.match_label.pack(side="left", padx=5)

        # Confidence
        conf_row = ctk.CTkFrame(pred_frame, fg_color="transparent")
        conf_row.pack(fill="x", padx=15, pady=10)

        ctk.CTkLabel(conf_row, text="Confidence:", font=ctk.CTkFont(size=11), text_color=Theme.TEXT_SECONDARY).pack(
            side="left")
        self.conf_value_label = ctk.CTkLabel(conf_row, text="0%", font=ctk.CTkFont(size=14, weight="bold"),
                                             text_color=Theme.ACCENT_SUCCESS)
        self.conf_value_label.pack(side="left", padx=10)

        self.conf_progress = ctk.CTkProgressBar(pred_frame, width=250, height=12, fg_color=Theme.BG_DARK,
                                                progress_color=Theme.ACCENT_SUCCESS)
        self.conf_progress.set(0)
        self.conf_progress.pack(pady=(0, 15))

        # Top-10 Predictions
        self._create_section_header(right_frame, "📊  Top-10 Predictions", row=6)
        top_frame = ctk.CTkFrame(right_frame, fg_color=Theme.BG_INPUT, corner_radius=10)
        top_frame.grid(row=7, column=0, sticky="ew", padx=15, pady=(0, 15))

        # Use a text widget for predictions (easier to style)
        self.predictions_text = ctk.CTkTextbox(
            top_frame, width=280, height=130, fg_color=Theme.BG_DARK,
            font=ctk.CTkFont(family="Consolas", size=11), text_color=Theme.TEXT_PRIMARY
        )
        self.predictions_text.pack(padx=10, pady=10)
        self.predictions_text.configure(state="disabled")

        # History section
        self._create_section_header(right_frame, "📜  History", row=8)
        hist_frame = ctk.CTkFrame(right_frame, fg_color=Theme.BG_INPUT, corner_radius=10)
        hist_frame.grid(row=9, column=0, sticky="ew", padx=15, pady=(0, 15))

        hist_header = ctk.CTkFrame(hist_frame, fg_color="transparent")
        hist_header.pack(fill="x", padx=10, pady=5)

        self.history_label = ctk.CTkLabel(
            hist_frame, text="---", font=ctk.CTkFont(size=11),
            text_color=Theme.ACCENT_SUCCESS, wraplength=260
        )
        self.history_label.pack(pady=(0, 10))

        self.clear_history_btn = ctk.CTkButton(
            hist_header, text="🗑️ Clear", width=60, height=24,
            fg_color=Theme.BTN_STOP, hover_color="#ff8fa3",
            font=ctk.CTkFont(size=10), command=self._clear_history
        )
        self.clear_history_btn.pack(side="right")

        # Statistics section
        self._create_section_header(right_frame, "📈  Statistics", row=10)
        stats_frame = ctk.CTkFrame(right_frame, fg_color=Theme.BG_INPUT, corner_radius=10)
        stats_frame.grid(row=11, column=0, sticky="ew", padx=15, pady=(0, 15))

        stats_grid = ctk.CTkFrame(stats_frame, fg_color="transparent")
        stats_grid.pack(pady=10)

        self.stats_fps_label = ctk.CTkLabel(stats_grid, text="FPS: --", font=ctk.CTkFont(size=11),
                                            text_color=Theme.TEXT_PRIMARY)
        self.stats_fps_label.grid(row=0, column=0, padx=20, pady=2)

        self.stats_buffer_label = ctk.CTkLabel(stats_grid, text="Buffer: 0/60", font=ctk.CTkFont(size=11),
                                               text_color=Theme.TEXT_PRIMARY)
        self.stats_buffer_label.grid(row=0, column=1, padx=20, pady=2)

        device_text = f"Device: {self.config.device.upper()}"
        device_color = Theme.ACCENT_SUCCESS if "cuda" in self.config.device else Theme.ACCENT_DANGER
        self.stats_device_label = ctk.CTkLabel(stats_grid, text=device_text, font=ctk.CTkFont(size=11),
                                               text_color=device_color)
        self.stats_device_label.grid(row=1, column=0, columnspan=2, pady=2)

    def _create_section_header(self, parent, text: str, row: int):
        """Create a section header label."""
        label = ctk.CTkLabel(
            parent, text=text, font=ctk.CTkFont(size=13, weight="bold"),
            text_color=Theme.TEXT_PRIMARY, anchor="w"
        )
        label.grid(row=row, column=0, sticky="w", padx=20, pady=(10, 5))

    def _setup_thumbnail_bar(self):
        """Setup the horizontal scrollable thumbnail bar."""
        thumb_container = ctk.CTkFrame(self.root, fg_color=Theme.BG_CARD, corner_radius=15,
                                       height=self.config.thumbnail_bar_height)
        thumb_container.grid(row=2, column=0, columnspan=2, sticky="ew", padx=15, pady=(0, 15))
        thumb_container.grid_propagate(False)
        thumb_container.grid_columnconfigure(0, weight=1)

        # Header
        header_frame = ctk.CTkFrame(thumb_container, fg_color="transparent", height=30)
        header_frame.pack(fill="x", padx=15, pady=(8, 0))
        header_frame.pack_propagate(False)

        ctk.CTkLabel(
            header_frame, text="▶  Videos", font=ctk.CTkFont(size=13, weight="bold"),
            text_color=Theme.TEXT_PRIMARY
        ).pack(side="left")

        self.thumb_count_label = ctk.CTkLabel(
            header_frame, text="Loading...", font=ctk.CTkFont(size=11),
            text_color=Theme.TEXT_SECONDARY
        )
        self.thumb_count_label.pack(side="left", padx=10)

        # Refresh button
        refresh_btn = ctk.CTkButton(
            header_frame, text="↻", width=30, height=24,
            fg_color=Theme.BG_INPUT, hover_color=Theme.BG_CARD_HOVER,
            command=self._load_thumbnails
        )
        refresh_btn.pack(side="right")

        # Scrollable frame for thumbnails
        self.thumb_scroll_frame = ctk.CTkScrollableFrame(
            thumb_container, fg_color=Theme.BG_INPUT, corner_radius=10,
            orientation="horizontal", height=self.config.thumbnail_height + 40
        )
        self.thumb_scroll_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))

    def _try_auto_load(self):
        """Try to auto-load model from default paths."""
        model_path = next((p for p in self.config.model_search_paths if Path(p).exists()), None)
        vocab_path = next((p for p in self.config.vocab_search_paths if Path(p).exists()), None)

        if model_path and vocab_path:
            success = self.model.load(model_path, vocab_path)
            if success:
                self.model_status_label.configure(
                    text=f"✅ Loaded: {self.model.vocab.num_classes} classes",
                    text_color=Theme.ACCENT_SUCCESS
                )
                self._enable_controls()
            else:
                self.model_status_label.configure(text="❌ Load failed", text_color=Theme.ACCENT_DANGER)
        else:
            self.model_status_label.configure(text="⚠️ Click 'Load Model'", text_color=Theme.ACCENT_WARNING)

    def _enable_controls(self):
        """Enable mode selection buttons after model is loaded."""
        self.webcam_btn.configure(state="normal")
        self.video_btn.configure(state="normal")
        self.analyze_btn.configure(state="normal")

    def _load_thumbnails(self):
        """Load video thumbnails from configured folder."""
        # Clear existing
        for widget in self.thumb_scroll_frame.winfo_children():
            widget.destroy()
        self.thumbnails.clear()
        self.video_files.clear()
        self.thumbnail_buttons.clear()
        self.selected_thumbnail_idx = -1

        video_folder = Path(self.config.video_folder)
        if not video_folder.exists():
            self.thumb_count_label.configure(text="Folder not found", text_color=Theme.ACCENT_DANGER)
            return

        # Find videos
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']:
            self.video_files.extend(video_folder.glob(f'*{ext}'))
        self.video_files.sort(key=lambda x: x.stem.lower())

        if not self.video_files:
            self.thumb_count_label.configure(text="No videos found", text_color=Theme.ACCENT_WARNING)
            return

        self.thumb_count_label.configure(text=f"{len(self.video_files)} videos", text_color=Theme.ACCENT_SUCCESS)

        # Generate thumbnails in background
        threading.Thread(target=self._generate_thumbnails, daemon=True).start()

    def _generate_thumbnails(self):
        """Generate thumbnails for all videos."""
        for i, video_path in enumerate(self.video_files):
            try:
                cap = cv2.VideoCapture(str(video_path))
                ret, frame = cap.read()
                cap.release()

                if not ret:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                pil_image = pil_image.resize(
                    (self.config.thumbnail_width, self.config.thumbnail_height),
                    Image.Resampling.LANCZOS
                )

                self.root.after(0, lambda p=video_path, img=pil_image, idx=i: self._add_thumbnail(p, img, idx))

            except Exception as e:
                print(f"⚠️ Error creating thumbnail for {video_path.name}: {e}")

    def _add_thumbnail(self, video_path: Path, pil_image: Image.Image, index: int):
        """Add a thumbnail to the bar."""
        # Create CTkImage
        ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image,
                                 size=(self.config.thumbnail_width, self.config.thumbnail_height))
        self.thumbnails[str(video_path)] = ctk_image

        # Container frame
        thumb_frame = ctk.CTkFrame(self.thumb_scroll_frame, fg_color="transparent")
        thumb_frame.pack(side="left", padx=5, pady=5)

        # Thumbnail button
        btn = ctk.CTkButton(
            thumb_frame, image=ctk_image, text="",
            width=self.config.thumbnail_width + 4,
            height=self.config.thumbnail_height + 4,
            fg_color=Theme.BG_DARK, hover_color=Theme.ACCENT_PRIMARY,
            border_width=2, border_color=Theme.BORDER,
            command=lambda p=video_path, i=index: self._on_thumbnail_click(p, i)
        )
        btn.pack()
        self.thumbnail_buttons.append(btn)

        # Label
        name = video_path.stem
        if len(name) > 14:
            name = name[:12] + ".."

        lbl = ctk.CTkLabel(
            thumb_frame, text=name, font=ctk.CTkFont(size=10),
            text_color=Theme.TEXT_SECONDARY
        )
        lbl.pack(pady=(3, 0))

    def _on_thumbnail_click(self, video_path: Path, index: int):
        """Handle thumbnail click."""
        # Update selection highlighting
        if self.selected_thumbnail_idx >= 0 and self.selected_thumbnail_idx < len(self.thumbnail_buttons):
            self.thumbnail_buttons[self.selected_thumbnail_idx].configure(border_color=Theme.BORDER)

        if index < len(self.thumbnail_buttons):
            self.thumbnail_buttons[index].configure(border_color=Theme.ACCENT_PRIMARY)
        self.selected_thumbnail_idx = index

        # Load video
        self._load_video_from_path(str(video_path))

    def _load_model_dialog(self):
        """Open dialog to load model."""
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Model", "*.pth *.pt"), ("All Files", "*.*")]
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

        if self.model.load(model_path, vocab_path):
            self.model_status_label.configure(
                text=f"✅ Loaded: {self.model.vocab.num_classes} classes",
                text_color=Theme.ACCENT_SUCCESS
            )
            self._enable_controls()
        else:
            self.model_status_label.configure(text="❌ Load failed", text_color=Theme.ACCENT_DANGER)
            messagebox.showerror("Error", "Failed to load model.")

    def _start_webcam(self):
        """Start webcam mode."""
        self._stop_internal()

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

        self._update_mode_ui()
        threading.Thread(target=self._process_loop, daemon=True).start()

    def _load_video(self):
        """Open dialog to load video."""
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")],
            initialdir=self.config.video_folder
        )
        if video_path:
            self._load_video_from_path(video_path)

    def _load_video_from_path(self, video_path: str):
        """Load video from path (non-blocking)."""
        self.running = False
        self.root.after(100, lambda: self._do_load_video(video_path))

    def _do_load_video(self, video_path: str):
        """Actually load the video."""
        if self.cap:
            self.cap.release()
        if self.tracker:
            self.tracker.close()

        self._reset_state()

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open: {video_path}")
            return

        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.video_total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_current_frame = 0

        # Ground truth from filename
        self.ground_truth = Path(video_path).stem.upper()
        self.gt_entry.delete(0, "end")
        self.gt_entry.insert(0, self.ground_truth)
        self.expected_label.configure(text=self.ground_truth)

        self.mode = "video"
        self.tracker = MediaPipeTracker()
        self.running = True
        self.paused = False

        self._update_mode_ui()
        self._clear_history()
        threading.Thread(target=self._process_loop, daemon=True).start()

    def _stop(self):
        """Stop button handler."""
        self._stop_internal()
        self._update_mode_ui()

    def _stop_internal(self):
        """Internal stop without UI update."""
        self.running = False
        time.sleep(0.2)

        if self.cap:
            self.cap.release()
            self.cap = None
        if self.tracker:
            self.tracker.close()
            self.tracker = None

        self.mode = "none"
        self.paused = False

    def _toggle_pause(self):
        """Toggle pause state."""
        self.paused = not self.paused
        self.play_pause_btn.configure(text="▶  Play" if self.paused else "⏸  Pause")

    def _restart_video(self):
        """Restart video from beginning."""
        if self.cap and self.mode == "video":
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.video_current_frame = 0
            self._clear_history()
            self.paused = False
            self.play_pause_btn.configure(text="⏸  Pause")

    def _update_speed(self, value):
        """Update playback speed."""
        self.config.video_playback_speed = float(value)
        self.speed_value_label.configure(text=f"{value:.1f}x")

    def _update_ground_truth(self):
        """Update ground truth from entry."""
        self.ground_truth = self.gt_entry.get().upper()
        self.expected_label.configure(text=self.ground_truth)

    def _update_mode_ui(self):
        """Update UI based on current mode."""
        if self.mode == "webcam":
            self.mode_status_label.configure(text="Mode: Webcam", text_color=Theme.ACCENT_SUCCESS)
            self.overlay_label.configure(text="SignNet — Webcam")
            self.stop_btn.configure(state="normal")
            self.play_pause_btn.configure(state="disabled")
            self.restart_btn.configure(state="disabled")
        elif self.mode == "video":
            name = Path(self.video_path).name if self.video_path else "Video"
            self.mode_status_label.configure(text=f"Mode: {name}", text_color=Theme.ACCENT_WARNING)
            self.overlay_label.configure(text=f"SignNet — {name}")
            self.stop_btn.configure(state="normal")
            self.play_pause_btn.configure(state="normal", text="⏸  Pause")
            self.restart_btn.configure(state="normal")
        else:
            self.mode_status_label.configure(text="Mode: None", text_color=Theme.TEXT_SECONDARY)
            self.overlay_label.configure(text="SignNet — Ready")
            self.stop_btn.configure(state="disabled")
            self.play_pause_btn.configure(state="disabled")
            self.restart_btn.configure(state="disabled")
            self.video_label.configure(image=None, text="")

    def _reset_state(self):
        """Reset all processing state."""
        self.landmark_buffer.clear()
        self.prediction_history.clear()
        self.gloss_history = []
        self.stable_prediction = ""
        self.stable_count = 0
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.current_top_k = []
        self.current_handedness = "NONE"
        self.frame_times.clear()
        self.fps = 0.0

    def _clear_history(self):
        """Clear prediction history."""
        self.gloss_history = []
        self.landmark_buffer.clear()
        self.prediction_history.clear()
        self.stable_prediction = ""
        self.stable_count = 0
        self.root.after(0, self._update_display)

    def _analyze_full_video(self):
        """Analyze entire video at once."""
        video_path = filedialog.askopenfilename(
            title="Select Video to Analyze",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")],
            initialdir=self.config.video_folder
        )
        if not video_path:
            return

        self._stop_internal()
        self.ground_truth = Path(video_path).stem.upper()
        self.gt_entry.delete(0, "end")
        self.gt_entry.insert(0, self.ground_truth)
        self.expected_label.configure(text=self.ground_truth)

        self.mode_status_label.configure(text=f"🔬 Analyzing...", text_color=Theme.ACCENT_PURPLE)

        def analyze():
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    return

                tracker = MediaPipeTracker()
                all_landmarks = []
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if self.mirror_var.get():
                        frame = cv2.flip(frame, 1)

                    landmarks, annotated, _ = tracker.process_frame(frame)
                    if landmarks is not None:
                        all_landmarks.append(landmarks)

                cap.release()
                tracker.close()

                if len(all_landmarks) < 5:
                    self.root.after(0, lambda: self.mode_status_label.configure(
                        text="❌ Not enough hands", text_color=Theme.ACCENT_DANGER))
                    return

                sequence = np.stack(all_landmarks)
                gloss, conf, top_k, handedness = self.model.predict(sequence)

                self.current_prediction = gloss
                self.current_confidence = conf
                self.current_top_k = top_k
                self.current_handedness = handedness

                is_correct = gloss == self.ground_truth
                status = f"✅ {gloss}" if is_correct else f"❌ {gloss} (exp: {self.ground_truth})"
                color = Theme.ACCENT_SUCCESS if is_correct else Theme.ACCENT_DANGER

                self.root.after(0, lambda: self.mode_status_label.configure(text=status, text_color=color))
                self.root.after(0, self._update_display)

            except Exception as e:
                self.root.after(0, lambda: self.mode_status_label.configure(
                    text=f"❌ Error", text_color=Theme.ACCENT_DANGER))

        threading.Thread(target=analyze, daemon=True).start()

    def _process_loop(self):
        """Main processing loop."""
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            start_time = time.time()
            ret, frame = self.cap.read()

            if not ret:
                if self.mode == "video":
                    if self.config.video_loop:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.video_current_frame = 0
                        continue
                    else:
                        self.paused = True
                        self.root.after(0, lambda: self.play_pause_btn.configure(text="▶  Play"))
                        continue
                continue

            if self.mode == "video":
                self.video_current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Mirror
            if self.mode == "webcam" or (self.mode == "video" and self.mirror_var.get()):
                frame = cv2.flip(frame, 1)

            # Process
            landmarks, annotated, has_hands = self.tracker.process_frame(frame)

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

            # Update overlay labels
            self.root.after(0, lambda h=has_hands: self.hands_overlay_label.configure(
                text="Hands OK" if h else "No Hands",
                text_color=Theme.ACCENT_SUCCESS if h else Theme.TEXT_SECONDARY
            ))

            # Convert frame for display
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Calculate display size maintaining aspect ratio
            display_width = 800
            display_height = int(display_width * frame.shape[0] / frame.shape[1])
            pil_image = pil_image.resize((display_width, display_height), Image.Resampling.LANCZOS)

            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(display_width, display_height))
            self.root.after(0, lambda img=ctk_image: self.video_label.configure(image=img))

            # FPS
            elapsed = time.time() - start_time
            self.frame_times.append(elapsed)
            if self.frame_times:
                self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))

            # Update display
            self.root.after(0, self._update_display)

            # Control speed
            if self.mode == "video":
                target = (1.0 / self.video_fps) / self.config.video_playback_speed
                time.sleep(max(0.001, target - elapsed))
            else:
                time.sleep(0.001)

    def _update_display(self):
        """Update all display elements."""
        # FPS overlay
        self.fps_overlay_label.configure(text=f"FPS: {self.fps:.1f}")

        # Prediction
        if self.current_prediction:
            self.prediction_label.configure(text=self.current_prediction)

            # Check GT match
            if self.mode == "video" and self.ground_truth:
                gt_score = next((p for g, p in self.current_top_k if g == self.ground_truth), 0.0)

                if gt_score > 0:
                    self.gt_score_label.configure(text=f"({gt_score * 100:.1f}%)")
                    color = Theme.ACCENT_SUCCESS if gt_score >= 0.5 else (
                        Theme.ACCENT_WARNING if gt_score >= 0.2 else Theme.ACCENT_DANGER)
                    self.gt_score_label.configure(text_color=color)
                else:
                    self.gt_score_label.configure(text="(not in top-10)", text_color=Theme.ACCENT_DANGER)

                if self.current_prediction == self.ground_truth:
                    self.match_label.configure(text="✓ MATCH", text_color=Theme.ACCENT_SUCCESS)
                    self.prediction_label.configure(text_color=Theme.ACCENT_SUCCESS)
                else:
                    self.match_label.configure(text="✗", text_color=Theme.ACCENT_DANGER)
                    self.prediction_label.configure(text_color=Theme.ACCENT_DANGER)
            else:
                self.match_label.configure(text="")
                self.gt_score_label.configure(text="")
                self.prediction_label.configure(text_color=Theme.ACCENT_SUCCESS)

            # Confidence
            conf_pct = int(self.current_confidence * 100)
            self.conf_value_label.configure(text=f"{conf_pct}%")
            self.conf_progress.set(self.current_confidence)

            color = Theme.ACCENT_SUCCESS if conf_pct >= 70 else (
                Theme.ACCENT_WARNING if conf_pct >= 50 else Theme.ACCENT_DANGER)
            self.conf_value_label.configure(text_color=color)
            self.conf_progress.configure(progress_color=color)
        else:
            self.prediction_label.configure(text="---", text_color=Theme.TEXT_SECONDARY)
            self.match_label.configure(text="")
            self.gt_score_label.configure(text="")
            self.conf_value_label.configure(text="0%")
            self.conf_progress.set(0)

        # Top-10 predictions
        try:
            self.predictions_text.configure(state="normal")
            self.predictions_text.delete("1.0", "end")
            for i, (gloss, prob) in enumerate(self.current_top_k):
                marker = " ★" if (self.mode == "video" and gloss == self.ground_truth) else ""
                line = f"{i + 1:2}. {gloss:18} {prob * 100:5.1f}%{marker}\n"
                self.predictions_text.insert("end", line)
            self.predictions_text.configure(state="disabled")
        except:
            pass

        # History
        if self.gloss_history:
            self.history_label.configure(text=" → ".join(self.gloss_history[-8:]))
        else:
            self.history_label.configure(text="---")

        # Statistics
        self.stats_fps_label.configure(text=f"FPS: {self.fps:.1f}")
        self.stats_buffer_label.configure(text=f"Buffer: {len(self.landmark_buffer)}/{self.config.buffer_size}")

        # Frame counter
        if self.mode == "video":
            self.frame_label.configure(text=f"Frame: {self.video_current_frame} / {self.video_total_frames}")

    def run(self):
        """Run the application."""
        print("\n🚀 Starting SignNet Modern GUI...")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind('<q>', lambda e: self._on_close())
        self.root.bind('<Escape>', lambda e: self._on_close())
        self.root.bind('<space>', lambda e: self._toggle_pause() if self.mode == "video" else None)
        self.root.bind('<r>', lambda e: self._restart_video() if self.mode == "video" else None)
        self.root.bind('<c>', lambda e: self._clear_history())

        self.root.mainloop()

    def _on_close(self):
        """Handle window close."""
        self.running = False
        time.sleep(0.2)
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
    print("SignNet Modern GUI")
    print("  Clean design with CustomTkinter")
    print("  Features: Webcam + Video + Thumbnail Browser")
    print("=" * 60)

    # Check for customtkinter
    try:
        import customtkinter
    except ImportError:
        print("\n❌ CustomTkinter not installed!")
        print("   Run: pip install customtkinter")
        return

    config = DemoConfig()

    try:
        app = SignNetModernGUI(config)
        app.run()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 Closed!")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()