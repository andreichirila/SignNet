"""
🎬 SignNetPlus Live Demo GUI
Real-time Sign Language Recognition with MediaPipe

Author: Roman Schläpfer, Andrei Chirila
Date: 2025-10-25
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

# ============================================================================
# 🏗️ MODEL LOADING
# ============================================================================

class SignNetModel:
    """Load and run SignBERT model for inference"""

    def __init__(self, model_path: Path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print(f"📦 Loading model from: {model_path}")

        # Load checkpoint (PyTorch 2.6+ compatibility)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Get vocab
        self.vocab = checkpoint['vocab']
        self.idx_to_gloss = {v: k for k, v in self.vocab.items()}

        # Get config
        config = checkpoint.get('config', {
            'vocab_size': len(self.vocab),
            'hidden_dim': 320,
            'num_layers': 3,
            'dropout': 0.4
        })

        # Import model architecture
        from SignNetPlusModel_Base import SignBERTBiGRU

        # Create model
        self.model = SignBERTBiGRU(
            vocab_size=config['vocab_size'],
            hidden_dim=config.get('hidden_dim', 320),
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.4)
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✅ Model loaded!")
        print(f"   Device: {self.device}")
        print(f"   Vocabulary: {len(self.vocab)} glosses")
        print(f"   Model params: {sum(p.numel() for p in self.model.parameters()):,}")

    def predict(self, landmarks: np.ndarray, threshold=0.3):
        """
        Predict sign from landmarks

        Args:
            landmarks: Array of shape (T, 1659) - sequence of landmarks
            threshold: Confidence threshold for prediction

        Returns:
            prediction: Predicted gloss
            confidence: Prediction confidence
        """
        if len(landmarks) < 5:  # Need minimum frames
            return None, 0.0

        with torch.no_grad():
            # Prepare input
            x = torch.FloatTensor(landmarks).unsqueeze(0).to(self.device)  # (1, T, 1659)
            lengths = torch.LongTensor([len(landmarks)]).to(self.device)

            # Forward pass
            log_probs = self.model(x, lengths)  # (T, 1, vocab_size)

            # Get predictions
            probs = torch.exp(log_probs).squeeze(1)  # (T, vocab_size)

            # Average over time
            avg_probs = probs.mean(dim=0)  # (vocab_size,)

            # Get top prediction
            confidence, pred_idx = torch.max(avg_probs, dim=0)
            confidence = confidence.item()
            pred_idx = pred_idx.item()

            # Check threshold
            if confidence < threshold:
                return None, confidence

            # Decode
            if pred_idx in self.idx_to_gloss:
                prediction = self.idx_to_gloss[pred_idx]

                # Skip special tokens
                if prediction in ['<BLANK>', '<PAD>']:
                    return None, confidence

                return prediction, confidence

            return None, confidence


# ============================================================================
# 🎥 MEDIAPIPE TRACKER
# ============================================================================

class MediaPipeTracker:
    """Track face, pose, and hands with MediaPipe"""

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

    def process_frame(self, frame):
        """
        Process frame and extract landmarks

        Returns:
            landmarks: Flattened array of all landmarks (1659 features)
            annotated_frame: Frame with drawn landmarks
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
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )

        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )

        # Extract landmarks as flat array
        landmarks = self.extract_landmarks(results)

        return landmarks, annotated, results

    def extract_landmarks(self, results):
        """Extract all landmarks as flat array (1659 features)"""
        landmarks = []

        # Left hand (21 points * 3 coords = 63)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)

        # Right hand (21 points * 3 coords = 63)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)

        # Pose (33 points * 4 coords (x,y,z,visibility) = 132)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            landmarks.extend([0.0] * 132)

        # Face (468 points * 3 coords = 1404)
        if results.face_landmarks:
            for lm in results.face_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 1404)

        # Total: 63 + 63 + 132 + 1404 = 1662
        # But we need 1659, so take first 1659
        return np.array(landmarks[:1659], dtype=np.float32)

    def close(self):
        self.holistic.close()


# ============================================================================
# 🖼️ GUI APPLICATION
# ============================================================================

class SignNetPlusGUI:
    """Main GUI application"""

    def __init__(self, model_path: Path):
        # Initialize window
        self.root = tk.Tk()
        self.root.title("SignNet+ Live Demo")
        self.root.geometry("1400x800")
        self.root.configure(bg='#1e1e1e')

        # Load model
        self.model = SignNetModel(model_path)

        # Initialize tracker
        self.tracker = MediaPipeTracker()

        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # State
        self.running = False
        self.landmark_buffer = deque(maxlen=30)  # Buffer for sequence
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.fps = 0
        self.frame_times = deque(maxlen=30)

        # Setup UI
        self.setup_ui()

        # Start video thread
        self.running = True
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()

    def setup_ui(self):
        """Setup user interface"""

        # Title
        title = tk.Label(
            self.root,
            text="🎬 SignNet+ Live Recognition",
            font=('Arial', 24, 'bold'),
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

        # Video label
        self.video_label = tk.Label(video_frame, bg='#000000')
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right panel - Info
        info_frame = tk.Frame(main_frame, bg='#2d2d2d', relief=tk.RAISED, borderwidth=2, width=400)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        info_frame.pack_propagate(False)

        # Info title
        info_title = tk.Label(
            info_frame,
            text="📊 Recognition Info",
            font=('Arial', 16, 'bold'),
            bg='#2d2d2d',
            fg='#ffffff'
        )
        info_title.pack(pady=10)

        # Prediction display
        pred_frame = tk.Frame(info_frame, bg='#3d3d3d', relief=tk.SUNKEN, borderwidth=2)
        pred_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            pred_frame,
            text="Predicted Sign:",
            font=('Arial', 12),
            bg='#3d3d3d',
            fg='#cccccc'
        ).pack(pady=5)

        self.pred_label = tk.Label(
            pred_frame,
            text="---",
            font=('Arial', 32, 'bold'),
            bg='#3d3d3d',
            fg='#00ff00',
            wraplength=350
        )
        self.pred_label.pack(pady=10)

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
            font=('Arial', 24, 'bold'),
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
            text="Buffer: 0/30 frames",
            font=('Arial', 11),
            bg='#3d3d3d',
            fg='#ffffff'
        )
        self.buffer_label.pack(pady=2)

        self.device_label = tk.Label(
            stats_frame,
            text=f"Device: {self.model.device}",
            font=('Arial', 11),
            bg='#3d3d3d',
            fg='#ffffff'
        )
        self.device_label.pack(pady=2)

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
            "✅ Face camera directly",
            "✅ Good lighting required",
            "✅ Perform signs clearly",
            "✅ Wait for buffer to fill",
            "✅ Green = recognized!",
            "",
            "Press Q or ESC to quit"
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

        # Bottom status
        self.status_label = tk.Label(
            self.root,
            text="🟢 Live Recognition Active",
            font=('Arial', 12),
            bg='#1e1e1e',
            fg='#00ff00'
        )
        self.status_label.pack(pady=5)

    def video_loop(self):
        """Main video processing loop"""

        while self.running:
            start_time = time.time()

            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process with MediaPipe
            landmarks, annotated, results = self.tracker.process_frame(frame)

            # Add to buffer if landmarks detected
            has_detection = (
                results.left_hand_landmarks is not None or
                results.right_hand_landmarks is not None
            )

            if has_detection:
                self.landmark_buffer.append(landmarks)

            # Predict if buffer has enough frames
            if len(self.landmark_buffer) >= 10:
                # Stack landmarks
                sequence = np.stack(list(self.landmark_buffer))

                # Predict
                prediction, confidence = self.model.predict(sequence)

                if prediction:
                    self.current_prediction = prediction
                    self.current_confidence = confidence

            # Add info overlay
            annotated = self.add_overlay(annotated)

            # Convert for Tkinter
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            # Resize to fit
            frame_pil = frame_pil.resize((960, 540), Image.Resampling.LANCZOS)

            frame_tk = ImageTk.PhotoImage(frame_pil)

            # Update UI
            self.video_label.configure(image=frame_tk)
            self.video_label.image = frame_tk

            # Calculate FPS
            elapsed = time.time() - start_time
            self.frame_times.append(elapsed)
            if len(self.frame_times) > 0:
                self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))

            # Update info labels
            self.update_info()

            # Small delay
            time.sleep(0.01)

        # Cleanup
        self.cap.release()
        self.tracker.close()

    def add_overlay(self, frame):
        """Add text overlay to frame"""
        h, w = frame.shape[:2]

        # Semi-transparent overlay
        overlay = frame.copy()

        # Top bar
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        cv2.putText(frame, "SignNet+ Live", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w - 150, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Prediction overlay
        if self.current_prediction:
            # Bottom bar
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h-100), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

            # Prediction text
            cv2.putText(frame, f"Sign: {self.current_prediction}", (20, h-60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Confidence
            conf_percent = int(self.current_confidence * 100)
            cv2.putText(frame, f"Confidence: {conf_percent}%", (20, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        return frame

    def update_info(self):
        """Update info panel"""
        # Prediction
        if self.current_prediction:
            self.pred_label.config(text=self.current_prediction)
            conf_percent = int(self.current_confidence * 100)
            self.conf_label.config(text=f"{conf_percent}%")
            self.conf_bar['value'] = conf_percent
        else:
            self.pred_label.config(text="---")
            self.conf_label.config(text="0%")
            self.conf_bar['value'] = 0

        # Stats
        self.fps_label.config(text=f"FPS: {self.fps:.1f}")
        self.buffer_label.config(text=f"Buffer: {len(self.landmark_buffer)}/30 frames")

    def run(self):
        """Run the application"""
        self.root.mainloop()

    def close(self):
        """Cleanup"""
        self.running = False
        if self.video_thread.is_alive():
            self.video_thread.join(timeout=2.0)
        self.cap.release()
        self.tracker.close()


# ============================================================================
# 🚀 MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("🎬 SignNet+ Live Demo")
    print("="*70)

    # Model path
    model_path = Path("Fertige_Models/current_model_converted.pth")

    print(f"\n✅ Model path: {model_path}")
    print("\n🎥 Starting live demo...")
    print("   Press Q or ESC to quit\n")

    # Create and run GUI
    try:
        app = SignNetPlusGUI(model_path)
        app.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 Shutting down...")
        cv2.destroyAllWindows()

    print("✅ Demo closed!")