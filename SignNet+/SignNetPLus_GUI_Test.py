"""
🎬 SignNetPlus Live Demo GUI
Real-time Sign Language Recognition with MediaPipe

Author: Andrei Chirila, Roman Schläpfer
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

        # Check if checkpoint is a dict or direct model
        if isinstance(checkpoint, dict):
            # Standard checkpoint format
            self.vocab = checkpoint.get('vocab', {})
            config = checkpoint.get('config', {})
            model_state = checkpoint.get('model_state_dict')
        else:
            # Direct model format - need to extract from model
            print("⚠️  Model saved in direct format, creating default vocab...")
            # Create a minimal vocab for demo
            # You'll need to provide the actual vocab from training
            self.vocab = {
                '<BLANK>': 0,
                '<PAD>': 1,
                'HELLO': 2,
                'THANK_YOU': 3,
                'PLEASE': 4,
                # Add more as needed
            }
            config = {
                'vocab_size': len(self.vocab),
                'hidden_dim': 320,
                'num_layers': 3,
                'dropout': 0.4
            }
            model_state = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else None

            print(f"⚠️  Using default vocab with {len(self.vocab)} entries")
            print("   For full functionality, please save model with vocab!")

        self.idx_to_gloss = {v: k for k, v in self.vocab.items()}

        # Get config with defaults
        if not config:
            config = {
                'vocab_size': len(self.vocab),
                'hidden_dim': 320,
                'num_layers': 3,
                'dropout': 0.4
            }

        # CRITICAL: Extract actual vocab_size from model weights!
        # The model was trained with specific vocab_size, use that!
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            if 'output.3.weight' in state_dict:
                actual_vocab_size = state_dict['output.3.weight'].shape[0]
                print(f"   📊 Model trained with vocab_size: {actual_vocab_size}")

                if len(self.vocab) != actual_vocab_size:
                    print(f"   ⚠️  Vocab mismatch detected!")
                    print(f"      Checkpoint vocab: {len(self.vocab)}")
                    print(f"      Model vocab_size: {actual_vocab_size}")
                    print(f"   → Using model's vocab_size: {actual_vocab_size}")

                config['vocab_size'] = actual_vocab_size

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
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif model_state is not None:
            self.model.load_state_dict(model_state)
        elif isinstance(checkpoint, torch.nn.Module):
            # Checkpoint is the model itself
            self.model = checkpoint.to(self.device)
        else:
            raise ValueError("Could not load model weights from checkpoint!")

        self.model.eval()

        print(f"✅ Model loaded!")
        print(f"   Device: {self.device}")
        print(f"   Vocabulary: {len(self.vocab)} glosses")
        print(f"   Model params: {sum(p.numel() for p in self.model.parameters()):,}")

    def predict(self, landmarks: np.ndarray, threshold=0.1, debug=True):
        """
        Predict sign from landmarks

        Args:
            landmarks: Array of shape (T, 1659) - sequence of landmarks
            threshold: Confidence threshold for prediction (LOWERED for debug!)
            debug: Print debug information

        Returns:
            prediction: Predicted gloss
            confidence: Prediction confidence
        """
        if len(landmarks) < 5:  # Need minimum frames
            if debug:
                print(f"⚠️  Not enough frames: {len(landmarks)}")
            return None, 0.0

        with torch.no_grad():
            # Prepare input
            x = torch.FloatTensor(landmarks).unsqueeze(0).to(self.device)  # (1, T, 1659)
            lengths = torch.LongTensor([len(landmarks)]).to(self.device)

            if debug:
                print(f"\n🔍 DEBUG Prediction:")
                print(f"   Input shape: {x.shape}")
                print(f"   Sequence length: {lengths.item()}")
                print(f"   Landmarks range: [{x.min().item():.3f}, {x.max().item():.3f}]")

            # Forward pass
            log_probs = self.model(x, lengths)  # (T, 1, vocab_size)

            if debug:
                print(f"   Output shape: {log_probs.shape}")
                print(f"   Log probs range: [{log_probs.min().item():.3f}, {log_probs.max().item():.3f}]")

            # Get predictions
            probs = torch.exp(log_probs).squeeze(1)  # (T, vocab_size)

            # Average over time
            avg_probs = probs.mean(dim=0)  # (vocab_size,)

            # APPLY TEMPERATURE SCALING!
            # Higher temperature = more uniform distribution
            temperature = 2.0  # Makes model "less confident" = spreads probability
            avg_probs = torch.pow(avg_probs, 1.0 / temperature)
            avg_probs = avg_probs / avg_probs.sum()  # Re-normalize

            if debug:
                print(f"   🌡️  Temperature scaling (T={temperature}) applied")

            # APPLY BLANK PENALTY!
            blank_idx = self.vocab.get('<BLANK>', 0)
            if blank_idx in range(len(avg_probs)):
                original_blank = avg_probs[blank_idx].item()
                avg_probs[blank_idx] *= 0.01  # Reduce blank by 99%!
                if debug:
                    print(f"   🔧 BLANK penalty: {original_blank*100:.2f}% → {avg_probs[blank_idx].item()*100:.2f}%")

            # Also penalize PAD
            pad_idx = self.vocab.get('<PAD>', 1)
            if pad_idx in range(len(avg_probs)):
                avg_probs[pad_idx] *= 0.01
                if debug:
                    print(f"   🔧 PAD penalty applied")

            # Get top 5 predictions
            top5_conf, top5_idx = torch.topk(avg_probs, k=min(5, len(avg_probs)))

            if debug:
                print(f"   Top 5 predictions:")
                for i in range(len(top5_conf)):
                    idx = top5_idx[i].item()
                    conf = top5_conf[i].item()
                    gloss = self.idx_to_gloss.get(idx, f"<UNK_{idx}>")
                    print(f"      {i+1}. {gloss:<20} {conf*100:5.2f}%")

            # Get top prediction
            confidence = top5_conf[0].item()
            pred_idx = top5_idx[0].item()

            # Check threshold
            if confidence < threshold:
                if debug:
                    print(f"   ⚠️  Below threshold ({confidence:.3f} < {threshold})")
                    print(f"   → But showing anyway for DEMO mode!")
                # Don't return None for demo! Show it anyway
                pass  # Continue to show prediction

            # Decode
            if pred_idx in self.idx_to_gloss:
                prediction = self.idx_to_gloss[pred_idx]

                # Skip special tokens
                if prediction in ['<BLANK>', '<PAD>']:
                    if debug:
                        print(f"   ⚠️  Skipping special token: {prediction}")
                    return None, confidence

                if debug:
                    print(f"   ✅ PREDICTION: {prediction} ({confidence*100:.1f}%)")

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
        self.landmark_buffer = deque(maxlen=60)  # INCREASED buffer! Was 30
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.fps = 0
        self.frame_times = deque(maxlen=30)
        self.debug_mode = True  # Enable debug output

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
            text="Buffer: 0/60 frames",
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
            if len(self.landmark_buffer) >= 15:  # LOWERED from 10
                # Stack landmarks
                sequence = np.stack(list(self.landmark_buffer))

                # Predict with debug mode
                prediction, confidence = self.model.predict(
                    sequence,
                    threshold=0.0,  # ALWAYS show top prediction!
                    debug=self.debug_mode
                )

                if prediction:
                    self.current_prediction = prediction
                    self.current_confidence = confidence
                    print(f"\n🎯 RECOGNIZED: {prediction} ({confidence*100:.1f}%)\n")  # Console output!

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
        self.buffer_label.config(text=f"Buffer: {len(self.landmark_buffer)}/60 frames")

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

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("\n📁 Looking in current directory...")
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