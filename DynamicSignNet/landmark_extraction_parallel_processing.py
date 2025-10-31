import os
import numpy as np
import cv2
import glob
import torch
from torch.utils.data import Dataset
import mediapipe as mp
import json
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool


class PhoenixDataset(Dataset):
    """
    Phoenix Dataset loader for extracted folder structure.

    Folder structure:
    root/
    ├── CLASS_NAME/
    │   ├── 0000/  (date subfolder)
    │   │   ├── frame1.png
    │   │   ├── frame2.png
    │   │   └── ...
    │   └── 0001/
    │       └── ...
    └── dataset_summary.txt
    """

    def __init__(self, root, max_frames=None):
        """
        Initialize dataset from extracted folder structure.

        Args:
            root: Root directory containing class folders
            max_frames: Maximum frames per sample (None = use all)
        """
        self.root = Path(root)
        self.max_frames = max_frames
        self.samples = []

        # Scan all class folders
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir() or class_dir.name == "dataset_summary.txt":
                continue

            class_name = class_dir.name

            # Scan all date subfolders within class
            for date_dir in sorted(class_dir.iterdir()):
                if not date_dir.is_dir():
                    continue

                frame_paths = sorted(glob.glob(os.path.join(str(date_dir), "*.png")))

                if len(frame_paths) == 0:
                    continue

                self.samples.append({
                    "id": f"{class_name}_{date_dir.name}",
                    "class": class_name,
                    "date_idx": date_dir.name,
                    "frames": frame_paths,
                    "annotation": [class_name]  # Single label from folder name
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = []

        for fp in sample["frames"]:
            img = cv2.imread(fp)
            if img is None:
                print(f"Warning: Could not read frame {fp}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)

        if len(frames) == 0:
            raise ValueError(f"No frames for sample: {sample['id']}")

        # Optionally trim
        if self.max_frames is not None:
            frames = frames[:self.max_frames]

        frames = np.stack(frames)
        return frames, sample["annotation"]


def process_one(args):
    """
    Process single sample: extract landmarks with MediaPipe.
    Includes hand handedness (left/right) information with correct indexing.

    Args:
        args: (idx, sample, save_dir, sample_id)
    """
    idx, sample, save_dir, sample_id = args

    # Initialize MediaPipe in worker process
    if not hasattr(process_one, "mp_hands"):
        import mediapipe as mp
        process_one.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False, 
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        process_one.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, 
            refine_landmarks=True
        )
        process_one.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False
        )

    mp_hands = process_one.mp_hands
    mp_face = process_one.mp_face
    mp_pose = process_one.mp_pose

    # Load frames
    frames = []
    for fp in sample["frames"]:
        img = cv2.imread(fp)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    if len(frames) == 0:
        print(f"Warning: No frames for sample {sample['id']}")
        return

    frames = np.stack(frames)

    # Extract landmarks from each frame
    landmarks = []
    handedness_data = []  # Track handedness for each frame

    for frame in frames:
        hand_lms = np.zeros(126, dtype=np.float32)  # 2 hands * 21 points * 3 coords
        hand_handedness = ["NONE", "NONE"]  # Track which hand is which [LEFT, RIGHT]

        face_lms = np.zeros(1434, dtype=np.float32)  # 478 points * 3 coords (refined)
        pose_lms = np.zeros(99, dtype=np.float32)    # 33 points * 3 coords

        # Extract hand landmarks with handedness-based indexing
        results_hands = mp_hands.process(frame)
        if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
            for hand_landmarks, handedness in zip(
                results_hands.multi_hand_landmarks[:2], 
                results_hands.multi_handedness[:2]
            ):
                # Get handedness (LEFT or RIGHT)
                hand_label = handedness.classification[0].label

                # Determine index based on handedness, not enumeration order
                # This ensures LEFT hand is always at 0-62, RIGHT at 63-125
                if hand_label == "LEFT":
                    hand_idx = 0
                else:  # "RIGHT"
                    hand_idx = 1

                hand_handedness[hand_idx] = hand_label

                # Extract coordinates with bounds checking
                for j, lm in enumerate(hand_landmarks.landmark):
                    base = hand_idx * 21 * 3 + j * 3
                    if base + 3 <= len(hand_lms):
                        hand_lms[base:base + 3] = [lm.x, lm.y, lm.z]

        # Extract face landmarks with bounds checking
        results_face = mp_face.process(frame)
        if results_face.multi_face_landmarks:
            for j, lm in enumerate(results_face.multi_face_landmarks[0].landmark):
                base = j * 3
                if base + 3 <= len(face_lms):
                    face_lms[base:base + 3] = [lm.x, lm.y, lm.z]

        # Extract pose landmarks with bounds checking
        results_pose = mp_pose.process(frame)
        if results_pose.pose_landmarks:
            for j, lm in enumerate(results_pose.pose_landmarks.landmark):
                base = j * 3
                if base + 3 <= len(pose_lms):
                    pose_lms[base:base + 3] = [lm.x, lm.y, lm.z]

        # Combine all landmarks
        combined = np.concatenate([hand_lms, face_lms, pose_lms])
        landmarks.append(combined)
        handedness_data.append(hand_handedness)

    landmarks = np.stack(landmarks)
    glosses = sample["annotation"]

    # Save landmarks, glosses, and handedness
    save_path = os.path.join(save_dir, f"{sample_id}.npz")
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(
        save_path, 
        landmarks=landmarks, 
        glosses=np.array(glosses),
        handedness=np.array(handedness_data)
    )

    # Optional: Save readable format for debugging
    # os.makedirs("./landmarks_readable", exist_ok=True)
    # np.savetxt(os.path.join("./landmarks_readable", f"{sample_id}_landmarks.csv"), 
    #            landmarks.mean(axis=0), delimiter=',')
    # with open(os.path.join("./landmarks_readable", f"{sample_id}_label.json"), 'w') as f:
    #     json.dump({'glosses': glosses, 'handedness': [list(h) for h in handedness_data]}, 
    #               f, ensure_ascii=False, indent=2)


def preprocess_and_save_mp(root, save_dir, max_frames=None, n_workers=8):
    """
    Preprocess Phoenix dataset and extract landmarks using multiprocessing.

    Args:
        root: Root directory containing extracted class folders
        save_dir: Directory to save NPZ files
        max_frames: Maximum frames per sample
        n_workers: Number of worker processes
    """
    # Load dataset
    dataset = PhoenixDataset(root, max_frames=max_frames)
    print(f"Loaded {len(dataset)} samples")

    os.makedirs(save_dir, exist_ok=True)

    # Prepare arguments for multiprocessing
    arglist = [
        (idx, dataset.samples[idx], save_dir, dataset.samples[idx]["id"]) 
        for idx in range(len(dataset))
    ]

    # Process with multiprocessing
    with Pool(n_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_one, arglist), 
            total=len(arglist),
            desc=f"Processing landmarks"
        ):
            pass

    print(f"Saved landmarks to {save_dir}")


if __name__ == "__main__":
    # Your extracted dataset root (contains class folders)
    root = "./phoenix_words_raw_png"  # e.g., "./output_dir" from extractor
    save_dir = "./word_landmarks_extracted"

    preprocess_and_save_mp(
        root=root,
        save_dir=save_dir,
        max_frames=None,  # Set to limit frames per sample
        n_workers=8
    )
