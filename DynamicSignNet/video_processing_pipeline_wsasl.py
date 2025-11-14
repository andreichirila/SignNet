#!/usr/bin/env python3
"""
WSASL (Kaggle World Sign Language) Dataset Processing Pipeline
WITH TRAIN/TEST SPLIT SUPPORT

Output structure with split subdirectories:
wsasl_frames/
├── train/
│   ├── gloss1/
│   │   ├── 0000/
│   │   │   ├── frame_000000.png
│   │   │   └── ...
│   │   └── 0001/
│   └── gloss2/
├── test/
│   ├── gloss1/
│   └── gloss2/
└── dataset_metadata.json

Usage:
    python wsasl_pipeline_split.py \
        --videos-root ./videos \
        --metadata ./wlasl_dataset_raw/WLASL_v0.3.json \
        --output-root ./wsasl_frames \
        --landmarks-root ./wsasl_landmarks \
        --n-workers 4
"""

import os
import cv2
import json
import numpy as np
import glob
import argparse
from pathlib import Path
from multiprocessing import Pool
import traceback
from tqdm import tqdm
from collections import defaultdict

try:
    import mediapipe as mp
except ImportError:
    print("ERROR: MediaPipe not installed. Install with: pip install mediapipe")
    exit(1)


# ============================================================================
# METADATA PARSING FOR WLASL
# ============================================================================

def load_wlasl_metadata(metadata_path):
    """
    Load WLASL metadata in correct structure.

    Structure (WLASL v0.3):
    [
      {
        "gloss": "word_label",
        "instances": [
          {"video_id": "00335", "split": "train", ...},
          {"video_id": "00603", "split": "train", ...},
          ...
        ]
      },
      ...
    ]

    Returns:
        dict: {video_id: {gloss, split, ...}}
    """
    if not Path(metadata_path).exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    metadata = {}

    if isinstance(raw_data, list):
        print(f"  Detected WLASL list format with {len(raw_data)} glosses")

        for gloss_entry in raw_data:
            gloss = gloss_entry.get("gloss", "unknown")
            instances = gloss_entry.get("instances", [])

            for instance in instances:
                video_id = instance.get("video_id")
                if video_id:
                    metadata[str(video_id)] = {
                        "gloss": gloss,
                        "split": instance.get("split", "unknown"),
                        **{k: v for k, v in instance.items() if k not in ["video_id", "gloss"]}
                    }

        print(f"  ✓ Extracted {len(metadata)} videos from {len(raw_data)} glosses")
        return metadata

    elif isinstance(raw_data, dict):
        print(f"  Detected dict format with {len(raw_data)} items")
        return raw_data

    else:
        raise ValueError(f"Unexpected metadata format: {type(raw_data)}")


def extract_frames_from_video(video_path, output_dir, max_frames=None):
    """Extract frames from a single video file."""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames is not None and saved_count >= max_frames:
            break

        frame_path = os.path.join(output_dir, f"frame_{saved_count:06d}.png")
        cv2.imwrite(frame_path, frame)
        saved_count += 1

    cap.release()
    return saved_count


# ============================================================================
# STEP 1: VIDEO EXTRACTION WITH SPLIT SUPPORT
# ============================================================================

def wlasl_video_to_frames(video_root="./videos", metadata_path="./wlasl_dataset_raw/WLASL_v0.3.json",
                          frame_output_root="./wsasl_frames", max_frames_per_video=None):
    """
    Convert WLASL videos to frames organized by split and gloss.

    Output structure:
    wsasl_frames/
    ├── train/
    │   ├── glossname1/
    │   │   ├── 0000/
    │   │   │   ├── frame_000000.png
    │   │   │   └── ...
    │   │   └── 0001/
    │   └── glossname2/
    ├── test/
    │   ├── glossname1/
    │   └── glossname2/
    └── dataset_metadata.json
    """

    if not os.path.exists(video_root):
        print(f"Error: Video root directory not found: {video_root}")
        return

    # Load metadata
    print(f"\nLoading metadata from: {metadata_path}")
    metadata = load_wlasl_metadata(metadata_path)
    print(f"✓ Loaded metadata for {len(metadata)} videos\n")

    # Group by split AND gloss
    split_gloss_videos = defaultdict(lambda: defaultdict(list))
    split_counts = defaultdict(int)

    for video_id, info in metadata.items():
        split = str(info.get("split", "unknown")).lower().strip()
        gloss = str(info.get("gloss", "unknown")).lower().strip()
        split_gloss_videos[split][gloss].append(str(video_id))
        split_counts[split] += 1

    print(f"✓ Split distribution:")
    for split, count in sorted(split_counts.items()):
        print(f"  {split:10s}: {count:5d} videos")

    total_glosses = len(set(metadata[vid]["gloss"] for vid in metadata))
    print(f"\n✓ Found {total_glosses} unique glosses\n")

    os.makedirs(frame_output_root, exist_ok=True)

    total_videos = 0
    total_frames = 0
    processed_metadata = {}
    failed_videos = []

    # Process each split
    for split in sorted(split_gloss_videos.keys()):
        print(f"{'='*70}")
        print(f"PROCESSING SPLIT: {split.upper()}")
        print(f"{'='*70}\n")

        gloss_dict = split_gloss_videos[split]

        # Process each gloss within split
        for gloss_idx, (gloss, video_ids) in enumerate(sorted(gloss_dict.items()), 1):
            print(f"[{gloss_idx:4d}/{len(gloss_dict):4d}] Gloss: {gloss:40s} ", end="", flush=True)

            # Create: split/gloss/0000, split/gloss/0001, etc.
            gloss_output = os.path.join(frame_output_root, split, gloss)
            os.makedirs(gloss_output, exist_ok=True)

            gloss_frames = 0
            processed_in_gloss = 0

            for local_idx, video_id in enumerate(sorted(video_ids)):
                # Try to find video file
                video_path = None
                for candidate in [
                    os.path.join(video_root, f"{video_id}.mp4"),
                    os.path.join(video_root, f"{video_id}.avi"),
                    os.path.join(video_root, f"{video_id}.mov"),
                    os.path.join(video_root, str(video_id)),
                ]:
                    if os.path.exists(candidate):
                        video_path = candidate
                        break

                if video_path is None:
                    failed_videos.append(video_id)
                    continue

                date_idx = f"{local_idx:04d}"
                frame_output_dir = os.path.join(gloss_output, date_idx)

                num_frames = extract_frames_from_video(video_path, frame_output_dir, max_frames_per_video)

                if num_frames > 0:
                    processed_metadata[str(video_id)] = {
                        "gloss": gloss,
                        "split": split,
                        "gloss_idx": gloss_idx - 1,
                        "local_idx": local_idx,
                        "num_frames": num_frames
                    }

                    gloss_frames += num_frames
                    total_frames += num_frames
                    processed_in_gloss += 1
                    total_videos += 1

            print(f"({processed_in_gloss:3d} videos, {gloss_frames:6d} frames)")

        print()

    # Save metadata
    metadata_output_path = os.path.join(frame_output_root, "dataset_metadata.json")
    with open(metadata_output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_metadata, f, indent=2, ensure_ascii=False)

    print(f"{'='*70}")
    print(f"✓ WLASL Video extraction complete!")
    print(f"  Total videos processed: {total_videos}")
    print(f"  Total frames extracted: {total_frames}")
    print(f"  Total glosses: {total_glosses}")
    print(f"  Failed videos: {len(failed_videos)}")
    if failed_videos:
        print(f"  (First 5: {failed_videos[:5]})")
    print(f"  Output root: {frame_output_root}")
    print(f"  Metadata: {metadata_output_path}")
    print(f"{'='*70}\n")


# ============================================================================
# STEP 2: DATASET CLASS WITH SPLIT SUPPORT
# ============================================================================

class WSASLDataset:
    """Dataset loader for extracted frames with split support."""

    def __init__(self, root, split=None, max_frames=None):
        """
        Initialize dataset.

        Args:
            root: Root directory containing split folders (train, test, etc.)
            split: Filter to specific split ('train', 'test', None for all)
            max_frames: Maximum frames per sample
        """
        self.root = Path(root)
        self.split = split
        self.max_frames = max_frames
        self.samples = []

        # Load metadata
        metadata_path = self.root / "dataset_metadata.json"
        self.metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

        # Find split folders
        split_dirs = []
        if split:
            split_path = self.root / split
            if split_path.exists():
                split_dirs = [split_path]
        else:
            # Use all split folders (train, test, val, etc.)
            split_dirs = [d for d in self.root.iterdir() if d.is_dir() and d.name not in ["dataset_metadata.json"]]

        # Scan gloss folders within each split
        for split_dir in sorted(split_dirs):
            split_name = split_dir.name

            for gloss_dir in sorted(split_dir.iterdir()):
                if not gloss_dir.is_dir():
                    continue

                gloss_name = gloss_dir.name

                # Scan video subfolders
                for video_dir in sorted(gloss_dir.iterdir()):
                    if not video_dir.is_dir():
                        continue

                    frame_paths = sorted(glob.glob(os.path.join(str(video_dir), "*.png")))

                    if len(frame_paths) > 0:
                        self.samples.append({
                            "id": f"{split_name}_{gloss_name}_{video_dir.name}",
                            "split": split_name,
                            "gloss": gloss_name,
                            "video_idx": video_dir.name,
                            "frames": frame_paths,
                            "annotation": [gloss_name]
                        })

    def __len__(self):
        return len(self.samples)

    def get_sample(self, idx):
        return self.samples[idx]


# ============================================================================
# STEP 3: LANDMARK EXTRACTION
# ============================================================================

def process_one(args):
    """Extract landmarks from single sample."""
    idx, sample, save_dir, sample_id = args

    try:
        if not hasattr(process_one, "mp_hands"):
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
            return

        frames = np.stack(frames)

        # Extract landmarks
        landmarks = []
        handedness_data = []

        for frame in frames:
            hand_lms = np.zeros(126, dtype=np.float32)
            hand_handedness = ["NONE", "NONE"]
            face_lms = np.zeros(1434, dtype=np.float32)
            pose_lms = np.zeros(99, dtype=np.float32)

            # Hand landmarks
            results_hands = mp_hands.process(frame)
            if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
                for hand_landmarks, handedness in zip(
                    results_hands.multi_hand_landmarks[:2], 
                    results_hands.multi_handedness[:2]
                ):
                    hand_label = handedness.classification[0].label
                    hand_idx = 0 if hand_label == "LEFT" else 1
                    hand_handedness[hand_idx] = hand_label

                    for j, lm in enumerate(hand_landmarks.landmark):
                        base = hand_idx * 21 * 3 + j * 3
                        if base + 3 <= len(hand_lms):
                            hand_lms[base:base + 3] = [lm.x, lm.y, lm.z]

            # Face landmarks
            results_face = mp_face.process(frame)
            if results_face.multi_face_landmarks:
                for j, lm in enumerate(results_face.multi_face_landmarks[0].landmark):
                    base = j * 3
                    if base + 3 <= len(face_lms):
                        face_lms[base:base + 3] = [lm.x, lm.y, lm.z]

            # Pose landmarks
            results_pose = mp_pose.process(frame)
            if results_pose.pose_landmarks:
                for j, lm in enumerate(results_pose.pose_landmarks.landmark):
                    base = j * 3
                    if base + 3 <= len(pose_lms):
                        pose_lms[base:base + 3] = [lm.x, lm.y, lm.z]

            combined = np.concatenate([hand_lms, face_lms, pose_lms])
            landmarks.append(combined)
            handedness_data.append(hand_handedness)

        landmarks = np.stack(landmarks)
        glosses = sample["annotation"]

        # Save
        save_path = os.path.join(save_dir, f"{sample_id}.npz")
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(
            save_path, 
            landmarks=landmarks, 
            glosses=np.array(glosses),
            handedness=np.array(handedness_data)
        )

    except Exception as e:
        print(f"Error: {sample_id}: {str(e)}")


def preprocess_and_save_mp(root, save_dir, split=None, max_frames=None, n_workers=8):
    """Extract landmarks using multiprocessing."""
    if not os.path.exists(root):
        print(f"Error: Root directory not found: {root}")
        return

    dataset = WSASLDataset(root, split=split, max_frames=max_frames)
    print(f"\n{'='*70}")
    print(f"Loaded {len(dataset.samples)} samples")
    if split:
        print(f"Filter: split='{split}'")
    print(f"{'='*70}\n")

    os.makedirs(save_dir, exist_ok=True)

    arglist = [
        (idx, dataset.samples[idx], save_dir, dataset.samples[idx]["id"]) 
        for idx in range(len(dataset.samples))
    ]

    print(f"Processing {len(arglist)} samples with {n_workers} workers...\n")
    with Pool(n_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_one, arglist), 
            total=len(arglist),
            desc="Extracting landmarks"
        ):
            pass

    print(f"\n{'='*70}")
    print(f"✓ Landmarks saved to: {save_dir}")
    print(f"{'='*70}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="WLASL Dataset Processing Pipeline with Split Support")
    parser.add_argument("--videos-root", type=str, default="./videos",
                        help="Root directory containing MP4 files")
    parser.add_argument("--metadata", type=str, default="./wlasl_dataset_raw/WLASL_v0.3.json",
                        help="Path to WLASL metadata JSON")
    parser.add_argument("--output-root", type=str, default="./wsasl_frames",
                        help="Output directory for frames (will have train/test/val subdirs)")
    parser.add_argument("--landmarks-root", type=str, default="./wsasl_landmarks",
                        help="Output directory for landmarks")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames per video")
    parser.add_argument("--n-workers", type=int, default=4,
                        help="Number of workers")
    parser.add_argument("--split", type=str, default=None,
                        help="Filter landmarks to specific split (train/test/val) or None for all")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip frame extraction")
    parser.add_argument("--skip-landmark-extraction", action="store_true",
                        help="Skip landmark extraction")

    args = parser.parse_args()

    print("\n" + "="*70)
    print("WLASL DATASET PROCESSING PIPELINE (WITH SPLIT SUPPORT)")
    print("="*70)

    if not args.skip_extraction:
        print("\nSTEP 1: Extracting frames from videos...")
        wlasl_video_to_frames(
            video_root=args.videos_root,
            metadata_path=args.metadata,
            frame_output_root=args.output_root,
            max_frames_per_video=args.max_frames
        )
    else:
        print("\nSkipping frame extraction")

    if not args.skip_landmark_extraction:
        print("STEP 2: Extracting landmarks...")
        preprocess_and_save_mp(
            root=args.output_root,
            save_dir=args.landmarks_root,
            split=args.split,
            max_frames=args.max_frames,
            n_workers=args.n_workers
        )

    print("="*70)
    print("✓ PIPELINE COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
