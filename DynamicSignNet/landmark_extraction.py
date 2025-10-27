import os
import numpy as np
import cv2
import glob
import torch
from torch.utils.data import Dataset
import mediapipe as mp
import json

# Assume PhoenixDataset as before to load frames + annotations

mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True)
mp_pose = mp.solutions.pose.Pose(static_image_mode=False)

class PhoenixDataset(Dataset):
    def __init__(self, root, annotation_file, split='train', max_frames=None):
        self.root = root
        self.split = split  # 'train', 'dev', or 'test'
        self.max_frames = max_frames
        self.samples = []

        with open(annotation_file, "r", encoding="utf-8") as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 4:
                    continue
                sample_id, folder_pattern, signer, annotation = parts
                # Extract main folder name (before '/')
                folder_name = folder_pattern.split('/')[0]
                # Construct full path by adding split subfolder ('train', 'dev', 'test')
                folder_path = os.path.join(self.root, "features", "fullFrame-210x260px", self.split, folder_name, "1")
                # print(folder_path)
                frame_paths = sorted(glob.glob(os.path.join(folder_path, "*.png")))
                self.samples.append({
                    "id": sample_id,
                    "frames": frame_paths,
                    "signer": signer,
                    "annotation": annotation.split()
                })
    def __len__(self):
        return len(self.samples)
        

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = []
        for fp in sample["frames"]:
            img = cv2.imread(fp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        if len(frames) == 0:
            raise ValueError(f"No frames for sample: {sample['id']}, path={sample['frames']}")
        # Optionally trim
        if self.max_frames is not None:
            frames = frames[:self.max_frames]
        frames = np.stack(frames)
        return frames, sample["annotation"]

                
def extract_landmarks(frames):
    """
    Extracts 2-hand, face, and pose landmarks from a sequence of RGB frames.

    Args:
        frames (np.ndarray): Array of shape (T, H, W, 3), dtype=uint8, RGB.

    Returns:
        np.ndarray: Landmark array of shape (T, 1629), where each row is
                    [hand_landmarks (126), face_landmarks (1404), pose_landmarks (99)].
    """
    landmarks = []
    for frame in frames:
        # Allocate arrays (always fixed size, even if no detections)
        hand_lms = np.zeros(126, dtype=np.float32)    # 2 hands × 21 × 3
        face_lms = np.zeros(1434, dtype=np.float32) # Refined face landmarks are 478 points giving 478*3=1434 features
        pose_lms = np.zeros(99, dtype=np.float32)     # 33 × 3

        # Hands
        results_hands = mp_hands.process(frame)
        if results_hands.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks[:2]):  # up to 2 hands
                for j, lm in enumerate(hand_landmarks.landmark):
                    base = i * 63 + j * 3
                    hand_lms[base:base+3] = [lm.x, lm.y, lm.z]

        # Face
        results_face = mp_face.process(frame)
        if results_face.multi_face_landmarks:
            for j, lm in enumerate(results_face.multi_face_landmarks[0].landmark):
                base = j * 3
                face_lms[base:base+3] = [lm.x, lm.y, lm.z]

        # Pose
        results_pose = mp_pose.process(frame)
        if results_pose.pose_landmarks:
            for j, lm in enumerate(results_pose.pose_landmarks.landmark):
                base = j * 3
                pose_lms[base:base+3] = [lm.x, lm.y, lm.z]

        # Combine into one vector
        combined = np.concatenate([hand_lms, face_lms, pose_lms])
        landmarks.append(combined)

    return np.stack(landmarks)


def save_landmarks_readable(landmarks, glosses, save_folder, sample_id):
    os.makedirs(save_folder, exist_ok=True)
    # Save landmarks as CSV
    csv_path = os.path.join(save_folder, f"{sample_id}_landmarks.csv")
    # landmarks shape = (T, feature_dim), save one frame per line with comma sep
    np.savetxt(csv_path, landmarks, delimiter=',')

    # Save label as JSON
    label_path = os.path.join(save_folder, f"{sample_id}_label.json")
    with open(label_path, 'w', encoding='utf-8') as f:
        json.dump({'glosses': glosses}, f, ensure_ascii=False, indent=2)

def build_vocab(annotation_file, save_path):
    vocab = {}
    idx = 0
    with open(annotation_file, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 4:
                continue
            glosses = parts[3].split()
            for gloss in glosses:
                if gloss not in vocab:
                    vocab[gloss] = idx
                    idx += 1
    # Save vocab dictionary as JSON
    with open(save_path, "w", encoding="utf-8") as f_out:
        json.dump(vocab, f_out, ensure_ascii=False, indent=2)
    print(f"Saved vocabulary of size {len(vocab)} to {save_path}")
    
def preprocess_and_save(root, annotation_file, split, save_dir, max_frames=None):
    dataset = PhoenixDataset(root, annotation_file, split=split, max_frames=max_frames)
    os.makedirs(save_dir, exist_ok=True)

    with open(annotation_file, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for idx, line in enumerate(f):
            parts = line.strip().split("|")
            if len(parts) < 4:
                continue
            sample_id = parts[0]  # the ID from the corpus file
            frames, glosses = dataset[idx]
            landmarks = extract_landmarks(frames)
            save_path = os.path.join(save_dir, f"{sample_id}.npz")
            np.savez_compressed(save_path, landmarks=landmarks, glosses=np.array(glosses))
            save_landmarks_readable(landmarks, glosses, "./landmarks_readable", sample_id=sample_id)
            if idx % 10 == 0:
                print(f"[{split}] Processed sample {idx}/{len(dataset)}")


if __name__ == "__main__":
    root = 'phoenix-2014.v3.tar/phoenix2014-release/phoenix-2014-multisigner'
    train_ann = os.path.join(root,"annotations","manual","train.corpus.csv")
    dev_ann = os.path.join(root,"annotations","manual","dev.corpus.csv")
    test_ann = os.path.join(root,"annotations","manual","test.corpus.csv")

    print(f"Train{train_ann}")

    preprocess_and_save(root, train_ann, "train", "./landmarks_train")
    preprocess_and_save(root, dev_ann, "dev", "./landmarks_dev")
    preprocess_and_save(root, test_ann, "test", "./landmarks_test")

    build_vocab(train_ann, "vocab.json")
