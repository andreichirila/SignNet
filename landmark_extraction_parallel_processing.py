import os
import numpy as np
import cv2
import glob
import torch
from torch.utils.data import Dataset
import mediapipe as mp
import json
from tqdm import tqdm

# Assume PhoenixDataset as before to load frames + annotations

#mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
#mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)
#mp_pose = mp.solutions.pose.Pose(static_image_mode=False)


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

from multiprocessing import Pool


def process_one(args):
    idx, sample, split, save_dir, sample_id = args
    if not hasattr(process_one, "mp_hands"):
        import mediapipe as mp
        process_one.mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
        # Enable refine_landmarks here
        process_one.mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True)
        process_one.mp_pose = mp.solutions.pose.Pose(static_image_mode=False)

    mp_hands = process_one.mp_hands
    mp_face = process_one.mp_face
    mp_pose = process_one.mp_pose

    frames = []
    for fp in sample["frames"]:
        img = cv2.imread(fp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    if len(frames) == 0:
        print(f"Warning: No frames for sample {sample['id']}")
        return
    frames = np.stack(frames)

    landmarks = []
    for frame in frames:
        hand_lms = np.zeros(126, dtype=np.float32)
        # Face landmarks size will be larger due to refine_landmarks=True, adjust accordingly below!
        # Refined face landmarks are 478 points giving 478*3=1434 features
        face_lms = np.zeros(1434, dtype=np.float32)
        pose_lms = np.zeros(99, dtype=np.float32)

        results_hands = mp_hands.process(frame)
        if results_hands.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks[:2]):
                for j, lm in enumerate(hand_landmarks.landmark):
                    base = i * 63 + j * 3
                    hand_lms[base:base + 3] = [lm.x, lm.y, lm.z]
        results_face = mp_face.process(frame)
        if results_face.multi_face_landmarks:
            for j, lm in enumerate(results_face.multi_face_landmarks[0].landmark):
                base = j * 3
                face_lms[base:base + 3] = [lm.x, lm.y, lm.z]
        results_pose = mp_pose.process(frame)
        if results_pose.pose_landmarks:
            for j, lm in enumerate(results_pose.pose_landmarks.landmark):
                base = j * 3
                pose_lms[base:base + 3] = [lm.x, lm.y, lm.z]

        combined = np.concatenate([hand_lms, face_lms, pose_lms])
        landmarks.append(combined)
    landmarks = np.stack(landmarks)
    glosses = sample["annotation"]

    save_path = os.path.join(save_dir, f"{sample_id}.npz")
    np.savez_compressed(save_path, landmarks=landmarks, glosses=np.array(glosses))

    os.makedirs("./landmarks_readable", exist_ok=True)
    np.savetxt(os.path.join("./landmarks_readable", f"{sample_id}_landmarks.csv"), landmarks, delimiter=',')
    with open(os.path.join("./landmarks_readable", f"{sample_id}_label.json"), 'w', encoding='utf-8') as f:
        json.dump({'glosses': glosses}, f, ensure_ascii=False, indent=2)
    if idx % 10 == 0:
        print(f"[{split}] Processed sample {idx}")


def preprocess_and_save_mp(root, annotation_file, split, save_dir, max_frames=None, n_workers=8):
    dataset = PhoenixDataset(root, annotation_file, split=split, max_frames=max_frames)
    os.makedirs(save_dir, exist_ok=True)
    arglist = [(idx, dataset.samples[idx], split, save_dir, dataset.samples[idx]["id"]) for idx in range(len(dataset))]
    with Pool(n_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_one, arglist), total=len(arglist)):
            pass



if __name__ == "__main__":
    root = 'phoenix-2014.v3.tar/phoenix2014-release/phoenix-2014-multisigner'
    train_ann = os.path.join(root,"annotations","manual","train.corpus.csv")
    dev_ann = os.path.join(root,"annotations","manual","dev.corpus.csv")
    test_ann = os.path.join(root,"annotations","manual","test.corpus.csv")
    preprocess_and_save_mp(root, train_ann, "train", "./landmarks_train", n_workers=8)
    # preprocess_and_save_mp(root, dev_ann, "dev", "./landmarks_dev", n_workers=8)
    # preprocess_and_save_mp(root, test_ann, "test", "./landmarks_test", n_workers=8)

