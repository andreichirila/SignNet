import os
import cv2
import mediapipe as mp
import numpy as np
import csv
import json

# Initialisiere MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# Schritt 1: Pfade definieren
root_dir = r"D:\OST\SignNet\SignNet+\phoenix-2014.v3\phoenix2014-release\phoenix-2014-multisigner"
raw_images_root = os.path.join(root_dir, "features", "fullFrame-210x260px")
split = "train"
output_dir = 'parsed_landmarks_dataset'
checkpoint_file = 'checkpoint.json'
os.makedirs(output_dir, exist_ok=True)

# Lade oder initialisiere Checkpoint
start_video_idx = 0
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
        last_video_id = checkpoint.get('last_video_id')
        print(f"Checkpoint geladen: last_video_id={last_video_id}")
else:
    last_video_id = None

# Vergewissern Sie sich, dass die Pfade existieren
if not os.path.exists(root_dir):
    raise ValueError(f"Root-Verzeichnis nicht gefunden: {root_dir}")
if not os.path.exists(raw_images_root):
    raise ValueError(f"Features-Verzeichnis nicht gefunden: {raw_images_root}")

# Schritt 2: Metadaten laden
annotations_path = os.path.join(root_dir, "annotations", "manual", f"{split}.corpus.csv")
metadata = []
with open(annotations_path, 'r', encoding='utf-8') as f:
    f.readline()
    for i, line in enumerate(f):
        parts = line.strip().split('|')
        if len(parts) >= 4:
            sample_info = {
                'id': parts[0],
                'video': parts[1],
                'signer': parts[2],
                'annotation': parts[3]
            }
            metadata.append(sample_info)
            if last_video_id and sample_info['id'] == last_video_id:
                start_video_idx = i + 1  # Starte nach dem letzten verarbeiteten Video

print(f"Metadaten für {len(metadata)} Videos geladen. Starte bei Index {start_video_idx} (Video-ID: {metadata[start_video_idx]['id'] if start_video_idx < len(metadata) else 'Ende'}).")

# Schritt 3: Alle Videos iterieren ab Checkpoint
for i, sample_info in enumerate(metadata[start_video_idx:], start=start_video_idx):
    try:
        video_id = sample_info['id']
        signer = sample_info['signer']
        annotation = sample_info['annotation']

        print(f"Verarbeite Sample {i + 1}/{len(metadata)}: signer={signer}, video_id={video_id}")

        video_dir = os.path.join(raw_images_root, split, video_id, '1')
        if not os.path.isdir(video_dir):
            print(f"Warnung: Ordner nicht gefunden: {video_dir}. Überspringe.")
            continue

        image_files = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.png')])
        print(f"\n--- Verarbeite Video {i + 1} ({signer}, {video_id}) mit {len(image_files)} Frames ---")

        video_csv = os.path.join(output_dir, f"{video_id}_landmarks.csv")
        with open(video_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_idx', 'signer', 'annotation'] + [f'coordinate {j}' for j in range(63)])

        for frame_idx, img_file in enumerate(image_files):
            img = cv2.imread(img_file)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    print(f"✅ Hand erkannt in Frame {frame_idx}.")
                    for hand_landmarks in results.multi_hand_landmarks:
                        normalized_landmarks = []
                        for lm in hand_landmarks.landmark:
                            normalized_landmarks.extend([lm.x, lm.y, lm.z])

                        wrist = normalized_landmarks[0:3]
                        for j in range(1, 21):
                            start_idx = j * 3
                            normalized_landmarks[start_idx:start_idx + 3] = [
                                normalized_landmarks[start_idx + k] - wrist[k] for k in range(3)
                            ]

                        with open(video_csv, 'a', newline='') as f:
                            writer = csv.writer(f)
                            row = [frame_idx, signer, annotation] + normalized_landmarks
                            writer.writerow(row)
                else:
                    print(f"❌ Keine Hand erkannt in Frame {frame_idx}.")

        # Speichere Checkpoint nach jedem Video
        with open(checkpoint_file, 'w') as f:
            json.dump({'last_video_id': video_id}, f)

    except Exception as e:
        print(f"Fehler in Sample {i + 1}: {e}. Überspringe.")
        with open(checkpoint_file, 'w') as f:
            json.dump({'last_video_id': video_id}, f)  # Speichere auch bei Fehler
        continue

hands.close()
print(f"\nVerarbeitung abgeschlossen. Fortschritt in {checkpoint_file} gespeichert.")