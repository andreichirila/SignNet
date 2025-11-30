import numpy as np
from pathlib import Path
from tqdm import tqdm

def clean_landmarks(input_dir, output_dir, keep_z=False):
    """
    Remove custom landmarks and optionally Z dimension.

    Args:
        keep_z: If False, only keep X,Y
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    npz_files = list(input_path.glob("*.npz"))

    for npz_file in tqdm(npz_files, desc=f"Cleaning {input_dir}"):
        sample = np.load(npz_file, allow_pickle=True)
        landmarks = sample['landmarks']  # [T, 1659]
        glosses = sample['glosses']

        T = landmarks.shape[0]

        # Reshape to [T, 553, 3]
        landmarks_reshaped = landmarks.reshape(T, 553, 3)

        # Remove custom landmarks (keep only 0-542)
        landmarks_clean = landmarks_reshaped[:, :543, :]  # [T, 543, 3]

        if not keep_z:
            # Keep only X,Y
            landmarks_clean = landmarks_clean[:, :, :2]  # [T, 543, 2]

        # Flatten back
        landmarks_final = landmarks_clean.reshape(T, -1)

        # Save
        output_file = output_path / npz_file.name
        np.savez(output_file, landmarks=landmarks_final, glosses=glosses)

    print(f"\n✓ Cleaned {len(npz_files)} files")
    print(f"  Original shape: [T, 1659]")
    print(f"  New shape: [T, {landmarks_final.shape[1]}]")


# RUN
for split in ['landmarks_train', 'landmarks_dev', 'landmarks_test']:
    clean_landmarks(
        input_dir=f'D:/OST/SignNet/SignNet+/{split}',
        output_dir=f'D:/OST/SignNet/SignNet+/{split}_cleaned',
        keep_z=False  # START ohne Z
    )