import numpy as np

# Quick check:

sample = np.load('../landmarks_train_cleaned/01April_2010_Thursday_heute_default-0.npz')
print("New shape:", sample['landmarks'].shape)
# Should be: [T, 1086] if keep_z=False (543 × 2)
# Or:        [T, 1629] if keep_z=True  (543 × 3)