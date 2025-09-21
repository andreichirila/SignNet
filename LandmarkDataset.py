import numpy as np
import torch
from torch.utils.data import Dataset

class LandmarkDataset(Dataset):
    def __init__(self, features, labels, augment=True, noise_std=0.01, scale_range=(0.9, 1.1)):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment  # Toggle Augmentation
        self.noise_std = noise_std  # Standardabweichung für Noise
        self.scale_range = scale_range  # Skalierungsbereich

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.features[idx].clone()  # Kopiere, um Original nicht zu ändern
        label = self.labels[idx]

        if self.augment and self.training:  # Nur im Training augmentieren
            # 1. Zufälliges Rauschen
            noise = torch.normal(mean=0.0, std=self.noise_std, size=features.shape)
            features += noise

            # 2. Zufällige Skalierung
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            wrist = features[0:3]  # Wrist-Koordinaten
            for i in range(1, 21):  # Skaliere relativ zu Wrist
                start_idx = i * 3
                features[start_idx:start_idx + 3] = wrist + (features[start_idx:start_idx + 3] - wrist) * scale

            # Optional: Z-Dropout für 2D/3D-Konsistenz
            if np.random.rand() < 0.3:  # 30% Chance
                for i in range(21):
                    features[i * 3 + 2] = 0.0  # Setze z-Koordinate auf 0

        return features, label

    def set_training(self, training):
        self.training = training  # Für model.train()/model.eval()