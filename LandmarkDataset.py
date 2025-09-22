import numpy as np
import torch
from torch.utils.data import Dataset

class LandmarkDataset(Dataset):
    def __init__(self, features, labels, augment=True, noise_std=0.01, scale_range=(0.9, 1.1), rotation_range=(-10, 10)):
        """
        Initialisiert das Dataset für Handlandmarks.
        :param features: Input-Features (Handlandmark-Koordinaten).
        :param labels: Ziel-Labels (z.B. Gebärdensprache-Buchstaben).
        :param augment: Ob Daten-Augmentation angewendet werden soll.
        :param noise_std: Standardabweichung für zufälliges Rauschen.
        :param scale_range: Bereich für zufällige Skalierung (min, max).
        :param rotation_range: Bereich für zufällige Rotation in Grad (min, max).
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.rotation_range = rotation_range  # Neuer Parameter für Rotationsbereich

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
            wrist = features[0:3]  # Wrist-Koordinaten (x, y, z)
            for i in range(1, 21):  # Skaliere relativ zu Wrist
                start_idx = i * 3
                features[start_idx:start_idx + 3] = wrist + (features[start_idx:start_idx + 3] - wrist) * scale

            # 3. Zufällige Rotation um die z-Achse
            theta = np.random.uniform(self.rotation_range[0], self.rotation_range[1])  # Zufälliger Winkel in Grad
            theta_rad = np.deg2rad(theta)  # Umrechnung in Radiant
            # 2x2 Rotationsmatrix für x,y (z bleibt unberührt)
            rot_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                                  [np.sin(theta_rad), np.cos(theta_rad)]])
            for i in range(21):  # Rotiere alle Landmarks
                start_idx = i * 3
                xy = features[start_idx:start_idx + 2] - wrist[:2]  # Subtrahiere Wrist (x,y)
                rotated_xy = np.dot(rot_matrix, xy.numpy())  # Rotiere (x,y)
                features[start_idx:start_idx + 2] = (wrist[:2] + rotated_xy).detach().clone().to(dtype=torch.float32)

            # 4. Z-Dropout für 2D/3D-Konsistenz
            if np.random.rand() < 0.3:  # 30% Chance
                for i in range(21):
                    features[i * 3 + 2] = 0.0  # Setze z-Koordinate auf 0

        return features, label

    def set_training(self, training):
        self.training = training  # Für model.train()/model.eval()