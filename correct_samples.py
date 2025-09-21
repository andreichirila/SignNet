import pandas as pd
import numpy as np

# Konfiguration
input_csv_1 = 'landmark_datasets/new_samples_german_sign_language.csv'
input_csv_2 = 'landmark_datasets/german_sign_language.csv'

output_csv = 'landmark_datasets/new_samples_german_sign_language_corrected.csv'
valid_labels = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm',
                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y'}

# Lade die CSVs, konkatenieren und filtere ungültige Labels
df1 = pd.read_csv(input_csv_1)
df2 = pd.read_csv(input_csv_2)
df = pd.concat([df1, df2], ignore_index=True)
df = df[df['label'].isin(valid_labels)].dropna()


# Extrahiere Feature-Spalten (63 Koordinaten)
feature_cols = [f'coordinate {i}' for i in range(63)]
X = df[feature_cols].values.astype(np.float32)

# Wrist-Normalisierung: Subtrahiere Wrist-Koordinaten (0:3) von allen Landmarks
for i in range(1, 21):  # Für die 20 anderen Landmarks
    start_idx = i * 3
    X[:, start_idx:start_idx + 3] -= X[:, 0:3]

# Setze Wrist-Koordinaten auf 0
X[:, 0:3] = 0

# Erstelle neuen DataFrame
df_corrected = pd.DataFrame(X, columns=feature_cols)
df_corrected['label'] = df['label'].values
df_corrected = df_corrected[['label'] + feature_cols]  # Reihenfolge: label, dann Features

# Sortiere nach Label
df_corrected = df_corrected.sort_values(by='label')

# Speichere korrigierte CSV
df_corrected.to_csv(output_csv, index=False)
print(f"Korrigierte Samples gespeichert in: {output_csv}")
print("Durchschnittliche Wrist-Koordinaten (sollten ~0 sein):", df_corrected.iloc[:, 1:4].mean().to_list())

# Zusätzliche Validierung: Prüfe Label-Verteilung
label_dist = df_corrected['label'].value_counts().to_dict()
print("Label-Verteilung nach Korrektur:\n", label_dist)
