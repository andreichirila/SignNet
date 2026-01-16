# SignNet - Gebärdensprache-Erkennung mit Deep Learning

Dieses Repository enthält mehrere Module für die Erkennung von Gebärdensprache mittels Computer Vision und Deep Learning. Das Projekt umfasst sowohl statische Erkennung (Fingeralphabet) als auch dynamische Erkennung (Wörter/Sätze in Gebärdensprache).

---

## Projektstruktur

```
SignNet/
├── Wurzelverzeichnis/            # Statische Erkennung (Fingeralphabet)
│   ├── SignNetGUI_static.py      # GUI für Echtzeit-Fingeralphabet-Erkennung
│   ├── HandTrackingOpenCV.py     # MediaPipe-basierte Handdetektion
│   ├── MLP_Model.py              # MLP-Netzwerkarchitektur
│   ├── LandmarkDataset.py        # PyTorch Dataset mit Augmentation
│   ├── correct_samples.py        # Datenbereinigung
│   └── requirements.txt          # Python-Abhängigkeiten
│
├── DynamicSignNet/               # Dynamische Erkennung (Sätze/Wörter)
│   ├── DynamicSignNet.py         # Transformer-CTC-Modell für Sequenzen
│   ├── SignNetWord.py            # Wort-Level Transformer-Klassifikator
│   ├── SignNetInference.py       # Hierarchische Inferenz mit Experten-Modellen
│   ├── SignNetConfig.py          # Modell-Konfigurationen
│   ├── landmark_extraction.py    # Landmark-Extraktion aus Videos
│   └── vocab.json                # Vokabular für CTC-Decoding
│
├── SignNet+/                     # Erweiterte Module
│   ├── MLP_Training.py           # Training-Skript für MLP-Modelle
│   ├── TransformerEncoder/       # GUI und Analyse-Tools
│   │   ├── SignNetGUI.py         # Moderne Echtzeit-GUI (CustomTkinter)
│   │   └── README.md             # Dokumentation für TransformerEncoder
│   ├── landmarks_train/          # Phoenix-2014 Trainings-Landmarks
│   ├── landmarks_dev/            # Validierungs-Landmarks
│   └── landmarks_test/           # Test-Landmarks
│
├── landmark_datasets/            # Daten für Fingeralphabet
│   ├── german_sign_language.csv  # Haupt-Dataset
│   └── german_sign_language_updated.csv
│
└── models/                       # Gespeicherte Modelle (.pth)
```

---

## Module im Detail

### 1. Wurzelverzeichnis - Statische Fingeralphabet-Erkennung

#### `SignNetGUI_static.py`
Hauptanwendung mit Tkinter-GUI für die Echtzeit-Erkennung des deutschen Fingeralphabets (24 Buchstaben: A-Y ohne J und Z).

**Funktionen:**
- Webcam-Erkennung mit Live-Vorschau
- Automatische Buchstaben-Erfassung nach 1 Sekunde Halten
- Wort-Bildung durch Aneinanderreihung erkannter Buchstaben
- Manuelle Aufnahme neuer Trainings-Samples

**Verwendung:**
```bash
python SignNetGUI_static.py
```

#### `HandTrackingOpenCV.py`
Wrapper-Klasse um MediaPipe Hands für konsistente Hand-Landmark-Extraktion.

**Features:**
- 21 3D-Landmarks pro Hand (x, y, z)
- Bounding-Box-Berechnung
- Händigkeit-Erkennung (Links/Rechts)

#### `MLP_Model.py`
Multi-Layer Perceptron mit 5 Schichten für die Klassifikation von 63 normalisierten Landmark-Koordinaten.

**Architektur:**
```
Input (63) → 512 → 256 → 128 → 64 → Output (24 Klassen)
```
Mit BatchNorm, GELU-Aktivierung und Dropout (0.3-0.4).

#### `LandmarkDataset.py`
PyTorch Dataset mit Online-Augmentation:
- Zufälliges Rauschen
- Skalierung
- Rotation um z-Achse
- Z-Koordinaten-Dropout (30% Chance)

#### `correct_samples.py`
Skript zur Bereinigung und Normalisierung der CSV-Daten:
- Wrist-Normalisierung (alle Landmarks relativ zum Handgelenk)
- Filterung ungültiger Labels
- Zusammenführung mehrerer CSV-Dateien

---

### 2. DynamicSignNet - Dynamische Gebärdensprache-Erkennung

#### `DynamicSignNet.py`
Transformer-Encoder-Modell mit CTC-Loss für die Erkennung von Gebärdensprach-Sequenzen.

**Komponenten:**
- `LandmarkDataset`: Lädt vorverarbeitete .npz-Dateien
- `SignLanguageTranslator`: Transformer mit temporaler Konvolution
- `AdvancedTemporalAugmentation`: Speed-Variation, Noise, Frame-Dropout
- Greedy- und Beam-Search-Decoding

**Architektur:**
```
Input (1659 Features) → Linear Projection → Temporal Conv → Positional Encoding
→ Transformer Encoder (6 Layers) → CTC Head → Gloss-Sequenz
```

**Training:**
```bash
cd DynamicSignNet
python DynamicSignNet.py
```

#### `SignNetWord.py`
Wort-Level Klassifikator mit erweitertem Feature Engineering:

**Features:**
- Velocity und Acceleration berechnet aus Landmarks
- Inter-Hand-Distanz
- Hand-zu-Gesicht-Distanz
- Handedness-Embedding

**Hierarchische Klassifikation:**
- Hauptmodell für allgemeine Klassen
- Experten-Modelle für verwechslungsanfällige Klassen (Richtungen, Wetter, etc.)

#### `SignNetInference.py`
Inferenz-Pipeline mit hierarchischer Experten-Architektur:

**Workflow:**
1. Hauptmodell liefert Top-K Vorhersagen
2. Falls Experten-Klasse in Top-K: Experten-Modell wird aktiviert
3. Konfidenz-Schwelle entscheidet über finale Vorhersage

#### `SignNetConfig.py`
Zentrale Konfiguration für alle Modelle:

```python
MAIN_MODEL_CONFIG = {
    'input_size': 1659,
    'hidden_size': 256,
    'num_layers': 4,
    'num_heads': 8,
    'dim_feedforward': 1024
}

HIERARCHY_CONFIG = {
    'direction_expert': ['NORD', 'SUED', 'WEST', 'OST', ...],
    'kommen_expert': ['KOMMEN', 'cl-KOMMEN', ...],
    'weather_expert': ['REGEN', 'SCHNEE', 'WOLKE', ...]
}
```

#### `landmark_extraction.py`
Extraktion von MediaPipe-Landmarks aus Video-Frames:

**Output pro Frame (1659 Features):**
- 2 Hände × 21 Landmarks × 3 (x,y,z) = 126
- 478 Face-Landmarks × 3 = 1434
- 33 Pose-Landmarks × 3 = 99

---

### 3. SignNet+ - Erweiterte Module

#### `MLP_Training.py`
Vollständiges Training-Skript für das statische MLP-Modell:

**Features:**
- MLflow-Integration für Experiment-Tracking
- Automatische Modell-Speicherung mit Metriken im Dateinamen
- Learning-Rate-Scheduler (ReduceLROnPlateau)
- Train/Val/Test Split (70/15/15)

**Verwendung:**
```bash
cd SignNet+
python MLP_Training.py
```

#### `TransformerEncoder/SignNetGUI.py`
Moderne GUI mit CustomTkinter für Echtzeit-Wort-Erkennung:

**Features:**
- Dark Theme mit modernem Design
- Live-Webcam- und Video-Modus
- Thumbnail-Browser für Testvideos
- Side-by-Side Ground-Truth-Vergleich
- Temporal Smoothing und Spam-Unterdrückung

---

### 4. landmark_datasets - Trainingsdaten

| Datei | Beschreibung |
|-------|--------------|
| `german_sign_language.csv` | Haupt-Dataset mit ~200k Samples |
| `german_sign_language_updated.csv` | Aktualisiertes Dataset |
| `new_samples_german_sign_language.csv` | Manuell aufgenommene Samples |

**Format:** 64 Spalten (1 Label + 63 Koordinaten)

---

### 5. models - Gespeicherte Modelle

**Naming-Convention:**
```
{Typ}_trainedModel-{Datum}-eval_loss-{loss}-eval_acc-{acc}-train_time-{time}.pth
```

**Beispiele:**
- `MLP_trainedModel-2025-09-21_18-43-07-eval_loss-0.062-eval_acc-0.989-...pth` (Beste MLP)
- `CNN_trainedModel-2025-09-21_15-46-22-eval_loss-0.027-eval_acc-0.99-...pth` (CNN)

---

## Installation

```bash
# Repository klonen
git clone https://github.com/andreichirila/SignNet.git
cd SignNet

# Virtual Environment erstellen
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Dependencies installieren
pip install -r requirements.txt
```

**Hauptabhängigkeiten:**
- `torch >= 1.10`
- `mediapipe >= 0.10.21`
- `opencv-python >= 4.5`
- `numpy < 2`
- `customtkinter` (für moderne GUI)
- `mlflow` (für Experiment-Tracking)

---

## Schnellstart

### Fingeralphabet-Erkennung starten
```bash
python SignNetGUI_static.py
```

### MLP-Modell trainieren
```bash
cd SignNet+
python MLP_Training.py
```

### Dynamische Erkennung (Wörter)
```bash
cd DynamicSignNet
python DynamicSignNet.py  # Training
```

### Moderne GUI für Wort-Erkennung
```bash
cd SignNet+/TransformerEncoder
python SignNetGUI.py
```

---

## Datenformate

### Statische Erkennung (CSV)
```
label, coordinate 0, coordinate 1, ..., coordinate 62
a, 0.5, 0.3, 0.1, ...
b, 0.4, 0.5, 0.2, ...
```

### Dynamische Erkennung (NPZ)
```python
data = np.load("sample.npz")
data['landmarks']  # Shape: (T, 1659) - Frames × Features
data['glosses']    # Array von Glossen/Wörtern
```

---

## MLflow-Integration

Das Projekt verwendet MLflow für Experiment-Tracking. Konfiguration in den Training-Skripten:

```python
mlflow.set_tracking_uri("https://mlflow.schlaepfer.me")
mlflow.set_experiment("Static Sign Net")  # oder "SignNetAdvanced++"
```

**Geloggte Metriken:**
- Train/Val Loss und Accuracy
- Learning Rate
- Modell-Architektur-Parameter
- System-Informationen (GPU, RAM, etc.)

---

## Autoren

- Roman Schläpfer
- Andrei Chirila

---

## Lizenz

MIT License
