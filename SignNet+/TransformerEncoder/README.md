# SignNet - Datei-Übersicht und Verwendung

## 📁 Dateistruktur

```
TransformerEncoder/
├── SignNetConfig.py          # Konfiguration (immer benötigt)
├── sign_classifier_word.py   # 🔴 TRAINING (das große Script)
├── SignNetAnalysis.py        # Analyse aus MLflow (optional)
└── evaluate_confusion_matrix.py  # Evaluation mit GPU (optional)
```

---

## 1️⃣ SignNetConfig.py
**Zweck:** Zentrale Konfiguration für alle anderen Scripts

**Wann benutzen:** Wird automatisch importiert, nicht direkt ausführen

**Enthält:**
- `MAIN_MODEL_CONFIG` - Architektur des Hauptmodels (512h, 6L)
- `EXPERT_MODEL_CONFIG` - Architektur für Expert-Models (64h, 2L)
- `HIERARCHY_CONFIG` - Welche Klassen zu welchem Expert gehören
- `OVERSAMPLE_CONFIG` - Oversampling-Faktoren für schwache Klassen
- `SAMPLE_COUNT_THRESHOLDS` - Definition low/mid/high

---

## 2️⃣ sign_classifier_word.py (TRAINING)
**Zweck:** Model trainieren

### Hauptmodel trainieren:
```bash
python sign_classifier_word.py \
    --data-dir ./word_landmarks_extracted \
    --dataset-type flat
```

### Expert-Model trainieren (z.B. direction_expert):
```bash
python sign_classifier_word.py \
    --data-dir ./word_landmarks_extracted \
    --dataset-type flat \
    --expert-name direction_expert
```

### Outputs:
- `./models_balanced/sign_classifier_best_enhanced.pth` - Bestes Model
- `./models_balanced/sign_classifier_swa_enhanced.pth` - SWA Model
- `main_vocab.json` oder `direction_expert_vocab.json` - Vokabular
- `val_indices.npy` - Validation-Indizes für Reproduzierbarkeit

---

## 3️⃣ evaluate_confusion_matrix.py (EVALUATION)
**Zweck:** Nach dem Training - Confusion Matrix und Fehleranalyse

**Voraussetzungen:**
- Trainiertes Model (.pth)
- Vocabulary JSON
- GPU empfohlen

### Verwendung:
```bash
python evaluate_confusion_matrix.py \
--model-path ./models_balanced/sign_classifier_best_enhanced.pth \
--data-dir ../Data/word_landmarks_extracted \
--vocab-path ./models_balanced/main_vocab.json \
--val-indices-path ./val_indices.npy
```

### Outputs (in `./evaluation_results/`):
- `confusion_top40_*.png` - Confusion Matrix der 40 schlechtesten Klassen
- `confusion_full_*.png` - Vollständige Confusion Matrix
- `stratum_analysis_*.png` - Accuracy nach Sample-Count (low/mid/high)
- `evaluation_results_*.json` - Alle Metriken

---

## 4️⃣ SignNetAnalysis.py (MLFLOW ANALYSE)
**Zweck:** Metriken aus MLflow laden und analysieren (OHNE Model neu zu laden)

**Voraussetzungen:**
- MLflow Zugang
- Abgeschlossener Training-Run

### Letzte Run analysieren:
```bash
python SignNetAnalysis.py --latest --data-dir ./word_landmarks_extracted
```

### Spezifische Run analysieren:
```bash
python SignNetAnalysis.py --run-id abc123xyz --data-dir ./word_landmarks_extracted
```

---

## 🚀 Typischer Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. TRAINING                                                │
│     python sign_classifier_word.py --data-dir ./data        │
│                                                             │
│     → Speichert Model + Vocab + Metriken in MLflow          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  2. ANALYSE (wähle eine Option)                             │
│                                                             │
│  Option A: Schnell (nur MLflow Metriken)                    │
│     python SignNetAnalysis.py --latest                      │
│                                                             │
│  Option B: Detailliert (Confusion Matrix)                   │
│     python evaluate_confusion_matrix.py \                   │
│         --model-path ./models/best.pth \                    │
│         --vocab-path ./main_vocab.json                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  3. EXPERT TRAINING (falls Confusion Matrix Cluster zeigt) │
│                                                             │
│     python sign_classifier_word.py \                        │
│         --expert-name direction_expert                      │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚠️ Wichtige Hinweise

1. **SignNetConfig.py** muss im GLEICHEN Ordner sein wie die anderen Scripts
2. **MLflow Credentials** sind in den Scripts hardcoded - ändere sie bei Bedarf
3. **GPU** ist nur für Training und evaluate_confusion_matrix.py nötig
4. **val_indices.npy** wird beim Training gespeichert - für konsistente Evaluation aufheben!