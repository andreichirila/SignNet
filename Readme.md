# SignNet 🤟
**Real-Time Sign Language Translation using Computer Vision & Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

> *Bridging communication barriers through AI-powered sign language recognition*

## 🎯 Übersicht

SignNet ist ein End-to-End Machine Learning System, das Gebärdensprache in Echtzeit in natürliche Sprache übersetzt. Das Projekt kombiniert modernste Computer Vision Techniken mit Natural Language Processing, um eine nahtlose Kommunikationsbrücke zu schaffen.

### ✨ Key Features

- **Echtzeit-Erkennung**: Sub-100ms Latenz für flüssige Kommunikation
- **Multi-Language Support**: Deutsche und Schweizer Gebärdensprache (DSGS)
- **Robuste Architektur**: Funktioniert bei verschiedenen Lichtverhältnissen und Handhaltungen
- **Edge-Ready**: Optimiert für lokale Inferenz ohne Cloud-Abhängigkeit

## 🏗️ Architektur

Video Input → OpenCV 
Hand Detection → MediaPipe
Pose Estimation → 3D Landmarks
Feature Extraction → CNN Features
Sequence Modeling → LSTM/Transformer
Text Output → NLP Post-processing
                         
### Model Pipeline

1. **Computer Vision Frontend**
   - MediaPipe für Hand Landmark Detection
   - 3D Pose Estimation (21 Keypoints pro Hand)
   - Temporal Smoothing & Noise Reduction

2. **Feature Engineering**
   - Landmark Normalisierung (Translation & Scale Invariant)
   - Velocities & Accelerations berechnen
   - Hand Shape & Motion Features

3. **Sequence Modeling**
   - Bidirectional LSTM für temporale Abhängigkeiten
   - Attention Mechanism für wichtige Gesten
   - CTC Loss für variable Sequenzlängen

4. **Language Processing**
   - Beam Search Decoding
   - Grammatik-basierte Post-Processing
   - Confidence Scoring

## 🚀 Quick Start

### Installation

```bash
# Repository klonen
git clone https://github.com/andreichirila/SignNet.git
cd SignNet

# Virtual Environment erstellen (// später)
python -m venv signnet_env
source signnet_env/bin/activate  # Linux/Mac
# signnet_env\Scripts\activate   # Windows

# Dependencies installieren (// später)
pip install -r requirements.txt