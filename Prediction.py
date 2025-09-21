import cv2
import torch
import torchvision.transforms as transforms
from ASLR_CNN_Model import SignLanguageCNN  # Importiere die modularisierte CNN-Klasse
from HandTrackingOpenCV import HandDetector
import os
import glob
import time  # Für FPS-Berechnung

# Config – warum? Zentrale Parameter für Wartbarkeit; passe num_classes an dein Dataset an
CONFIG = {
    'img_size': 64,  # Modell-Input-Größe (muss zum Training passen)
    'num_classes': 25,  # Anzahl Klassen (z.B. 26 für A-Z oder 30 für DGS mit Umlauten)
    'class_names': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'sch', 't',
               'u', 'v', 'w', 'x', 'y'],
    # DGS-Beispiel; lade dynamisch, wenn möglich
    'confidence_threshold': 0.7,  # Mindest-Confidence für Prediction-Anzeige
    'frame_skip': 2,  # Reduziert Latency durch Frame-Skipping
    'model_dir': './models'  # Pfad zu gespeicherten Modellen
}


def load_latest_model(model_dir, num_classes, device):
    """
    Lädt das neueste trainierte Modell – warum? Automatisiert für Entwicklung; priorisiert ctime für Neuheit.
    :param model_dir: Verzeichnis mit .pth-Dateien
    :param num_classes: Für Instanziierung der Architektur
    :param device: GPU/CPU
    :return: Geladenes Modell
    """
    model_files = glob.glob(os.path.join(model_dir, "*.pth"))
    if not model_files:
        raise ValueError(f"Kein trainiertes Modell in {model_dir} gefunden! Trainiere zuerst.")

    latest_model = max(model_files, key=os.path.getctime)
    print(f"Lade neuestes Modell: {latest_model}")

    model = SignLanguageCNN(num_classes).to(device)
    model.load_state_dict(torch.load(latest_model, map_location=device))
    model.eval()  # Eval-Modus: Deaktiviert Dropout und BN-Updates für konsistente Predictions
    return model


def preprocess_hand_image(img_hand, img_size):
    """
    Preprocessing für Hand-Bild – warum? Muss exakt zum Training-Transform matchen (RGB, Resize, Normalize).
    :param img_hand: Cropped BGR-Bild von OpenCV
    :param img_size: Zielgröße (z.B. 64x64)
    :return: Normalisierter Tensor (1, 3, H, W)
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Konvertiert CV2 BGR zu PIL RGB (wichtig für torchvision!)
        transforms.Resize((img_size, img_size)),  # Skaliert auf Modell-Input
        transforms.ToTensor(),  # Zu [0,1]-Tensor (C, H, W)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalisiert zu [-1,1] wie im Training
    ])

    input_tensor = transform(img_hand).unsqueeze(0)  # Fügt Batch-Dimension hinzu (1, C, H, W)
    return input_tensor


def predict_letter(model, input_tensor, class_names, threshold):
    """
    Führt Prediction aus und gibt Buchstabe + Confidence zurück – warum? Trennt Logik für Testbarkeit.
    :param model: Geladenes CNN
    :param input_tensor: Preprocessed Input
    :param class_names: Liste der Klassen
    :param threshold: Confidence-Cutoff
    :return: Tuple (letter, confidence) oder ("Unknown", 0.0)
    """
    with torch.no_grad():  # Spart Speicher/Gradienten während Inferenz
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)  # Konvertiert Logits zu Wahrscheinlichkeiten
        conf, predicted = torch.max(probs, 1)
        confidence = conf.item()

        if confidence > threshold:
            letter = class_names[predicted.item()]
        else:
            letter = "Unknown"
            confidence = 0.0

    return letter, confidence


def main():
    # Initialisiere Device – warum? GPU für Speed, Fallback zu CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Verwende Device: {device}")

    # Lade Modell
    model = load_latest_model(CONFIG['model_dir'], CONFIG['num_classes'], device)

    # HandDetector – warum? MediaPipe für robuste Echtzeit-Handerkennung
    detector = HandDetector(detectionCon=0.8, maxHands=1)  # Hohe Confidence, nur eine Hand

    # Webcam-Setup – warum? Hohe Res für Detection, aber frame_skip für <30ms Latency
    capture = cv2.VideoCapture(0)  # cam_id=0
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    frame_counter = 0
    fps_start = time.time()
    fps_counter = 0

    try:
        while True:
            ret, img = capture.read()
            if not ret:
                print("Kamera-Fehler – beende.")
                break

            frame_counter += 1
            if frame_counter % CONFIG['frame_skip'] == 0:
                # Hand Detection
                hands, img_annotated, bbox = detector.findHands(img, draw=True)  # Zeichnet Bounding Box

                if bbox:  # Hand erkannt
                    # Crop Hand-Region – warum? Isoliert die Hand für bessere Feature-Extraktion
                    h, w, _ = img.shape
                    size = max(bbox[2], bbox[3]) + 140  # Buffer gegen Amputation
                    cx, cy = int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)
                    x1, y1 = max(int(cx - size / 2), 0), max(int(cy - size / 2), 0)
                    x2, y2 = min(int(cx + size / 2), w), min(int(cy + size / 2), h)
                    img_hand = img[y1:y2, x1:x2]  # BGR-Crop behalten

                    # Preprocess & Predict
                    input_tensor = preprocess_hand_image(img_hand, CONFIG['img_size']).to(device)
                    letter, confidence = predict_letter(model, input_tensor, CONFIG['class_names'],
                                                        CONFIG['confidence_threshold'])

                    # Annotiere – warum? Visuelles Feedback mit Farbe (Grün=OK, Rot=Unknown)
                    color = (0, 255, 0) if confidence > 0 else (0, 0, 255)
                    text = f"{letter}: {confidence:.2f}" if confidence > 0 else "Unknown"
                    cv2.putText(img_annotated, text, (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # FPS-Tracking – warum? Monitort Performance (sollte >15 FPS für Echtzeit)
                fps_counter += 1
                if fps_counter % 30 == 0:  # Alle 30 Frames
                    fps = 30 / (time.time() - fps_start)  # Geschätzte FPS
                    print(f"FPS: {fps:.1f}")
                    fps_start = time.time()

                # Zeige Frame
                cv2.imshow("Live Sign Language Prediction", img_annotated)

            # ESC zum Beenden
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("\nUnterbrochen durch User (Ctrl+C).")
    finally:
        capture.release()
        cv2.destroyAllWindows()
        print("Ressourcen freigegeben. Exiting.")


if __name__ == "__main__":
    main()