import cv2
from HandTrackingOpenCV import HandDetector
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)

cam_id = 0
capture = cv2.VideoCapture(cam_id)

# Reduce image resolution to reduce hand detection latency
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

detector = HandDetector(detectionCon=0.8, maxHands=1)

cv2.namedWindow("image")

frame_skip = 2  # Process every 2nd frame
frame_counter = 0

try:
    while True:
        ret, img = capture.read()

        if ret:
            frame_counter += 1

            # Process only every "frame_skip" frames
            if frame_counter % frame_skip == 0:
                try:
                    hands, img_hand, bbox = detector.findHands(img, False)

                    if bbox:
                        height, width, _ = img_hand.shape
                        size = max(bbox[2], bbox[3])  # Use the larger of width/height
                        size += 140  # Add buffer for margins
                        center_x = int(bbox[0] + bbox[2] / 2)
                        center_y = int(bbox[1] + bbox[3] / 2)

                        # Compute bounding box coordinates
                        x1 = int(center_x - size / 2)
                        y1 = int(center_y - size / 2)
                        x2 = int(center_x + size / 2)
                        y2 = int(center_y + size / 2)

                        # Ensure the square box remains within image dimensions
                        x1 = max(x1, 0)
                        y1 = max(y1, 0)
                        x2 = min(x2, width)
                        y2 = min(y2, height)

                        # Crop the image to the square bounding box
                        img_hand = img_hand[y1:y2, x1:x2]

                        # Extract red color channel (to focus on hand region in preprocessing).
                        gray = img_hand[:, :, 2]

                        # Apply binary threshold using cv2.THRESH_OTSU
                        _, hand_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                        # "Opening" to clean noise
                        hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_OPEN,
                                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

                        # "Closing" to close gaps
                        hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_CLOSE,
                                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

                        # Apply mask and convert to grayscale
                        img_hand = cv2.bitwise_and(img_hand, img_hand, mask=hand_mask)
                        img_hand = cv2.cvtColor(img_hand, cv2.COLOR_BGR2GRAY)
                        img_hand = cv2.resize(img_hand, (500, 500))

                        # Display processed hand image
                        cv2.imshow("image", img_hand)

                    else:
                        # Display original if no hands are detected
                        cv2.imshow("image", img)

                except Exception as e:
                    print(f"Error during hand detection: {e}")
                    cv2.imshow("image", img)

            # Exit window when ESC key is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
        else:
            break
except KeyboardInterrupt:
    print("\nProgram interrupted by user (Ctrl+C).")
finally:
    # Release resources and close all OpenCV windows
    if capture.isOpened():
        capture.release()
    cv2.destroyAllWindows()
    print("Resources released. Exiting.")