
import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Locate the existing trained CNN model.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "handwritten_digit_cnn.keras"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"CNN model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# Webcam and drawing settings.
CAMERA_INDEX = 0
BRUSH_SIZE = 12
PREDICTION_DELAY = 0.5

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def preprocess_canvas(canvas):
    """Convert the white digit on a black canvas to MNIST-style input."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    # Find the drawn digit.
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    points = cv2.findNonZero(binary)

    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(points)
    digit = binary[y:y + h, x:x + w]

    # Keep the aspect ratio and fit the digit into a 20x20 area.
    scale = 20.0 / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center the digit on a 28x28 black canvas.
    result = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = digit

    # Match the model's grayscale input format.
    return result.astype("float32") / 255.0


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Close other camera apps and try again."
        )

    canvas = None
    previous_point = None
    last_prediction = None
    confidence = 0.0

    print("Air-writing started!")
    print("Write a digit with your index finger.")
    print("Press C to clear, P to predict, Q to quit.")

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as hands:

            while True:
                success, frame = cap.read()
                if not success:
                    print("Could not read webcam frame.")
                    break

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]

                if canvas is None:
                    canvas = np.zeros_like(frame)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                finger_detected = False

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(
                        frame, hand, mp_hands.HAND_CONNECTIONS
                    )

                    # Index fingertip is landmark 8.
                    tip = hand.landmark[8]
                    x = int(tip.x * width)
                    y = int(tip.y * height)
                    finger_detected = True

                    cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

                    # Draw while the index finger is raised and
                    # the middle finger is folded.
                    index_up = (
                        hand.landmark[8].y < hand.landmark[6].y
                    )
                    middle_folded = (
                        hand.landmark[12].y > hand.landmark[10].y
                    )

                    if index_up and middle_folded:
                        if previous_point is not None:
                            cv2.line(
                                canvas,
                                previous_point,
                                (x, y),
                                (255, 255, 255),
                                BRUSH_SIZE,
                                cv2.LINE_AA,
                            )
                        previous_point = (x, y)
                    else:
                        previous_point = None
                else:
                    previous_point = None

                # Show the live drawing beside the camera feed.
                canvas_view = canvas.copy()
                combined = np.hstack((frame, canvas_view))

                cv2.putText(
                    combined,
                    "AIR WRITING",
                    (width + 20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    combined,
                    "Raise index finger to draw",
                    (width + 20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )

                prediction_text = (
                    f"Prediction: {last_prediction}"
                    if last_prediction is not None
                    else "Prediction: --"
                )
                cv2.putText(
                    combined,
                    prediction_text,
                    (20, height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    combined,
                    f"Confidence: {confidence:.1%}",
                    (20, height - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    combined,
                    "C: Clear   P: Predict   Q: Quit",
                    (width + 20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )

                cv2.imshow("Live Air-Writing Digit Recognition", combined)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("c"):
                    canvas[:] = 0
                    previous_point = None
                    last_prediction = None
                    confidence = 0.0
                elif key == ord("p"):
                    processed = preprocess_canvas(canvas)

                    if processed is None:
                        print("Draw a digit before predicting.")
                        last_prediction = None
                        confidence = 0.0
                    else:
                        probabilities = model.predict(
                            processed.reshape(1, 28, 28, 1),
                            verbose=0,
                        )[0]
                        last_prediction = int(np.argmax(probabilities))
                        confidence = float(np.max(probabilities))

                        print(
                            f"Predicted digit: {last_prediction} | "
                            f"Confidence: {confidence:.2%}"
                        )

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Air-writing application closed.")


if __name__ == "__main__":
    main()
