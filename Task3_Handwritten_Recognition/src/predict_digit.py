import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "handwritten_digit_cnn.keras"


def preprocess_image(image_path):
    # Convert the image to grayscale
    image = Image.open(image_path).convert("L")

    # Convert black digits on white backgrounds to white on black
    image = ImageOps.invert(image)

    # Find the bounding box around the digit
    image = image.point(lambda pixel: 255 if pixel > 30 else 0)
    bbox = image.getbbox()

    if bbox is None:
        raise ValueError("No visible digit found in the image.")

    # Crop the digit and preserve its proportions
    image = image.crop(bbox)
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Place it in the center of a 28x28 black canvas
    canvas = Image.new("L", (28, 28), 0)
    x_offset = (28 - image.width) // 2
    y_offset = (28 - image.height) // 2
    canvas.paste(image, (x_offset, y_offset))

    # Normalize pixels and add batch/channel dimensions
    image_array = np.array(canvas, dtype="float32") / 255.0
    return image_array.reshape(1, 28, 28, 1)


def predict_digit(image_path):
    if not MODEL_PATH.exists():
        print("Model not found. Please train the model first.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    image_array = preprocess_image(image_path)

    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit]) * 100

    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {confidence:.2f}%")

    # Show the three most likely digits
    top_digits = np.argsort(probabilities)[-3:][::-1]
    print("Top predictions:")
    for digit in top_digits:
        print(f"  {digit}: {probabilities[digit] * 100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_digit.py path_to_image.png")
    else:
        predict_digit(sys.argv[1])