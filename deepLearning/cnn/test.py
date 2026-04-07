# test.py

import torch
import cv2
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from main import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# LOAD MODEL
# ==============================
model = CNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

# ==============================
# TRANSFORM
# ==============================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ==============================
# EXTRACT DIGITS (FINAL FIX)
# ==============================
def extract_digits(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 🔥 Remove texture noise
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # 🔥 Otsu threshold (best for your case)
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 🔥 Morphology (clean image)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Debug (optional)
    # plt.imshow(thresh, cmap="gray")
    # plt.title("Threshold")
    # plt.show()

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    digit_images = []
    image_area = image.shape[0] * image.shape[1]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # STRONG FILTERING
        if area < 0.01 * image_area:
            continue

        if h < 40 or w < 20:
            continue

        digit = thresh[y:y+h, x:x+w]

        # Make square
        size = max(w, h)
        square = np.zeros((size, size), dtype=np.uint8)

        x_offset = (size - w) // 2
        y_offset = (size - h) // 2

        square[y_offset:y_offset+h, x_offset:x_offset+w] = digit

        digit = cv2.resize(square, (28, 28))

        digit_images.append((x, digit, area))

    # Keep only biggest digits
    digit_images = sorted(digit_images, key=lambda x: x[2], reverse=True)[:5]

    # Sort left → right
    digit_images = sorted(digit_images, key=lambda x: x[0])

    return [d[1] for d in digit_images]

# ==============================
# PREDICT
# ==============================
def predict(digits):
    results = []

    for d in digits:
        img = Image.fromarray(d)
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            _, pred = torch.max(output, 1)

        results.append(str(pred.item()))

    return "".join(results)

# ==============================
# MAIN
# ==============================
def main():
    folder = "image"

    if not os.path.exists(folder):
        print("image folder not found")
        return

    files = os.listdir(folder)

    for file in files:
        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        path = os.path.join(folder, file)

        print(f"\nProcessing: {file}")

        image = cv2.imread(path)

        if image is None:
            print("Cannot read image")
            continue

        digits = extract_digits(image)

        if not digits:
            print("No digits detected")
            continue

        result = predict(digits)

        print(f"Prediction: {result}")

        # Show extracted digits
        plt.figure(figsize=(8, 2))
        for i, d in enumerate(digits):
            plt.subplot(1, len(digits), i + 1)
            plt.imshow(d, cmap="gray")
            plt.axis("off")

        plt.suptitle(result)
        plt.show()


if __name__ == "__main__":
    main()
