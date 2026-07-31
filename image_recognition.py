import os
import time
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.utils import load_img, img_to_array

# Load pretrained model
model = MobileNetV2(weights="imagenet")

# Sample image URL
FALLBACK_IMAGE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg"
FALLBACK_IMAGE_PATH = "sample_image.jpg"

print("=" * 50)
print("IMAGE RECOGNITION USING MOBILE NET V2")
print("=" * 50)

# Ask user for image path
image_path = input(
    "Enter image path (or press Enter to use sample image): "
).strip()

# Use sample image if no path provided
if image_path == "":
    if not os.path.exists(FALLBACK_IMAGE_PATH):
        print("Downloading sample image...")
        urllib.request.urlretrieve(FALLBACK_IMAGE_URL, FALLBACK_IMAGE_PATH)
    image_path = FALLBACK_IMAGE_PATH

# Check if image exists
if not os.path.exists(image_path):
    print("Error: Image file not found!")
    exit()

# Load image
image = load_img(image_path, target_size=(224, 224))

# Display image
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.axis("off")
plt.title("Input Image")
plt.show()

# Preprocess image
image_array = img_to_array(image)
image_batch = np.expand_dims(image_array, axis=0)
processed_image = preprocess_input(image_batch)

# Predict
start = time.time()
predictions = model.predict(processed_image, verbose=0)
elapsed = (time.time() - start) * 1000

# Decode predictions
results = decode_predictions(predictions, top=5)[0]

print("\nPrediction Time: {:.2f} ms".format(elapsed))
print("\nTop 5 Predictions:\n")

for i, (_, label, probability) in enumerate(results, start=1):
    print(f"{i}. {label:<20} {probability * 100:.2f}%")