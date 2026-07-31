# 🖼️ Image Recognition using MobileNetV2

An AI-powered image recognition web application built with **TensorFlow**, **MobileNetV2**, and **Streamlit**. The application allows users to upload an image and predicts the object present in the image by displaying the **Top 5 predictions** along with their confidence scores.

---

## 🚀 Features

- Upload JPG, JPEG, or PNG images
- Real-time image classification
- Uses the pre-trained MobileNetV2 deep learning model
- Displays the uploaded image
- Shows Top 5 predictions with confidence percentages
- Interactive confidence bar chart
- User-friendly Streamlit interface

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- MobileNetV2
- NumPy
- Pillow
- Matplotlib
- Streamlit

---

## 📂 Project Structure

```
Image-Recognition-Project/
│
├── app.py                  # Streamlit web application
├── image_recognition.py    # Terminal-based image recognition script
├── requirements.txt        # Required Python libraries
├── README.md               # Project documentation
├── .gitignore              # Git ignored files
└── sample_image.jpg        # Optional sample image
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Image-Recognition-Project.git
```

### 2. Open the project folder

```bash
cd Image-Recognition-Project
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

---

## 🖥️ How It Works

1. Upload an image.
2. The image is resized to **224 × 224 pixels**.
3. The image is preprocessed for MobileNetV2.
4. The pre-trained MobileNetV2 model predicts the image class.
5. The application displays the **Top 5 predictions** with confidence scores.

---

## 📊 Model

- **Model:** MobileNetV2
- **Framework:** TensorFlow/Keras
- **Dataset Used for Training:** ImageNet
- **Number of Classes:** 1000

---

## 📸 Screenshot

Add a screenshot of your application here after deployment.

Example:

```
screenshots/app.png
```

---

## 🎯 Future Improvements

- Support drag-and-drop image upload
- Webcam image recognition
- Live video object detection
- Display object descriptions
- Prediction history
- Download prediction results

---

## 👨‍💻 Author

**Harsh**



## 📄 License

This project is licensed under the MIT License.

---

⭐ If you found this project useful, consider giving it a star on GitHub!