# Age and Gender Detection using OpenCV and Caffe

This project demonstrates real-time age and gender detection using pre-trained Caffe models for face, age, and gender detection. The application captures live video, detects faces, and predicts the age and gender of the detected faces. The results are displayed in real-time on the video feed.


## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Models Used](#models-used)
- [Key Features](#key-features)
- [Smoothing Mechanism](#smoothing-mechanism)
- [Acknowledgements](#acknowledgements)

---

## Requirements

Make sure you have the following libraries installed:

- Python 3.x
- OpenCV 4.10.0 or later
- NumPy

You can install these packages using `pip`:

pip install opencv-python numpy


## Installation

1. Clone the repository:

```bash
git clone https://github.com/muhannad-khaled/age-gender-detection.git
```

2. Download the following Caffe pre-trained models:

   - `opencv_face_detector.pbtxt`
   - `opencv_face_detector_uint8.pb`
   - `age_deploy.prototxt`
   - `age_net.caffemodel`
   - `gender_deploy.prototxt`
   - `gender_net.caffemodel`

3. Place the downloaded models in the project directory.


## How to Run

1. Open a terminal in the project directory.
2. Run the Python script:

python main.py

3. The webcam will open, and you can see the real-time age and gender predictions on the video stream.
4. Press **`q`** to exit the video stream.


## Project Structure


├── main.py                  # Main Python script for age-gender detection
├── opencv_face_detector.pbtxt  # Face detection model configuration
├── opencv_face_detector_uint8.pb  # Face detection model
├── age_deploy.prototxt      # Age detection model configuration
├── age_net.caffemodel       # Age detection model
├── gender_deploy.prototxt   # Gender detection model configuration
├── gender_net.caffemodel    # Gender detection model



## Models Used

- Face Detection Model: Based on the OpenCV pre-trained face detection model.
- Age Detection Model: Pre-trained on age data and capable of detecting age ranges from 0 to 100.
- Gender Detection Model: Pre-trained on gender data to classify gender into "Male" and "Female".


## Key Features

1. Real-Time Detection: Detects age and gender from a live video stream.
2. Smoothing Mechanism: Stabilizes the age predictions over multiple frames to reduce fluctuations.
3. High Confidence Predictions: Applies confidence thresholds for more reliable gender classification.


## Smoothing Mechanism

To stabilize noisy age predictions, the project implements a smoothing mechanism using a queue. The age predictions are averaged over the last 5 frames, and the most frequent prediction is displayed. This reduces rapid changes in the predicted age values.


## Acknowledgements

- Pre-trained models used in this project are from OpenCV and the Caffe framework.
- Special thanks to the creators of OpenCV and Caffe for their contributions to the open-source community.

---

## License

This project is licensed under the MIT License.
