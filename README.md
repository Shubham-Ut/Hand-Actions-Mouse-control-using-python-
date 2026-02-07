# Hand-Actions-Mouse-control-using-python-
This project is a computer vision–based virtual mouse system that allows users to control their computer cursor using hand gestures captured through a webcam. It uses MediaPipe Hand Landmarker, OpenCV, and PyAutoGUI to detect hand landmarks in real time and convert finger movements into mouse actions. 


🔹 How It Works

The webcam captures live video frames.

MediaPipe detects hand landmarks (like fingertip positions).

The index finger tip controls cursor movement.

Pinch gestures trigger mouse clicks:

Thumb + Index finger → Left click

Thumb + Middle finger → Right click

Smoothing, sensitivity, and dead-zone logic reduce cursor shaking and improve control. 


🔹 Key Features

Real-time hand tracking

Smooth cursor movement using interpolation and smoothing function

Adjustable sensitivity and stability controls

Gesture-based clicking system

Supports single-hand operation

Low-latency webcam processing 

🔹 Technologies Used

Python

OpenCV – video capture & processing

MediaPipe Tasks API – hand landmark detection

PyAutoGUI – system mouse control

NumPy – coordinate mapping and math operations 


🔹 Use Cases

Touchless computer control

Accessibility assistance

AI / Computer Vision learning project

Gesture-based UI experiments
