# Face Recognition System Report

## Overview
The provided Python program implements a **face registration and recognition system** using a webcam. It allows users to register their faces with names and later recognize them in real-time. The system combines **computer vision, machine learning, and GUI technologies** to create an interactive and functional face recognition application.

## Technologies Used
1. **Python** – Main programming language.
2. **OpenCV (`cv2`)** – Handles webcam video capture, image processing, and face recognition using the LBPH (Local Binary Patterns Histograms) algorithm.
3. **MediaPipe (`mediapipe`)** – Provides a fast and accurate **face detection module**.
4. **Tkinter (`tk`)** – Used to create a simple **Graphical User Interface (GUI)** for user interactions.
5. **NumPy (`numpy`)** – Provides array operations to handle image data.
6. **Pickle (`pickle`)** – Stores and loads label data for mapping between user names and numeric labels.
7. **File Handling (`os`)** – Checks if model and label files exist and manages saving/loading.

## Code Workflow and Explanation

### 1. Initialization
- `MediaPipe Face Detection` is initialized to detect faces in webcam frames.
- File paths are defined for:
  - `trained_model.yml` – Stores the trained LBPH face recognizer.
  - `labels.pickle` – Stores the mapping between numeric labels and user names.
- The LBPH recognizer is either loaded (if previously trained) or initialized.

### 2. Face Capture Function (`capture_faces`)
- Captures multiple images of a user’s face from the webcam.
- Converts frames from **BGR to RGB** for MediaPipe.
- Detects faces and crops the bounding box around the face.
- Resizes and converts faces to **grayscale** (required by LBPH).
- Returns a list of preprocessed face images for training.

### 3. Registering a Face (`register_face`)
- Takes a user name from the GUI input.
- Prevents duplicate registrations.
- Captures face samples via `capture_faces`.
- Assigns a **numeric label** to the user.
- Updates or trains the LBPH recognizer with the new face samples.
- Saves the trained model and updated labels to disk.
- Shows a confirmation message on successful registration.

### 4. Recognizing Faces (`recognize_face`)
- Captures video from the webcam in real-time.
- Detects faces using MediaPipe.
- Crops, resizes, and converts detected faces to grayscale.
- Uses LBPH recognizer to predict the face label and confidence.
- Displays the recognized name (or "Unknown") and confidence on the video frame.
- Allows exiting by pressing `'q'`.

### 5. GUI Setup
- Simple Tkinter interface with:
  - An **entry field** for the user name.
  - A **"Register Face"** button to capture and register a new face.
  - A **"Recognize Face"** button to start real-time recognition.
- Uses Tkinter message boxes to provide feedback and instructions to users.

## Key Features
1. **Real-time Face Detection and Recognition** – Uses MediaPipe and LBPH for efficient recognition.
2. **Face Registration** – Supports multiple samples to improve recognition accuracy.
3. **Persistent Storage** – Saves models and label mappings for future sessions.
4. **Interactive GUI** – Easy-to-use interface for non-technical users.

## How It Works (Summary Flow)
1. User enters their name and clicks "Register Face."
2. System captures multiple face images via webcam.
3. Faces are converted to grayscale, labeled, and used to train the LBPH recognizer.
4. Model and label mapping are saved.
5. Later, clicking "Recognize Face" activates the webcam.
6. Faces are detected, recognized, and labeled in real-time.

This system demonstrates a combination of **computer vision**, **machine learning**, and **user interface design**, making it suitable for educational purposes, small access control projects, or prototyping facial recognition applications.

