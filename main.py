import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox
import numpy as np
import os
import pickle

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Paths for saving model and labels
MODEL_PATH = 'trained_model.yml'
LABELS_PATH = 'labels.pickle'

# Load or initialize recognizer and labels
if os.path.exists(MODEL_PATH):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
else:
    recognizer = cv2.face.LBPHFaceRecognizer_create()

if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, 'rb') as f:
        label_to_name = pickle.load(f)
        name_to_label = {v: k for k, v in label_to_name.items()}
        next_label = max(label_to_name.keys()) + 1
else:
    label_to_name = {}
    name_to_label = {}
    next_label = 0

def capture_faces(name, num_samples=100):
    cap = cv2.VideoCapture(0)
    faces = []
    count = 0

    messagebox.showinfo("Capture Instructions", f"Please turn your face in different angles. Capturing {num_samples} samples...")

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            # Crop face
            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue

            # Resize to fixed size and convert to gray
            face_img = cv2.resize(face_img, (200, 200))
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

            faces.append(gray_face)
            count += 1

            cv2.imshow('Capturing', frame)
            cv2.waitKey(100)  # Wait 100ms between captures

    cap.release()
    cv2.destroyAllWindows()

    if len(faces) == 0:
        messagebox.showerror("Error", "No faces detected during capture.")
        return None

    return faces

def register_face():
    name = entry_name.get().strip()
    if not name:
        messagebox.showerror("Error", "Please enter a name.")
        return

    if name in name_to_label:
        messagebox.showerror("Error", "Name already registered.")
        return

    global next_label
    label = next_label
    next_label += 1

    faces = capture_faces(name)
    if faces is None:
        return

    labels = [label] * len(faces)

    # Update recognizer
    if os.path.exists(MODEL_PATH):
        recognizer.update(faces, np.array(labels))
    else:
        recognizer.train(faces, np.array(labels))

    recognizer.write(MODEL_PATH)

    # Update labels
    label_to_name[label] = name
    name_to_label[name] = label
    with open(LABELS_PATH, 'wb') as f:
        pickle.dump(label_to_name, f)

    messagebox.showinfo("Success", f"Face registered for {name}.")

def recognize_face():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue

            face_img = cv2.resize(face_img, (200, 200))
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

            label, confidence = recognizer.predict(gray_face)
            if confidence < 100:  # Threshold for recognition
                name = label_to_name.get(label, "Unknown")
                text = f"{name} ({confidence:.2f})"
            else:
                text = "Unknown"

            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI Setup
root = tk.Tk()
root.title("Face Recognition System")

tk.Label(root, text="Name:").pack(pady=10)
entry_name = tk.Entry(root)
entry_name.pack()

btn_register = tk.Button(root, text="Register Face", command=register_face)
btn_register.pack(pady=10)

btn_recognize = tk.Button(root, text="Recognize Face", command=recognize_face)
btn_recognize.pack(pady=10)

root.mainloop()