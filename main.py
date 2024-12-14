import os
import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans

# Paths and Configuration
VIDEO_PATH = "data/input_video.mp4"
FRAME_DIR = "data/frames"
SELECTED_FRAMES_DIR = "outputs/selected_frames"
DETECTIONS_DIR = "outputs/detections"
NUM_FRAMES_TO_SELECT = 20

os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(SELECTED_FRAMES_DIR, exist_ok=True)
os.makedirs(DETECTIONS_DIR, exist_ok=True)


# 1. Frame Extraction
def extract_frames(video_path, frame_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Extracting {frame_count} frames...")

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frame_dir, f"frame_{frame_id}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_id += 1

    cap.release()
    print(f"Frames saved in {frame_dir}")


# 2. Frame Selection Methods
def uniform_sampling(frame_dir, num_frames):
    frames = sorted(os.listdir(frame_dir))
    step = max(1, len(frames) // num_frames)
    selected_frames = frames[::step][:num_frames]
    return [os.path.join(frame_dir, frame) for frame in selected_frames]


def scene_change_detection(frame_dir, num_frames):
    frames = sorted(os.listdir(frame_dir))
    histograms = []
    frame_paths = []

    for frame in frames:
        img = cv2.imread(os.path.join(frame_dir, frame))
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        histograms.append(hist.flatten())
        frame_paths.append(os.path.join(frame_dir, frame))

    histograms = np.array(histograms)
    diff = np.linalg.norm(np.diff(histograms, axis=0), axis=1)
    scene_changes = np.argsort(-diff)[:num_frames]
    return [frame_paths[i] for i in scene_changes]


def kmeans_based_selection(frame_dir, num_frames):
    frames = sorted(os.listdir(frame_dir))
    features = []
    frame_paths = []

    for frame in frames:
        img = cv2.imread(os.path.join(frame_dir, frame))
        resized = cv2.resize(img, (64, 64)).flatten()
        features.append(resized)
        frame_paths.append(os.path.join(frame_dir, frame))

    features = np.array(features)
    kmeans = KMeans(n_clusters=num_frames, random_state=42)
    kmeans.fit(features)
    selected_indices = np.argmin(
        np.linalg.norm(features - kmeans.cluster_centers_[:, None], axis=2), axis=1
    )
    return [frame_paths[i] for i in selected_indices]


# 3. Object Detection
def detect_objects(frame_paths, output_dir):
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    detections = {}

    for frame_path in frame_paths:
        img = cv2.imread(frame_path)
        results = model(img)
        results.save(output_dir)

        frame_name = os.path.basename(frame_path)
        detections[frame_name] = results.pandas().xyxy[0].to_dict(orient="records")

    return detections


# 4. Main Workflow
def main():
    # Step 1: Extract Frames
    print("\n--- Step 1: Frame Extraction ---")
    extract_frames(VIDEO_PATH, FRAME_DIR)

    # Step 2: Frame Selection
    print("\n--- Step 2: Frame Selection ---")
    method = "kmeans"  # Options: "uniform", "scene_change", "kmeans"
    if method == "uniform":
        selected_frames = uniform_sampling(FRAME_DIR, NUM_FRAMES_TO_SELECT)
    elif method == "scene_change":
        selected_frames = scene_change_detection(FRAME_DIR, NUM_FRAMES_TO_SELECT)
    elif method == "kmeans":
        selected_frames = kmeans_based_selection(FRAME_DIR, NUM_FRAMES_TO_SELECT)

    print(f"Selected Frames: {selected_frames}")
    for frame in selected_frames:
        output_path = os.path.join(SELECTED_FRAMES_DIR, os.path.basename(frame))
        cv2.imwrite(output_path, cv2.imread(frame))

    # Step 3: Object Detection
    print("\n--- Step 3: Object Detection ---")
    detections = detect_objects(selected_frames, DETECTIONS_DIR)
    print(f"Detections saved in {DETECTIONS_DIR}")

    # Step 4: Evaluation (Optional Placeholder)
    print("\n--- Step 4: Evaluation ---")
    print(f"Evaluation results: {detections}")  # Replace with your evaluation metrics logic

    print("\nWorkflow Complete!")


if __name__ == "__main__":
    main()
