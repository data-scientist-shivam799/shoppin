# AI Video Analysis: Shoppable Item Detection

## Objective
The goal of this project is to detect shoppable items in a 60-second video, reducing the computational time by selecting only 20 frames out of 900 frames for object detection, while maintaining high accuracy.

## Problem Statement
Given a video with 900 frames, we aim to detect shoppable items by selecting a reduced set of 20 frames. The selected frames should provide sufficient information for the object detection algorithm to identify shoppable items (e.g., products in retail scenarios). The challenge is to minimize computational time while maintaining high detection accuracy.

## Approach

### Frame Selection Techniques:
1. **Random Frame Selection**: Select random frames from the video.
2. **Keyframe Extraction**: Select frames where significant changes in the scene are detected.
3. **Scene Change Detection**: Identify frames that show significant scene transitions (e.g., camera panning or new product introduction).
4. **Time-based Sampling**: Select frames evenly distributed throughout the video.

Each method was compared in terms of the number of frames processed and the accuracy of object detection on the selected frames.

### Object Detection:
- **YOLO (You Only Look Once)** model is used for object detection.
- Pre-trained YOLO model is applied to the selected frames to identify shoppable items (e.g., clothing, accessories, electronics).
- The detection output includes bounding boxes around the objects with a confidence score.

### Evaluation Metrics:
- **Accuracy**: The percentage of correctly identified shoppable items in the selected frames.
- **Processing Time**: Time taken to process the selected frames.
- **Precision** and **Recall**: To measure the quality of the object detection.
