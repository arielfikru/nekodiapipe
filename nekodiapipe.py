# Import necessary modules
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse

# Initialize MediaPipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to draw landmarks on image
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        mp_drawing.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))
    return annotated_image

# Function to display grid of images with labels and save the combined image
def make_image_grid_and_save(images, labels, rows, cols, original_image_path):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 7))
    for ax, img, label in zip(axes.flat, images, labels):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        ax.set_title(label)
    plt.show()

    # Save the combined image to a file in the "preview" subfolder
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_image = np.hstack(images)
    os.makedirs("preview", exist_ok=True)
    save_path = os.path.join("preview", f"{current_time}_combined_{os.path.basename(original_image_path)}.png")
    cv2.imwrite(save_path, combined_image)
    print(f"Saved {save_path}")

# Function to process the image and perform all tasks
def process_image(image_path):
    # Create a PoseLandmarker object.
    base_options = python.BaseOptions(model_asset_path='./model/pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    # Load the input image.
    image = mp.Image.create_from_file(image_path)
    rgb_image = cv2.imread(image_path)

    # Detect pose landmarks from the input image.
    detection_result = detector.detect(image)

    # Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(rgb_image, detection_result)

    # Check if segmentation masks are present
    if detection_result.segmentation_masks:
        segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
        visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
    else:
        print("Segmentation mask not available.")
        visualized_mask = np.zeros_like(rgb_image)

    # Create a black background for the pose skeleton only
    black_background = np.zeros_like(rgb_image)
    pose_landmarks_list = detection_result.pose_landmarks
    for pose_landmarks in pose_landmarks_list:
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        mp_drawing.draw_landmarks(
            black_background,
            pose_landmarks_proto,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))

    # Display and save the combined image
    images = [rgb_image, annotated_image, black_background]
    labels = ["Original Image", "Image with Pose Skeleton", "Pose Skeleton Only"]
    make_image_grid_and_save(images, labels, 1, 3, image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image to detect and visualize pose landmarks.")
    parser.add_argument("image_path", type=str, help="The path to the input image.")
    args = parser.parse_args()

    process_image(args.image_path)
