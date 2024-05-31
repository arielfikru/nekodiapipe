from ultralytics import YOLO
import cv2
import numpy as np
import random
import string
import matplotlib.pyplot as plt

# Load the YOLOv8 model for pose estimation
model = YOLO("yolov8s-pose.pt")

# Define the input image path
image_path = "input.png"

# Run inference on the provided image
results = model(image_path)

# Read the original image
img = cv2.imread(image_path)
height, width, _ = img.shape
aspect_ratio = width / height

# Iterate through each detected person
for idx, result in enumerate(results):
    # Extract keypoints
    keypoints = result.keypoints.data.cpu().numpy()

    # Identify keypoints relevant to the abdomen (e.g., hips and lower torso keypoints)
    # Assuming keypoints indices for hips are 11 (right hip) and 12 (left hip)
    abdomen_keypoints = keypoints[0][[11, 12], :2]

    # Check if abdomen keypoints are found
    if abdomen_keypoints.size == 0:
        print("Tidak ditemukan keypoints pada hips")
        continue  # Skip to the next person

    # Calculate the minimum y-coordinate to define the lower boundary of the upper body
    min_abdomen_y = np.min(abdomen_keypoints[:, 1])

    # Calculate the crop area
    top = 0
    bottom = int(min_abdomen_y)
    crop_height = bottom - top
    crop_width = int(crop_height * aspect_ratio)

    # Center the crop horizontally
    center_x = width // 2
    left = max(center_x - crop_width // 2, 0)
    right = min(center_x + crop_width // 2, width)

    # Read the original image again to avoid rendering keypoints
    img_no_keypoints = cv2.imread(image_path)

    # Perform the cropping without keypoints
    cropped_img_no_keypoints = img_no_keypoints[top:bottom, left:right]

    # Resize the cropped image to maintain the aspect ratio of the original image
    final_img_no_keypoints = cv2.resize(cropped_img_no_keypoints, (width, height))

    # Generate a random 6-digit string for the output file name
    random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    output_filename_no_keypoints = f"{random_str}_person_{idx}_result_no_keypoints.jpg"

    # Save the resulting image with upper body cropped without keypoints
    cv2.imwrite(output_filename_no_keypoints, final_img_no_keypoints)

    # Render keypoints on the original image
    img_with_keypoints = result.plot()

    # Perform the cropping with keypoints
    cropped_img_with_keypoints = img_with_keypoints[top:bottom, left:right]

    # Resize the cropped image to maintain the aspect ratio of the original image
    final_img_with_keypoints = cv2.resize(cropped_img_with_keypoints, (width, height))

    # Generate a random 6-digit string for the output file name
    output_filename_with_keypoints = f"{random_str}_person_{idx}_result_with_keypoints.jpg"

    # Save the resulting image with upper body cropped with keypoints
    cv2.imwrite(output_filename_with_keypoints, final_img_with_keypoints)

    # Plot the images
    plot = False
    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        ax[1].imshow(cv2.cvtColor(final_img_with_keypoints, cv2.COLOR_BGR2RGB))
        ax[1].set_title(f"{random_str}_person_{idx}_result_with_keypoints.jpg")
        ax[1].axis("off")

        ax[2].imshow(cv2.cvtColor(final_img_no_keypoints, cv2.COLOR_BGR2RGB))
        ax[2].set_title(f"{random_str}_person_{idx}_result_no_keypoints.jpg")
        ax[2].axis("off")

        fig.suptitle(f"Yolo Cropping for Person {idx}", fontsize=16)
        plt.show()
