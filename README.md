# MediaPipe Pose Detection

This repository contains a script for detecting and visualizing pose landmarks on images using MediaPipe. The script processes an input image to generate three versions: the original image, the image with pose landmarks overlaid, and a black background with only the pose landmarks.

## Requirements

- Python 3.10+
- MediaPipe
- OpenCV
- NumPy
- Matplotlib

## Setup

1. Clone this repository and navigate to the project directory.
2. Install the required packages:
    ```
    pip install -r requirements.txt
    ```
3. Download the pose landmark model:
    ```
    wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
    ```
4. Ensure you have an image file named `image.jpg` in the project directory or update the `image_path` variable in the script to point to your image.

## Usage

Run the script to process the image and save the output in the `test` subfolder:
```
python mediapipe_inference.py
```

The script will generate a combined image consisting of the original image, the image with pose landmarks, and the black background with pose landmarks only. This combined image will be saved in the `test` subfolder with a timestamp in the filename.

## Limitations

- MediaPipe may fail to detect poses in certain types of images such as:
  - Chibi characters
  - Images of people sleeping
  - Full body images where the legs are not clearly visible

Further research and adjustments are needed to improve pose detection in these cases.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements

- [MediaPipe](https://mediapipe.dev) for providing the pose detection solution.
- [Google](https://www.google.com) for hosting the pose landmark model.
