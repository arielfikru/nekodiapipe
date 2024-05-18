# NekodiaPipe

This repository contains scripts for detecting and visualizing pose landmarks on images using MediaPipe. The scripts process an input image to generate different visualizations, including the original image, the image with pose landmarks overlaid, and a black background with only the pose landmarks. Additionally, one script can crop the upper body from the input image.

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
    wget -O ./model/pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
    ```
4. Ensure you have an image file to process or update the `image_path` argument when running the scripts.

## Usage

### Regular Inference

Run the script `nekodiapipe.py` to process the image and save the output in the `preview` subfolder:
```
python nekodiapipe.py path/to/your/image.jpg
```

The script will generate a combined image consisting of the original image, the image with pose landmarks, and the black background with pose landmarks only. This combined image will be saved in the `preview` subfolder with a timestamp in the filename.

### Inference and Crop Upper Body

Run the script `nekodiapipe_upperbody.py` to process the image, crop the upper body, and save the output in the `preview` subfolder:
```
python nekodiapipe_upperbody.py path/to/your/image.jpg
```

The script will generate a combined image consisting of the original image, the image with pose landmarks, the black background with pose landmarks only, and the cropped upper body image. This combined image will be saved in the `preview` subfolder with a timestamp in the filename. The cropped upper body image will also be saved separately.

## Limitations

- MediaPipe may fail to detect poses in certain types of images such as:
  - Chibi characters
  - Images of people sleeping
  - Images of people sitting down
  - Full body images where the legs are not clearly visible

Further research and adjustments are needed to improve pose detection in these cases.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements

- [MediaPipe](https://mediapipe.dev) for providing the pose detection solution.
- [Google](https://www.google.com) for hosting the pose landmark model.
