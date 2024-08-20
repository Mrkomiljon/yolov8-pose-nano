# yolov8-pose-nano
# Pose Estimation and Skeleton Visualization

This project provides a script for performing pose estimation and visualizing skeletons on images, videos, or webcam feeds using a pre-trained deep learning model.

## Features

- **Pose Estimation**: Leverages a deep learning model to detect keypoints in human poses.
- **Skeleton Visualization**: Draws a colored skeleton on the detected keypoints.
- **Multiple Input Sources**: Supports images, video files, and real-time webcam feeds as input sources.
- **Customizable Output**: Allows setting custom output dimensions for the resulting visualization.

## Installation

### Prerequisites

- Python 3.7+
- PyTorch
- OpenCV
- NumPy

### Setup

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/pose-estimation.git
    cd pose-estimation
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have the pre-trained model weights in the `./weights/` directory:

    ```plaintext
    ./weights/best.pt
    ```

## Usage

You can run the pose estimation and skeleton visualization script using the command line.

### Command Line Arguments

- `--input-size`: The size to which the input will be resized before processing. Default is `640`.
- `--output-width`: The width of the output frame. Default is `1920`.
- `--output-height`: The height of the output frame. Default is `1024`.
- `--source`: Path to the image, video file, or webcam index (`0`, `1`, etc.) for real-time processing. Required.

### Examples

1. **Processing an Image**:

    ```bash
    python infer.py --source ./data/sample_image.jpg --output-width 1280 --output-height 720
    ```

2. **Processing a Video**:

    ```bash
    python infer.py --source ./data/sample_video.mp4 --output-width 1920 --output-height 1080
    ```

3. **Using Webcam**:

    ```bash
    python infer.py --source 0 --output-width 1280 --output-height 720
    ```

### Output

- For image input, the result is saved as an image in the `results/` directory.
- For video or webcam input, the result is saved as a video in the `results/` directory.
- The output consists of the original input with an overlay of the detected skeleton and keypoints.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
