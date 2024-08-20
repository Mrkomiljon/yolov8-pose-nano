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
    git clone https://github.com/Mrkomiljon/yolov8-pose-nano.git
    cd yolov8-pose-nano
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

1. Prepare the COCO dataset and structure it as follows:
 ### Run the script to download and extract the COCO dataset:

```bash
    python download_coco.py
```

   By default, the dataset will be saved in the `../Dataset/COCOPose` directory.
   ### Run the script to generate the image lists:

```bash
    python generate_image_list.py
```

   This will create `train2017.txt` and `val2017.txt` in the `Dataset/COCOPose/` directory.

    ```plaintext
    Dataset/
    ├── COCOPose/
    │   ├── images/
    │   │   ├── train2017/
    │   │   ├── val2017/
    │   ├── annotations/
    │   │   ├── person_keypoints_train2017.json
    │   │   ├── person_keypoints_val2017.json
    │   ├── train2017.txt
    │   ├── val2017.txt
    ```

3. Ensure you have the pre-trained model weights (if any) in the `./weights/` directory:

    ```plaintext
    ./weights/best.pt
    ```

## Training

To start training, run the following command:

```bash
python main.py --train --batch-size 16 --epochs 1000 # desired number
```
Command Line Arguments
--input-size: The size to which input images will be resized before processing. Default is 640.
--batch-size: The batch size used for training. Default is 32.
--local_rank: The rank of the current process in distributed training. Automatically set by PyTorch.
--epochs: The number of epochs to train the model. Default is 1000.
--train: Flag to start the training process.
Distributed Training
To train on multiple GPUs, use the following command:

```bash
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --train --batch-size 16 --epochs 300
```
This command will utilize 4 GPUs for training.

### Training Configuration
- The training configuration is stored in the utils/args.yaml file. You can modify this file to adjust various settings, such as learning rates, optimizer parameters, and data augmentation techniques.

## Checkpoints
- Model checkpoints are saved in the weights/ directory during training.
- The best-performing model is saved as weights/best.pt.
- The most recent model is saved as weights/last.pt.
- Monitoring Training
- Training progress, including loss and evaluation metrics, is logged and can be monitored via the console output. A CSV log is also maintained at weights/step.csv, which records the performance metrics (BoxAP and PoseAP) for each epoch.

### Evaluation
After training, you can evaluate the model on the COCO validation set using:

```bash
python main.py --test
```
### Demo
To visualize the model in action using your webcam, run:

```bash
python main.py --demo --input-size 640
```
## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

### Reference

* https://github.com/ultralytics/yolov5
* https://github.com/ultralytics/ultralytics
