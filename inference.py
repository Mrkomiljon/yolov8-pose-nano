import argparse
import cv2
import numpy as np
import os
import torch

from nets import nn
from utils import util


@torch.no_grad()
def infer(args):
    # Load the model
    model = torch.load('./weights/best.pt', map_location='cuda')['model'].float()
    model.half()
    model.eval()

    # Prepare output path
    os.makedirs('results', exist_ok=True)
    if args.source.isdigit():  # Webcam
        output_filename = 'results/webcam.mp4'
    else:
        base_filename = os.path.basename(args.source)
        output_filename = f'results/{os.path.splitext(base_filename)[0]}.mp4'

    # Handle input source (image, video, or webcam)
    if args.source.isdigit():  # Webcam
        cap = cv2.VideoCapture(int(args.source))
    elif args.source.endswith(('.mp4', '.avi', '.mov')):  # Video file
        cap = cv2.VideoCapture(args.source)
    else:  # Image file
        frame = cv2.imread(args.source)
        if frame is None:
            print(f"Error: Unable to open image file {args.source}")
            return
        output_frame = process_frame(frame, model, args)
        output_frame = cv2.resize(output_frame, (args.output_width, args.output_height))  # Resize output frame
        cv2.imwrite(output_filename, output_frame)
        print(f"Saved output to {output_filename}")
        return

    if not cap.isOpened():
        print(f"Error: Unable to open video or webcam source {args.source}")
        return

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, 20.0, (args.output_width, args.output_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        output_frame = process_frame(frame, model, args)
        output_frame = cv2.resize(output_frame, (args.output_width, args.output_height))  # Resize output frame
        out.write(output_frame)  # Write the frame to the output video
        cv2.imshow('Inference', output_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved output video to {output_filename}")


def process_frame(frame, model, args):
    original_shape = frame.shape[:2]  # (height, width)
    stride = int(max(model.stride.cpu().numpy()))  # model stride

    # Resize and pad the image while keeping aspect ratio
    r = min(args.input_size / original_shape[0], args.input_size / original_shape[1])
    resized_shape = (int(round(original_shape[1] * r)), int(round(original_shape[0] * r)))  # width, height
    pad_w, pad_h = args.input_size - resized_shape[0], args.input_size - resized_shape[1]
    pad_w, pad_h = np.mod(pad_w, stride), np.mod(pad_h, stride)  # adjust padding to be a multiple of stride
    pad_w //= 2
    pad_h //= 2

    # Resize and pad the image
    resized_frame = cv2.resize(frame, resized_shape, interpolation=cv2.INTER_LINEAR)
    padded_frame = cv2.copyMakeBorder(resized_frame, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # Convert BGR to RGB and prepare for model
    image = padded_frame[:, :, ::-1].transpose(2, 0, 1)  # HWC to CHW
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image).to('cuda').half() / 255.0  # Convert to tensor and scale to [0, 1]
    image = image.unsqueeze(0)

    # Inference
    outputs = model(image)

    # Non-Maximum Suppression
    outputs = util.non_max_suppression(outputs, 0.25, 0.7, model.head.nc)

    # Skeleton definition
    skeleton = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], 
        [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], 
        [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], 
        [2, 4], [3, 5], [4, 6], [5, 7]
    ]
    
    # Define colors for each line in the skeleton
    skeleton_colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (75, 0, 130),   # Indigo
        (0, 128, 128),  # Teal
        (255, 105, 180),  # Pink
        (173, 255, 47),  # Green Yellow
        (220, 20, 60),   # Crimson
        (0, 206, 209),   # Dark Turquoise
        (238, 130, 238), # Violet
        (240, 230, 140), # Khaki
        (123, 104, 238), # Medium Slate Blue
        (124, 252, 0),   # Lawn Green
    ]

    # Create a blank image to draw the skeleton
    skeleton_image = np.zeros_like(frame)

    # Post-process the outputs
    for output in outputs:
        output = output.clone()
        if len(output):
            kps_output = output[:, 6:].view(len(output), *model.head.kpt_shape)
        else:
            kps_output = torch.zeros((0, *model.head.kpt_shape))

        # Undo padding and scaling for keypoints
        kps_output[..., 0] -= pad_w
        kps_output[..., 1] -= pad_h
        kps_output[..., 0] /= r
        kps_output[..., 1] /= r
        kps_output[..., 0].clamp_(0, original_shape[1])
        kps_output[..., 1].clamp_(0, original_shape[0])

        # Draw keypoints and skeleton on the original frame and the blank skeleton image
        for kpt in kps_output:
            points = []
            for i, k in enumerate(kpt):
                x_coord, y_coord, conf = k
                if conf > 0.5:
                    cv2.circle(frame, (int(x_coord), int(y_coord)), 5, (0, 0, 255), -1)
                    cv2.circle(skeleton_image, (int(x_coord), int(y_coord)), 5, (0, 0, 255), -1)
                    points.append((int(x_coord), int(y_coord)))
                else:
                    points.append(None)
            
            # Draw skeleton lines with different colors
            for idx, sk in enumerate(skeleton):
                pt1, pt2 = points[sk[0] - 1], points[sk[1] - 1]
                if pt1 is not None and pt2 is not None:
                    color = skeleton_colors[idx % len(skeleton_colors)]  # Cycle through colors
                    cv2.line(frame, pt1, pt2, color, 2)
                    cv2.line(skeleton_image, pt1, pt2, color, 2)

    # Concatenate the original frame with the skeleton image side by side
    output_frame = np.hstack((frame, skeleton_image))

    return output_frame



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int, help='Input size for the model')
    parser.add_argument('--output-width', default=1920, type=int, help='Width of the output frame')
    parser.add_argument('--output-height', default=1024, type=int, help='Height of the output frame')
    parser.add_argument('--source', type=str, required=True, help='Path to image, video file, or webcam index (0, 1, ...)')
    
    args = parser.parse_args()
    
    infer(args)


if __name__ == "__main__":
    main()
