import torch
from torchvision import transforms
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory of 'utils' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model():
    model = torch.load('C:\\Users\\austi\\Documents\\EE299\\yolov7\\yolov7-w6-pose.pt', map_location=device)['model']
    model.float().eval()
    if torch.cuda.is_available():
        model.half().to(device)
    return model

model = load_model()

def run_inference(image):
    image = letterbox(image, 960, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    if torch.cuda.is_available():
        image = image.half().to(device)
    image = image.unsqueeze(0)
    with torch.no_grad():
        output, _ = model(image)
    return output, image

def draw_keypoints(output, image):
    output = non_max_suppression_kpt(output, 
                                     0.05, 
                                     0.25, 
                                     nc=model.yaml['nc'], 
                                     nkpt=model.yaml['nkpt'], 
                                     kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
    return nimg

def save_keypoint_video(input_video, output_video, snippet_duration=10, start_time=0):
    cap = cv2.VideoCapture(input_video)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = snippet_duration * fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened() and frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output, _ = run_inference(frame_rgb)
        frame_with_keypoints = draw_keypoints(output, _)
        
        # Resize frame to match output video dimensions if necessary
        frame_with_keypoints = cv2.resize(frame_with_keypoints, (width, height))
        
        out.write(frame_with_keypoints)
        frame_count += 1
        
        cv2.imshow('Pose Estimation', frame_with_keypoints)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Define input and output video paths
input_video_path = "C:\\Users\\austi\\Documents\\EE299\\yolov7\\freestyle-training.mp4"
output_video_path = "C:\\Users\\austi\\Documents\\EE299\\yolov7\\freestyle-training-keypoints-snippet.mp4"

# Save a 5-second snippet showing the keypoint segments applied (starting at 10 seconds)
save_keypoint_video(input_video_path, output_video_path, start_time=170)
