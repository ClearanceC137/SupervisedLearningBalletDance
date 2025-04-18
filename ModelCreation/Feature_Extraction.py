import os
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to extract keypoints from a video
def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert the frame to RGB for MediaPipe processing
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        # Extract pose landmarks if available
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints_frame = [landmark.x for landmark in landmarks] + [landmark.y for landmark in landmarks]
            keypoints.append(keypoints_frame)
    
    cap.release()
    return np.array(keypoints)

# Function to compute velocity and acceleration
def compute_temporal_features(keypoints):
    velocity = np.diff(keypoints, axis=0)  # Velocity: difference between frames
    acceleration = np.diff(velocity, axis=0)  # Acceleration: difference of velocity
    
    # Flatten velocity and acceleration
    velocity_features = velocity.reshape(velocity.shape[0], -1)
    acceleration_features = acceleration.reshape(acceleration.shape[0], -1)
    
    return velocity_features, acceleration_features

# Directory path to your dataset
dataset_dir = r'C:\Users\clear\Desktop\BSC Computer Science hons\Second Semester\Big Data\Project\UJAnnChor-main\UJAnnChor-main\AnnChor1000-Original-Videos'

keypoints_features = []
acceleration_features = []
velocity_features = []
labels = []

# Iterate through each class folder
for label_folder in os.listdir(dataset_dir):
    label_path = os.path.join(dataset_dir, label_folder)
    print(f"label Path: {label_path}")   
    if os.path.isdir(label_path):
        for video_file in os.listdir(label_path):
            video_path = os.path.join(label_path, video_file)
            keypoints = extract_keypoints(video_path)
            
            # If keypoints are extracted, append features and compute temporal data
            if keypoints.size > 0:
                keypoints_features.append(keypoints)
                labels.append(label_folder)
                
                velocity, acceleration = compute_temporal_features(keypoints)
                velocity_features.append(velocity)
                acceleration_features.append(acceleration)


keypoints_features = np.array(keypoints_features)
velocity_features = np.array(velocity_features)
acceleration_features = np.array(acceleration_features)

# Convert labels to numpy array
labels = np.array(labels)

# Save keypoints_features to a text file
np.savetxt('keypoints_features.txt', keypoints_features, delimiter=',')

# Save acceleration_features to a text file
np.savetxt('acceleration_features.txt', acceleration_features, delimiter=',')

# Save velocity_features to a text file
np.savetxt('velocity_features.txt', velocity_features, delimiter=',')

# Save labels to a text file
np.savetxt('labels.txt', labels, fmt='%s')  # fmt='%s' for string labels
