import cv2
import os
import numpy as np
import pickle
# Path to the main directory containing subfolders with videos
main_directory = r"C:\Users\clear\Desktop\BSC Computer Science hons\Second Semester\Big Data\Project\UJAnnChor-main\UJAnnChor-main\AnnChor1000-Original-Videos"

# Frame rate settings
fps = 24  # Desired frames per second kv

# Create ORB detector
orb = cv2.ORB_create()

# Iterate through each subfolder in the main directory
for class_folder in os.listdir(main_directory):
    class_path = os.path.join(main_directory, class_folder)
    if os.path.isdir(class_path):  # Check if it's a directory
        # Iterate through each video file in the class folder
        for video_file in os.listdir(class_path):
            video_path = os.path.join(class_path, video_file)
            if video_file.endswith('.mp4'):  # Check if the file is a video
                cap = cv2.VideoCapture(video_path)

                frame_interval = int(cap.get(cv2.CAP_PROP_FPS)) // fps  # Calculate frame interval
                frame_count = 0
                video_features = []  # To store descriptors for the current video

                while cap.isOpened():
                    ret, frame = cap.read()

                    if not ret:
                        print(f"Warning: Failed to read frame from {video_file}")
                        break  # Exit loop if there are no more frames

                    try:
                        # Convert the frame to grayscale
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # Detect keypoints and compute descriptors
                        keypoints, descriptors = orb.detectAndCompute(gray_frame, None)

                        # Save descriptors if detected
                        if descriptors is not None:
                            video_features.append(descriptors)
                        else:
                            print(f"Warning: No descriptors found in frame {frame_count} of {video_file}")

                    except cv2.error as e:
                        print(f"Error processing frame {frame_count} of {video_file}: {e}")
                        continue  # Skip to the next frame if an error occurs

                    frame_count += 1

                # Release the video capture object
                cap.release()

                # Save the features data for the current video
                save_path = os.path.join(r"C:\Users\clear\Desktop\BSC Computer Science hons\Second Semester\Big Data\Features",  f"{os.path.splitext(video_file)[0]}.pkl")
                with open(save_path, 'wb') as f:
                    pickle.dump(video_features, f)  # Save list of descriptors using pickle

print("Processed videos and saved features individually.")
