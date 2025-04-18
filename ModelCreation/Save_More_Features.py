import cv2
import os
import numpy as np

# Path to the main directory containing subfolders with videos
main_directory = r"C:\Users\clear\Desktop\BSC Computer Science hons\Second Semester\Big Data\Project\UJAnnChor-main\UJAnnChor-main\AnnChor1000-Original-Videos"

# Frame rate settings
fps = 24  # Desired frames per second

# Create ORB detector
orb = cv2.ORB_create()

# Dictionary to store features data and labels
features_data = {}
class_labels = []

# Iterate through each subfolder in the main directory
for class_folder in os.listdir(main_directory):
    class_path = os.path.join(main_directory, class_folder)
    print(class_path)
    if os.path.isdir(class_path):  # Check if it's a directory
        # Store the class label
        class_labels.append(class_folder)

        # Iterate through each video file in the class folder
        for video_file in os.listdir(class_path):
            video_path = os.path.join(class_path, video_file)
            #print(video_path,"\n")
            if video_file.endswith('.mp4'):  # Check if the file is a video
                cap = cv2.VideoCapture(video_path)

                frame_interval = int(cap.get(cv2.CAP_PROP_FPS)) // fps  # Calculate frame interval

                frame_count = 0
                saved_frame_count = 0
                video_features = []  # To store descriptors for the current video

                while cap.isOpened():
                    ret, frame = cap.read()
                    
                    if not ret:
                        break  # Exit loop if there are no more frames

                    # Save every 'frame_interval' frame
                    if True:
                        # Convert the frame to grayscale
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # Detect keypoints and compute descriptors
                        keypoints, descriptors = orb.detectAndCompute(gray_frame, None)

                        if descriptors is not None:
                            video_features.append(descriptors)  # Store descriptors

                        saved_frame_count += 1

                    frame_count += 1

                # Release the video capture object
                cap.release()

                # Save the features data for the current video
                features_data[video_file] = {
                    "descriptors": video_features,
                    "label": class_folder
                }

# Optionally: Save the features data to a file (e.g., using numpy or pickle)
# For example, saving to a numpy file:
np.save(r"C:\Users\clear\Desktop\BSC Computer Science hons\Second Semester\Big Datafeatures_data.npy", features_data)

# Print the collected features and labels
print(features_data)
print(f"Processed {len(features_data)} videos from {len(class_labels)} classes.")
