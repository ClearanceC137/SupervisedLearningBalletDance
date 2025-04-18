import numpy as np
import os
import cv2  # Ensure OpenCV is installed for video processing
from keras.models import load_model
import mediapipe as mp


# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# Load the trained model
model = load_model('my_model.h5')

# Function to extract keypoints from a video
def process_video(video_path):
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
            keypoints_frame = ([landmark.x for landmark in landmarks] + 
                               [landmark.y for landmark in landmarks] + 
                               [landmark.z for landmark in landmarks] + 
                               [landmark.visibility for landmark in landmarks])
            keypoints.append(keypoints_frame)
    
    cap.release()
    return np.array(keypoints)
# Format size of input
def pad_to_num_rows(combined_features_list, num_rows):
    """
    Pads each combined feature array to the specified number of rows.
    
    Parameters:
    combined_features_list (list): List of combined feature arrays where each array may have a different number of rows.
    num_rows (int): The desired number of rows for all arrays.

    Returns:
    padded_combined_features (list): List of padded arrays with the specified number of rows.
    """
    padded_combined_features = []
    for features in combined_features_list:
        rows_to_add = num_rows - features.shape[0]  # Calculate how many rows to add
        if rows_to_add > 0:
            # Pad with rows of zeros at the bottom
            padded_array = np.vstack((features, np.zeros((rows_to_add, features.shape[1]))))
        else:
            # If no padding is needed or if the array has more rows than num_rows, slice it to num_rows
            padded_array = features[:num_rows]
        padded_combined_features.append(padded_array)
    
    return padded_combined_features
# Process the video and make predictions
video_path = r'C:\Users\clear\Desktop\Projects\Classifying Dance Moves\uploads\ACM0000.mp4'
# Process the video and make predictions
input_data = [process_video(video_path)]  # Get input data into a list for pad_to_num_rows(input_data,min_rows) to process properly
min_rows = 3000
input_data = pad_to_num_rows(input_data,min_rows)
format_input_data = np.array([input_data[0]])  # 3D format for prediction
predictions = model.predict(format_input_data)
predicted_class = np.argmax(predictions, axis=1)[0]  # Get the index of predicted class
accuracy = np.max(predictions) * 100  # Get the highest probability as accuracy