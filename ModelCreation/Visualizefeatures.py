# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 19:39:51 2024

@author: clear
"""

import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import matplotlib.pyplot as plt

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
num_features = 33
def visualize_keypoints_matplotlib(frame, keypoints_frame):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert frame to RGB for Matplotlib
    
    for i in range(0, num_features , 1):
        x = keypoints_frame[i] * frame.shape[1]  # Normalize x
        y = keypoints_frame[32+i+1] * frame.shape[0]  # Normalize y
        plt.scatter(x, y, c='r', s=20)  # Plot each keypoint as a red dot
    
    plt.show()
# Function to extract keypoints from a video
def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    
    while cap.isOpened():
        ret, frame = cap.read()    # ret : check if the a frame was extracted succesfully and frame : is the frame of the video at a specific time
        if not ret:
            break
        # Convert the frame to RGB for MediaPipe processing
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # converts to (r,g , b) format
        results = pose.process(image_rgb)  # This processes the RGB image using the MediaPipe Pose model, which detects human body pose landmarks (e.g., joints like shoulders, elbows, hips, etc.).
        
        # Extract pose landmarks if available
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark    # it contains a list of landmark objects, where each landmark represents a specific point on the body (e.g., shoulder, elbow, knee) with x, y, and z coordinates.
            keypoints_frame = [landmark.x for landmark in landmarks] + [landmark.y for landmark in landmarks] # extracting the x and y coordinates of each landmark.
            visualize_keypoints_matplotlib(frame, keypoints_frame)   # data visualization
            keypoints.append(keypoints_frame)
    
    cap.release()
    return np.array(keypoints)
video_path = r'C:\Users\clear\Desktop\BSC Computer Science hons\Second Semester\Big Data\Project\UJAnnChor-main\UJAnnChor-main\AnnChor1000-Original-Videos\SOL\SOL0001.mp4'
key_points =extract_keypoints(video_path)
def background_subtraction(video_path):
    # Initialize video capture and background subtractor
    cap = cv2.VideoCapture(video_path)
    backSub = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply background subtraction
        fgMask = backSub.apply(frame)

        # Display the resulting frame
        cv2.imshow('Frame', frame)
        cv2.imshow('FG Mask', fgMask)

        # Break the loop if 'ESC' is pressed
        if cv2.waitKey(30) & 0xFF == 27:  # Press 'ESC' to exit
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


#background_subtraction(video_path)