from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import os
import cv2  # Ensure OpenCV is installed for video processing
from keras.models import load_model
import mediapipe as mp
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

app = Flask(__name__)

# Load the trained model
model = load_model('TNN_Trained_Model.h5')     # max_input_row = 3000 & 52%

Dance_Moves= [
    "ACM : Diana and Acteon variation (male) : Diana and Acteon Pas de Deux",
    "ALB : Albrecht variation (male) : Giselle",
    "AUR : Aurora variation (female) : Sleeping Beauty",
    "BAS : Basilio variation (male) : Don Quixote",
    "BLU : Bluebird variation (female) : Sleeping",
    "BRO : Bridesmaid variation 1 (female) : Don Quixote",
    "CAL : Calvary Halt variation (female) : The Calvary Halt",
    "COL : Colas variation (male) : La Fille Mal Gardee",
    "CUP : Cupid variation (female) : Don Quixote",
    "DAA : Diana and Acteon variation (female) : Diana and Acteon Pas de Deux",
    "DUL : Dulcinea variation (female) : Don Quixote",
    "ESM : Esmeralda variation (female) : La Esmeralda",
    "FPF : Flames of Paris variation (female) : Flames of Paris",
    "FPM : Flames of Paris variation (male) : Flames of Paris",
    "FRA : Franz variation (male) : Coppelia",
    "GAM : Gamzatti variation (female) : La Bayadere",
    "GIS : Giselle variation (female) : Giselle",
    "GPM : Grand Pas Classique variation (male) : Paquita",
    "GUL : Gulnara variation (female) : Le Corsaire",
    "HAF : Harlequinade variation (female) : Harlequinade",
    "HAM : Harlequinade variation (male) : Harlequinade",
    "KIT : Kitri variation (female) : Don Quixote",
    "LA1 : La Bayadere Shade 1 variation (female) : La Bayadere",
    "LA2 : La Bayadere Shade 2 variation (female) : La Bayadere",
    "LA3 : La Bayadere Shade 3 variation (female) : La Bayadere",
    "LAN : Lankedem variation (male) : Le Corsaire",
    "LCM : Le Corsaire variation (male) : Le Corsaire",
    "LFQ : La Fille Mal Gardee Quick variation (female) : La Fille Mal Gardee",
    "LFS : La Fille Mal Gardee Slow variation (female) : La Fille Mal Gardee",
    "LIF : Lilac Fairy variation (female) : Sleeping Beauty",
    "LUC : Lucien variation (male) : Paquita",
    "NUP : Nutcracker Prince variation (male) : The Nutcracker",
    "OD1 : Odile version 1 variation (female) : Swan Lake",
    "OD2 : Odile version 2 variation (female) : Swan Lake",
    "ODA : Odalisque Pas de Trois 2nd variation (female) : Le Corsaire",
    "PAA : Paquita Arabesque variation (female) : Paquita",
    "PAR : Paquita Renverse variation (female) : Paquita",
    "PAV : Paquita Pas de Trois 1st variation (female) : Paquita",
    "PHD : Pharoah's Daughter variation (female) : Pharoah's Daughter",
    "QOD : Queen of the Dryads variation (female) : Don Quixote",
    "RMT : Raymonda Tableau variation (female) : Raymonda",
    "SAT : Satanella variation (male) : Satanella",
    "SBP : Sleeping Beauty Prince variation (male) : Sleeping Beauty",
    "SIE : Siegfried variation (male) : Swan Lake",
    "SOL : Solor variation (male) : La Bayadere",
    "SWA : Swanhilda variation (female) : Coppelia",
    "TCF : Tchaikovsky Pas de Deux variation (female) : Tchaikovsky Pas de Deux",
    "TCM : Tchaikovsky Pas de Deux variation (male) : Tchaikovsky Pas de Deux",
    "TMF : Talisman variation (female) : The Talisman",
    "TMM : Talismane variation (male) : The Talisman",
    "WAL : Walpurgis Night variation (female) : Walpurgisnacht"
]


def visualize_keypoints_matplotlib(frame, keypoints_frame):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert frame to RGB for Matplotlib
    
    for i in range(0, 33 , 1):
        x = keypoints_frame[i] * frame.shape[1]  # Normalize x
        y = keypoints_frame[32+i+1] * frame.shape[0]  # Normalize y
        plt.scatter(x, y, c='r', s=20)  # Plot each keypoint as a red dot
    # Save the plot to an in-memory buffer
    # Convert the Matplotlib plot to a PIL image
    # Create a BytesIO buffer to save the figure
    buf = io.BytesIO()
    plt.savefig(buf, format='png')  # Save the plot to the buffer
    buf.seek(0)  # Rewind the buffer to the beginning
    
    # Convert the buffer directly to Base64
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    return img_str
# Function to extract keypoints from a video
def Get_image_with_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    image = []
    
    for i in range(1):
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
            image.append(visualize_keypoints_matplotlib(frame, keypoints_frame))   # data visualization
            
    
    cap.release()
    return image

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

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Create uploads directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    video_path = os.path.join('uploads', file.filename)
    file.save(video_path)  # Save the uploaded file
    images_base64 = Get_image_with_keypoints(video_path)
    return render_template('predict.html', video_path=video_path,file_name =file.filename,images = images_base64 )

@app.route('/predict', methods=['POST'])
def predict():
    video_path = request.form['video_path']  # Get the path of the uploaded video
    
    try:
        # Process the video and make predictions
        input_data = [process_video(video_path)]  # Get input data into a list for pad_to_num_rows(input_data,min_rows) to process properly
        min_rows = 3000
        input_data = pad_to_num_rows(input_data,min_rows)
        format_input_data = np.array([input_data[0]])  # 3D format for prediction
        predictions = model.predict(format_input_data)
        predicted_class = np.argmax(predictions, axis=1)[0]  # Get the index of predicted class
        accuracy = np.max(predictions) * 100  # Get the highest probability as accuracy
    except Exception as e:
        return f'Error during prediction: {str(e)}', 500  # Internal Server Error

    # Render the result HTML with the prediction and accuracy data
    return render_template('result.html', predicted_class=Dance_Moves[predicted_class], accuracy=f"{accuracy:.2f}")

if __name__ == '__main__':
    app.run(debug=True)
