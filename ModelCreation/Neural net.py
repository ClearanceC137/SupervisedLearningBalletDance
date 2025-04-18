import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Reshape, Dropout, Bidirectional

# Load keypoints_features from a pickle file
with open('keypoints_features.pkl', 'rb') as f:
    keypoints_features = pickle.load(f)
    
# Load labels from a pickle file
with open('labels.pkl', 'rb') as f:
    labels = pickle.load(f)
    
def pad_to_max_rows(combined_features_list):
    """
    Pads each combined feature array to the number of rows of the longest array.
    
    Parameters:
    combined_features_list (list): List of combined feature arrays where each array may have a different number of rows.
    
    Returns:
    padded_combined_features (list): List of padded arrays with the same number of rows.
    """
    # Step 1: Find the maximum number of rows
    max_rows = max([features.shape[0] for features in combined_features_list])
    
    # Step 2: Pad each array with zeros so they all have max_rows rows
    padded_combined_features = []
    for features in combined_features_list:
        rows_to_add = max_rows - features.shape[0]  # Calculate how many rows to add
        if rows_to_add > 0:
            # Pad with rows of zeros at the bottom
            padded_array = np.vstack((features, np.zeros((rows_to_add, features.shape[1]))))
        else:
            padded_array = features  # If no padding is needed
        padded_combined_features.append(padded_array)
    
    return padded_combined_features

def pad_to_min_rows(combined_features_list):
    """
    Truncates each combined feature array to the number of rows of the shortest array.
    
    Parameters:
    combined_features_list (list): List of combined feature arrays where each array may have a different number of rows.
    
    Returns:
    truncated_combined_features (list): List of arrays, all truncated to the same number of rows.
    """
    # Step 1: Find the minimum number of rows
    min_rows = min([features.shape[0] for features in combined_features_list])
    
    # Step 2: Truncate each array to have min_rows rows
    truncated_combined_features = []
    for features in combined_features_list:
        # Truncate the array to the first min_rows rows
        truncated_array = features[:min_rows, :]
        truncated_combined_features.append(truncated_array)
    
    return truncated_combined_features

# Example of your current 3D array (shape: num_frames, 66)
# X would be the full array of all frames and features
def transform_to_4d(X_3d, num_keypoints=33):
    """
    Transforms a 3D array where each row is a frame and columns are [33 X, 33 Y] into a 4D array
    with dimensions (num_samples, num_frames, num_keypoints, [x, y]).
    
    Parameters:
    - X_3d: The 3D array to be transformed (num_frames, 66).
    
    Returns:
    - X_4d: The transformed 4D array (num_samples, num_frames, num_keypoints, 2).
    """
    # Step 1: Separate x and y coordinates
    x_coords = X_3d[:, :num_keypoints]  # First 33 columns are x-coordinates
    y_coords = X_3d[:, num_keypoints:]  # Next 33 columns are y-coordinates
    
    # Step 2: Stack x and y together to form [x, y] pairs for each keypoint
    keypoints_2d = np.stack((x_coords, y_coords), axis=-1)  # Shape becomes (num_frames, 33, 2)
    
    # Step 3: Reshape into 4D (num_samples, num_frames, num_keypoints, 2) if there are multiple samples
    return keypoints_2d
keypoints_features = pad_to_min_rows(keypoints_features)
Sample_feature_coord = []
for feature in keypoints_features:
    Sample_feature_coord.append(transform_to_4d(feature))
    
X = np.array(Sample_feature_coord)

# Create a LabelEncoder instance
label_encoder = LabelEncoder()

# Fit and transform the labels
encoded_labels = label_encoder.fit_transform(labels)

y = encoded_labels
num_classes = 51
# One-hot encode the labels if needed
y = np.eye(num_classes)[y]  # One-hot encoding

# Normalize input data (flatten and normalize the (x, y) coordinates)
scaler = MinMaxScaler()
X_reshaped = X.reshape(-1, 33 * 2)  # Flatten (33, 2) to 66
X_scaled = scaler.fit_transform(X_reshaped).reshape(X.shape)

# Split into training and validation sets if not already split
# Assuming X_train, X_val, y_train, y_val are ready, if not you can split them like this:
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential()

# Input layer for 4D data (465 frames, 33 keypoints, each with 2D (x, y) coordinates)
model.add(InputLayer(input_shape=(465, 33, 2)))

# Reshape layer to flatten the (33, 2) keypoints into 66 for each frame
model.add(Reshape((465, 33 * 2)))

# Bidirectional LSTM layers to capture temporal dependencies
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.5))  # Dropout to prevent overfitting
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))

# Output layer for classification (softmax activation for classification tasks)
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Display the model architecture
model.summary()

# Train the model
history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val),
                    epochs=30, batch_size=32)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')

# Predict the classes for the validation set
y_val_pred = np.argmax(model.predict(X_val), axis=-1)
y_val_true = np.argmax(y_val, axis=-1)

# Classification report
print(classification_report(y_val_true, y_val_pred, target_names=[f'Class {i}' for i in range(num_classes)]))