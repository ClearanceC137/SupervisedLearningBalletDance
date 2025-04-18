# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 19:21:21 2024

@author: clear
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Masking, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import seaborn as sns
import pandas as pd  # Importing pandas

# Directory where the .npy files are stored
features_dir = r'C:\Users\clear\Desktop\BSC Computer Science hons\Second Semester\Big Data\features'
# Initialize lists to hold the features and labels
features = []
labels = []

# Loop through each file in the directory
for filename in os.listdir(features_dir):
    if filename.endswith('.pkl'):  # Change to .pkl for pickle files
        # Load the feature vector from the pickle file
        with open(os.path.join(features_dir, filename), 'rb') as f:
            feature = pickle.load(f)
        class_label = filename[:3]  # Get the first three letters
        # Append the feature and class label to the respective lists
        features.append(feature)
        labels.append(class_label)
        del feature
        del class_label

labels = np.array(labels)  # Shape will be (num_samples,)

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
min_rows = 3000
features = pad_to_num_rows(features,min_rows)

#features = np.array(features)

# Step 1: Convert string labels to integer labels
unique_labels = np.unique(labels)  # Get unique string labels
label_map = {label: idx for idx, label in enumerate(unique_labels)}  # Create a mapping

# Convert the labels to integer using the mapping
int_labels = np.array([label_map[label] for label in labels])
del label_map
# Step 2: Define the minimum length (frames)
#min_length = min(len(f) for f in features)  # Find the min length from your features
# Step 3: Pad your sequences to the minimum length
padded_features = np.array([f[:min_rows] for f in features])  # Truncate to min_length
del features
#features = np.array(features)
# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_features, int_labels, test_size=0.2, random_state=42)

# Step 5: One-hot encode your labels
num_classes = len(unique_labels)  # Number of unique classes after mapping
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Check shapes
print(f"Train Features Shape: {X_train.shape}")
print(f"Train Labels Shape: {y_train.shape}")
print(f"Test Features Shape: {X_test.shape}")
print(f"Test Labels Shape: {y_test.shape}")

# Define TCN model with Masking
def build_tcn_model(input_shape, num_classes):
    model = Sequential()
    
    # Masking layer to ignore padded zeros
    model.add(Masking(mask_value=0., input_shape=input_shape))  # Mask padding values (assumes 0 is the pad value)
    
    # TCN layers
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='causal'))
    model.add(Dropout(0.3))
    
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='causal'))
    model.add(Dropout(0.3))
    
    model.add(Conv1D(filters=256, kernel_size=2, activation='relu', padding='causal'))
    model.add(Dropout(0.3))
    
    # Flatten and Dense layer
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

# Prepare data for training
# Assuming X_train and X_test are your training and testing features
#maxlen = 60  # Adjust maxlen as needed

#X_train_padded = pad_sequences(X_train, padding='post', dtype='float32', maxlen=maxlen)  # Pad sequences to the maximum length
#X_test_padded = pad_sequences(X_test, padding='post', dtype='float32', maxlen=maxlen)

# Build the model
input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, num_features)
num_classes = y_train.shape[1]  # Number of classes
model = build_tcn_model(input_shape, num_classes)

# Compile the model using the custom loss function
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
del padded_features
history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test),
                    epochs=80, 
                    batch_size=32, 
                    callbacks=[early_stopping])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')


# Save the model
model.save('my_model.h5')  # Save the model to a file named 'my_model.h5'


# Make predictions on the test set
y_pred = model.predict(X_test)

# Convert predictions from one-hot encoding to class labels
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Assuming y_true_classes and y_pred_classes are defined
report = classification_report(y_true_classes, y_pred_classes, output_dict=True)

# Create a DataFrame from the report
report_df = pd.DataFrame(report).transpose()

# Select the metrics you want to plot (precision, recall, f1-score)
metrics = ['precision', 'recall', 'f1-score']
report_df = report_df[metrics].iloc[:-3]  # Exclude 'accuracy', 'macro avg', and 'weighted avg'

# Plot the metrics for each class
report_df.plot(kind='bar', figsize=(10, 6))
plt.title('Classification Report Metrics')
plt.xlabel('Classes')
plt.ylabel('Scores')
plt.xticks(rotation=90)
plt.legend(loc='best')
plt.show()

# Save the model
#model.save('my_model.h5')  # Save the model to a file named 'my_model.h5'
# Assuming `history` contains your training history
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.show()



