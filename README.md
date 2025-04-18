# 🩰 Automated Ballet Move Classification using Machine Learning

This project explores the intersection of video analysis and machine learning to classify ballet moves from solo performance footage. It uses pose estimation and temporal features to train a supervised learning model that predicts dance classes with meaningful accuracy.

---

## 📂 Project Structure

### Part 1: Feature Extraction

We begin by extracting keypoints from videos using the **MediaPipe Pose** model.

1. **Initialization**  
   The MediaPipe Pose model is initialized to analyze each frame for body landmarks.

2. **Keypoint Extraction**  
   Each frame is converted to RGB and analyzed. If pose landmarks are found, `(x, y, z)` coordinates and visibility are recorded.

3. **Temporal Features**  
   Velocity and acceleration between frames are calculated to understand movement patterns over time.

4. **Saving Features**  
   All extracted data is saved as `.pkl` files for reuse during training and evaluation.

---

### Part 2: Feature Visualization

To confirm extraction accuracy and better understand the data:

1. **Keypoint Display**  
   Keypoints are plotted using **Matplotlib** on video frames as red dots.

2. **Frame Processing**  
   Frames are processed to RGB and landmarks extracted per frame.

3. **Background Subtraction**  
   Implemented using **MOG2**, this isolates the dancer from static backgrounds.

4. **Interactive Playback**  
   Visualizations are shown in real-time, highlighting movement and pose transitions dynamically.

---

### Part 3: Building the Machine Learning Model

We use a **Temporal Convolutional Network (TCN)** to classify dance moves based on extracted features.

#### 📊 Data Preparation

- **Libraries:** NumPy, TensorFlow, Matplotlib  
- **Loading Features:** Extracted `.pkl` feature files are loaded with corresponding labels  
- **Padding:** Features are padded to ensure uniform input shape (`min_length = 3000` rows)

#### 🏷️ Label Encoding

- String labels are converted into integer values  
- Padded arrays are truncated to maintain consistency

#### 🔀 Data Splitting

- Train/test split at 80:20 ratio  
- Labels are one-hot encoded for classification tasks

#### 🧠 Model Architecture

- **Masking Layer:** Ignores padding during training  
- **Conv1D Layers:** Extract temporal patterns  
- **Dropout Layers:** Prevent overfitting  
- **Dense Softmax Output:** Predicts the dance move class

#### 🛠️ Training & Evaluation

- Early stopping based on validation loss  
- Trained over 80 epochs  
- Model evaluated on test data (loss, accuracy)  
- Predictions are decoded and reported

#### 📈 Results

- A **classification report** is generated showing:
  - Precision, Recall, F1-score
- Results are stored in a DataFrame and visualized to interpret model performance across dance classes

---

## 📦 Dataset

- Source: [UJAnnChor Dataset](https://github.com/dvanderhaar/UJAnnChor)  
- Size: ~10.5 GB  
- Annotated ballet solo videos, categorized by move  
- Used for both training and evaluation
---

## 🔍 Use Cases

- Dancer training & feedback systems  
- Performance evaluation tools  
- Virtual reality and motion analysis  
- Education in performing arts

---

## 🚀 Technologies Used

- Python  
- MediaPipe  
- OpenCV  
- NumPy & Pandas  
- TensorFlow/Keras  
- Matplotlib  
- Scikit-learn  

---

## 📌 Status

> ✅ Model trained  
> ✅ Features extracted and visualized  
> 🔄 Further optimization in progress
