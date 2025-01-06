
# Real-Time Sign Language Translator 

## Overview
This project leverages computer vision and AI to translate sign language gestures into text, supporting smoother communication for the deaf and hard-of-hearing community. Through real-time gesture recognition, this system accurately identifies hand shapes and movements, facilitating interaction in diverse settings.

The system captures sign language gestures via a webcam, using hand landmark detection to classify gestures through a trained machine learning model. The model recognizes the alphabet (A-Z), along with unique gestures for backspace, space, and clear text. It includes special adjustments for gestures requiring movement, like "J" and "Z" in ASL, enhancing accuracy and usability.

<p align="center">––––––––––––––––––––––––––––––––––––––––––––</p>

## Requirements
Install the necessary dependencies:

- **Mediapipe**: `0.10.14`
- **OpenCV (cv2)**: `4.10.0`
- **Scikit-learn**: `1.5.2`
- **customtkinter**: 5.2.1

Install them with the following command:

**pip install mediapipe==0.10.14 opencv-python==4.10.0.84 scikit-learn==1.5.2 customtkinter==5.2.1**


## Mediapipe
Mediapipe is used for hand landmark detection. By extracting the x and y coordinates of each hand landmark, these features are passed to the classifier for gesture recognition, ensuring precise and reliable gesture tracking.

## CustomTkinter
CustomTkinter provides a sleek, dark-themed user interface. The application interface displays real-time webcam feeds and recognized gestures, with options to adjust the speed of recognition using a delay setting.

## Dataset
Data for sign language alphabets were collected, with 1,000 images per letter. Certain letters, such as "J" and "Z", require specific adaptations to account for motion. A `collect_data.py` module automates data collection, capturing labeled images of each hand gesture. Landmark extraction is performed by `create_data.py`, standardizing coordinates for model training.

The project also includes custom gestures for:
- **Backspace**: Deletes the last character
- **Clear All**: Clears all text
- **Space**: Adds a space between words


## Model: Random Forest Classifier for Gesture Recognition

The Random Forest Classifier, chosen for its robust handling of decision trees, is well-suited to gesture recognition. With accurate predictions, the model distinguishes between gesture patterns using hand landmark data.

### Model Details:
- **Feature Extraction**:
  - Hand landmarks are extracted via Mediapipe, with coordinates normalized for consistency across distances and hand positions.
- **Data Preparation**:
  - Landmarks and labels are processed into training and testing datasets to ensure high recognition accuracy. Each unique ASL and ISL gesture is captured as a distinct landmark pattern.
- **Training**:
  - The Random Forest Classifier achieves over 99% accuracy on validation data. Training is managedfor 'Indian Sign Language' and 'American Sign Language' by trainI.py and `train_classifierA.py` respectively, which outputs a 'ISL_model.p' and `ASL_model.p` model files loaded for real-time prediction.
- **Real-Time Prediction**:
  - `applicationI.py` and applicationA.py files handles real-time prediction respectively, loading the trained model and interpreting hand landmarks into text characters.

## Real-Time Detection

The `detection.py` module captures video from the webcam, processing each frame to extract hand landmarks and predict corresponding sign language characters.

- **Adjustable Delay**:
  - A customizable delay, managed by a counter, allows users to adjust the pace of sign language to text conversion. By setting the delay, users can control how quickly the application registers gestures, improving usability during sentence construction.

## Usage
To run the translator application:
1. Connect your webcam.
2. Run `app.py` to start real-time sign language translation.
3. Detected gestures are displayed as text, allowing sentence formation in real time. Gestures for backspace, space, and clear text enhance the user experience.

We can apply the same process to any sign language by considering Indian Sign Language and American Sign Language individually.



