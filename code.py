# Import required libraries
import io
import os
import cv2
import numpy as np
from flask import Flask, render_template, request

# Initialize the Flask app
app = Flask(__name__)

# Define a function to analyze a video and extract its labels
def analyze_video(video_path):
    # Load the pre-trained MobileNet SSD model
    model_path = 'frozen_inference_graph.pb'
    config_path = 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
    net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
    # Read the video file
    cap = cv2.VideoCapture(video_path)
    # Specify the input image size and scale factor
    input_size = (300, 300)
    scale_factor = 1/127.5
    # Initialize the labels list
    labels = []
    # Loop through all the frames in the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize the frame to the input size and preprocess it
        frame = cv2.resize(frame, input_size)
        blob = cv2.dnn.blobFromImage(frame, scale_factor, input_size, (127.5, 127.5, 127.5), True, False)
        # Pass the frame through the model and retrieve the results
        net.setInput(blob)
        detections = net.forward()
        # Loop through all the detections in the current frame
        for i in range(detections.shape[2]):
            # Extract the class ID and confidence score of the current detection
            class_id = int(detections[0, 0, i, 1])
            score = detections[0, 0, i, 2]
            # Check if the current detection is a person with high confidence
            if class_id == 1 and score > 0.9:
                # Add the 'person' label to the list of labels
                labels.append('person')
                # Stop processing the current frame if a person is detected
                break
    # Release the video file and return the list of labels
    cap.release()
    return labels

# Define a function to search for videos based on their content
def search_video_by_label(video_dir, label):
    # Loop through all the videos in the specified directory
    for filename in os.listdir(video_dir):
        if filename.endswith('.mp4'):
            # Analyze the video and extract its labels
            video_path = os.path.join(video_dir, filename)
            labels = analyze_video(video_path)
            # Check if the specified label is in the list of labels
            if label in labels:
                # Return the path to the matching video
                return video_path
    # If no matching video is found, return None
    return None

# Define a function to handle the search form submission
@app.route('/', methods=['GET', 'POST'])
def search_video():
    # Handle the form submission
    if request.method == 'POST':
        # Get the label from the form data
        label = request.form['label']
        # Search for videos with the specified label
        video_dir = 'path/to/your/video/directory'
        video_path = search_video_by_label(video_dir, label)
        # Display the path to the matching video, if found
        if video_path is not None:
            result = 'Matching video found: ' + video_path
        else:
            result = 'No matching video found.'
# Run the app
if __name__ == '__main__':
    app.run(debug=True)