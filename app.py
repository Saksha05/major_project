from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import torch
import pandas as pd
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from werkzeug.utils import secure_filename



# Initialize the Flask app
app = Flask(__name__)

# Configure upload and processed folders
UPLOAD_FOLDER = './uploads'
PROCESSED_FOLDER = './processed'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to upload form (index page)
@app.route('/')
def upload_form():
    return render_template('index.html')

# Route to handle video upload and processing
@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        
        if file.filename == '':
            return 'No selected file'
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)

            # Process the video
            output_video_path = os.path.join(app.config['PROCESSED_FOLDER'], 'output_shortened.mp4')
            dataset_path = os.path.join(app.config['PROCESSED_FOLDER'], 'shortened_frames_dataset.csv')
            process_video(video_path, output_video_path, dataset_path)

            # Return the processed video download link
            return render_template('download.html', video_link=output_video_path)

    # If the request is GET, render the upload page
    return render_template('upload.html')


@app.route('/query', methods=['POST'])
def query_process_video():
    color = request.form.get('color', 'white')

    input_video_path = os.path.join(app.config['PROCESSED_FOLDER'], 'output_shortened.mp4')
    query_output_video_path = os.path.join(app.config['PROCESSED_FOLDER'], 'query_based_output.mp4')

    query_based(input_video_path, query_output_video_path, color=color)

    # Check if the file exists before rendering the template
    if not os.path.exists(query_output_video_path):
        return "Query-based output video could not be created.", 500

    # Render the download page for query-based output
    return render_template('query_download.html', query_video_link='query_based_output.mp4')




def process_video_chunk(start_frame, end_frame, input_video_path, fps):
    cap = cv2.VideoCapture(input_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_data = []
    selected_frames = []
    skip_count = 2  # Number of frames to skip after processing a frame with detected movement
    frame_number = start_frame

    while frame_number < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = str(timedelta(seconds=int(frame_number / fps)))

        # Perform object detection
        results = model(frame)

        # Check for human detections (class id 0)
        humans_detected = False
        for *box, conf, cls in results.xyxy[0]:  # xyxy format
            if int(cls) == 0:  # 0 corresponds to 'person'
                x1, y1, x2, y2 = map(int, box)
                humans_detected = True

                # Display timestamp above the detected human position
                cv2.putText(
                    frame,
                    timestamp,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),  # Blue color for the timestamp
                    2
                )

        if humans_detected:
            selected_frames.append(frame)  # Store the frame with detected human
            frame_data.append({
                'frame_number': frame_number,
                'timestamp': timestamp
            })

            # Skip the next few frames
            for _ in range(skip_count):
                cap.read()
                frame_number += 1

        frame_number += 1

    cap.release()
    return frame_data, selected_frames


def process_video(input_video_path, output_video_path, dataset_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Couldn't open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: FPS value is 0. Manually setting to 30 FPS.")
        fps = 30  # Default to 30 FPS if OpenCV fails to retrieve FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    chunk_size = frame_count // os.cpu_count()
    chunks = [(i, min(i + chunk_size, frame_count)) for i in range(0, frame_count, chunk_size)]

    frame_data = []
    selected_frames = []

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_video_chunk, start, end, input_video_path, fps)
            for start, end in chunks
        ]
        for future in futures:
            data, frames = future.result()
            frame_data.extend(data)
            selected_frames.extend(frames)

    # Write selected frames to output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for frame in selected_frames:
        out.write(frame)
    out.release()

    # Save the dataset to CSV
    df = pd.DataFrame(frame_data)
    df.to_csv(dataset_path, index=False)



def query_based(input_video_path, output_video_path, color='white'):
    # Create VideoCapture object and get video properties
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Define solid color ranges in HSV
    color_ranges = {
        'white': (np.array([0, 0, 180], dtype=np.uint8), np.array([180, 40, 255], dtype=np.uint8)),  # Adjusted white range for accuracy
        'red': (np.array([0, 50, 50], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),
        'blue': (np.array([100, 50, 50], dtype=np.uint8), np.array([130, 255, 255], dtype=np.uint8)),
        'green': (np.array([50, 50, 50], dtype=np.uint8), np.array([70, 255, 255], dtype=np.uint8)),
        'black': (np.array([0, 0, 0], dtype=np.uint8), np.array([180, 255, 50], dtype=np.uint8)),
        # Add more colors if needed
    }

    # Query parameters
    query_object = 'person'  # Object type: 'person', 'car', 'bicycle', etc.

    # Get the HSV range for the query color
    color_range = color_ranges.get(color)
    if not color_range:
        raise ValueError(f"Color '{color}' not found in the defined color ranges.")

    def is_color_present(frame, box, color_range):
        """Check if the specified color is present in the region of interest."""
        x1, y1, x2, y2 = map(int, box)
        roi = frame[y1:y2, x1:x2]  # Region of interest (bounding box)

        # Ensure ROI is valid
        if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
            return False

        # Convert ROI to HSV for color detection
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, color_range[0], color_range[1])

        # Check if the target color occupies a significant portion
        color_pixels = cv2.countNonZero(mask)
        total_pixels = roi.shape[0] * roi.shape[1]
        return color_pixels / total_pixels > 0.15  # Match if >15% of the area is target color

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)

        # Filter detections based on query object type and color
        frame_written = False
        for *box, conf, cls in results.xyxy[0]:
            if model.names[int(cls)] == query_object:  # Check if object type matches
                if is_color_present(frame, box, color_range):
                    if not frame_written:  # Write the frame once if there's a match
                        out.write(frame)
                        frame_written = True

    # Release video objects
    cap.release()
    out.release()






@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)



if __name__ == '__main__':
    # Ensure necessary directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    app.run(debug=True)
