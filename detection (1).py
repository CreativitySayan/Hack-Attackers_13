import tensorflow as tf
import cv2
import numpy as np

# Load the saved TensorFlow model
model = tf.saved_model.load(r"C:\Users\SANJANA\Downloads\ssd_mobilenet_v2_320x320_coco17_tpu-8\ssd_mobilenet_v2_320x320_coco17_tpu-8\saved_model")

# Extract the model's serving signature
model_fn = model.signatures['serving_default']

# Category index with real-world heights of objects (in meters)
category_index = {
    1: {'id': 1, 'name': 'person', 'real_height': 1.7},  # Average human height in meters
    3: {'id': 3, 'name': 'car', 'real_height': 1.5},  # Approx car height
    4: {'id': 4, 'name': 'motorcycle', 'real_height': 1.0},
    8: {'id': 8, 'name': 'truck', 'real_height': 3.0},
    10: {'id': 10, 'name': 'traffic light', 'real_height': 2.5},
    18: {'id': 18, 'name': 'dog', 'real_height': 0.5},  # Approx dog height
    44: {'id': 44, 'name': 'bottle', 'real_height': 0.25},  # Bottle height
    61: {'id': 61, 'name': 'chair', 'real_height': 1.0},
    62: {'id': 62, 'name': 'couch', 'real_height': 0.8},
    63: {'id': 63, 'name': 'potted plant', 'real_height': 0.5},
    64: {'id': 64, 'name': 'bed', 'real_height': 0.6},
    67: {'id': 67, 'name': 'dining table', 'real_height': 0.75},
    72: {'id': 72, 'name': 'tv', 'real_height': 0.6},
    73: {'id': 73, 'name': 'laptop', 'real_height': 0.05},
    77: {'id': 77, 'name': 'cell phone', 'real_height': 0.01},
    82: {'id': 82, 'name': 'refrigerator', 'real_height': 1.8}
}

# Focal length estimation (in pixels)
KNOWN_FOCAL_LENGTH = 700  # You can calibrate this for your camera

# Function to detect objects
def detect_objects(frame, detection_model):
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]  # Add batch dimension

    # Run inference using the model
    detections = detection_model(input_tensor)

    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
    detection_scores = detections['detection_scores'][0].numpy()

    return detection_boxes, detection_classes, detection_scores

# Function to estimate distance based on object height
def estimate_distance(focal_length, real_height, apparent_height):
    distance = (real_height * focal_length) / apparent_height
    return distance

# Open webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert frame to RGB
    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect objects in the frame
    boxes, classes, scores = detect_objects(input_frame, model_fn)

    # Draw detected objects and estimate distance
    for i in range(len(scores)):
        if scores[i] > 0.5:  # Filter low-confidence detections
            class_id = classes[i]
            if class_id in category_index:
                box = boxes[i]
                (startY, startX, endY, endX) = box

                # Convert normalized box coordinates to pixel coordinates
                startX = int(startX * frame.shape[1])
                startY = int(startY * frame.shape[0])
                endX = int(endX * frame.shape[1])
                endY = int(endY * frame.shape[0])

                # Draw bounding box
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                # Calculate the apparent height of the object in pixels
                apparent_height = endY - startY

                # Get the real height of the detected object (from category_index)
                real_height = category_index[class_id]['real_height']

                # Estimate distance to the object
                distance = estimate_distance(KNOWN_FOCAL_LENGTH, real_height, apparent_height)

                # Display label, confidence, and distance
                label = f"{category_index[class_id]['name']}: {int(scores[i] * 100)}% Distance: {distance:.2f} meters"
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display frame with detections and distances
    cv2.imshow('Object Detection with Distance', frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
