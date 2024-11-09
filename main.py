import streamlit as st
import cv2
from ultralytics import YOLO, hub
import numpy as np
from PIL import Image

hub.login('7b3ebd78c549c86f4d8bbe2d09510be47a18a4d0f5')

model = YOLO('https://hub.ultralytics.com/models/9jCkYeiQfmVE6bIkpnkY')

# Streamlit app
st.title("Plantex Gen 1")

# Capture image from camera
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # Read the image from the camera buffer
    img = Image.open(img_file_buffer)
    img_array = np.array(img)

    # Run YOLO object detection
    results = model(img_array)

    # Draw bounding boxes on the image
    annotated_image = img_array.copy()
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, score, class_id = box[:6].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = model.names[int(class_id)]
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_image,
                f"{label} {score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    # Display the annotated image with detections
    st.image(annotated_image, caption="Detected objects", use_column_width=True)
