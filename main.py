import cv2
import numpy as np
import streamlit as st
from PIL import Image

@st.cache_resource
def load_model():
    net = cv2.dnn.readNetFromCaffe(
        "models/deploy.prototxt",
        "models/res10_300x300_ssd_iter_140000.caffemodel"
    )
    return (net)

net = load_model()

st.title("Face Detector")
st.write("ResNet Model + SSD — OpenCV v3.3")
in_frame = st.camera_input("Camera")

if in_frame is not None:
    image_pil = Image.open(in_frame)
    frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
        swapRB=False
    )

    net.setInput(blob)
    detections = net.forward()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for i in range(detections.shape[2]):
        trust = detections[0, 0, i, 2]

        if trust < 0.5:
            continue
        
        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        x1, y1, x2, y2 = box.astype("int")

        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    st.image(frame_rgb, caption="Result", width=700)