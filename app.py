from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError


@st.cache_resource
def load_face_detector() -> cv2.CascadeClassifier:
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if detector.empty():
        raise RuntimeError("The OpenCV face-detection model could not be loaded.")
    return detector


def detect_faces(
    image: Image.Image, detector: cv2.CascadeClassifier
) -> tuple[Image.Image, int]:
    image_array = np.array(image)
    grayscale = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    annotated = image_array.copy()
    for index, (x, y, width, height) in enumerate(faces, start=1):
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 200, 80), 3)
        cv2.putText(
            annotated,
            f"Face {index}",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 80),
            2,
            cv2.LINE_AA,
        )

    return Image.fromarray(annotated), len(faces)


st.set_page_config(page_title="Face detector", page_icon=":material/face:")
st.title(":material/face: Face detector")
st.write("Upload a photo to find and highlight faces.")

uploaded_file = st.file_uploader(
    "Choose a photo",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is None:
    st.info("Choose a JPG, PNG, or WebP image to get started.")
else:
    try:
        original_image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("This file could not be opened as an image. Try another photo.")
    else:
        try:
            face_detector = load_face_detector()
        except (cv2.error, RuntimeError) as error:
            st.error(f"Face detection is unavailable: {error}")
        else:
            result_image, face_count = detect_faces(original_image, face_detector)

            if face_count:
                st.success(f"Found {face_count} face(s).")
            else:
                st.info("No faces were detected. Try a clear, well-lit photo.")

            original_column, result_column = st.columns(2)
            with original_column:
                st.image(
                    original_image,
                    caption="Original photo",
                    alt="Original uploaded photo",
                )
            with result_column:
                st.image(
                    result_image,
                    caption="Detection result",
                    alt=f"Uploaded photo with {face_count} detected faces highlighted",
                )

            output = BytesIO()
            result_image.save(output, format="PNG")
            st.download_button(
                "Download result",
                data=output.getvalue(),
                file_name="face-detection.png",
                mime="image/png",
                icon=":material/download:",
            )
