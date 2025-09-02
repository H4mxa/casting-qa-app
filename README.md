# Casting Defects Detection App

This is web application that uses a YOLOv8 model to detect casting defects in uploaded images.

## Features

*   Upload one or more images for defect detection.
*   Adjustable confidence and IoU thresholds for the YOLOv8 model.
*   Selectable image size for inference.
*   Displays the original image and the image with bounding boxes around the detected defects.
*   Download the results as a ZIP file.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/casting-qa-dashboard.git
    ```
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## Dependencies

The following libraries are required to run the application:

*   streamlit
*   ultralytics
*   torch
*   torchvision
*   numpy
*   pillow
*   opencv-python-headless
