Here's a comprehensive GitHub README for your project:

---

# YOLO Object Detection: Two-Pin and Three-Pin Plug Detector

This project demonstrates how to train and evaluate a YOLO (You Only Look Once) object detection model to detect two-pin and three-pin plugs. The dataset is managed via Roboflow, and the YOLO model is trained using the Ultralytics library.

## Table of Contents

- [Overview](#overview)  
- [Requirements](#requirements)  
- [Setup](#setup)  
- [Dataset](#dataset)  
- [Training](#training)  
- [Validation](#validation)  
- [Prediction](#prediction)  
- [Results](#results)  
- [Acknowledgments](#acknowledgments)

---

## Overview

The objective of this project is to train a YOLOv11 model to detect and classify two-pin and three-pin plugs in images. The project uses the Ultralytics YOLO framework and Roboflow for dataset preparation.

---

## Requirements

- Python 3.8 or later
- Key Python libraries:
  - `ultralytics`
  - `roboflow`
  - `IPython`

Install all dependencies using the following commands:

```bash
pip install ultralytics
pip install roboflow
```

---

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/prithadash/Two-pin-Three-pin-plug-detector
   cd Two-pin-Three-pin-plug-detector
   ```

2. **Import Required Libraries**:
   The script imports the Ultralytics and Roboflow libraries to manage the YOLO environment and dataset.

3. **Authenticate Roboflow**:
   Ensure you have an API key from Roboflow. Replace `"b0V2HszsdbIBAhRGuz4c"` in the script with your API key.

---

## Dataset

The dataset is hosted on Roboflow and contains images of two-pin and three-pin plugs. To download the dataset:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="your_api_key")
project = rf.workspace("object-detection-jq94x").project("two-pin-three-pin-detector")
version = project.version(5)
dataset = version.download("yolov11")
```

---

## Training

To train the YOLO model on the dataset:

```bash
!yolo task=detect mode=train data={dataset.location}/data.yaml model="yolo11n.pt" epochs=60 imgsz=640
```

- **Key Parameters**:
  - `task=detect`: Specifies object detection.
  - `mode=train`: Indicates training mode.
  - `data`: Path to the dataset YAML file.
  - `model`: The YOLO model architecture.
  - `epochs`: Number of training epochs.
  - `imgsz`: Image size for training.

---

## Validation

To validate the model performance on the test dataset:

```bash
!yolo task=detect mode=val model="/content/runs/detect/train/weights/best.pt" data={dataset.location}/data.yaml
```

---

## Prediction

To run predictions on the test dataset:

```bash
!yolo task=detect mode=predict model="/content/runs/detect/train/weights/best.pt" conf=0.25 source={dataset.location}/test/images save=True
```

- The `predict` mode generates predictions for the test images.
- Results are saved in the `/content/runs/detect/predict/` folder.

---

## Results

### Key Visuals:
- **Confusion Matrix**:
  ![Confusion Matrix](runs/detect/train/confusion_matrix.png)
  
- **Label Distribution**:
  ![Label Distribution](runs/detect/train/labels.jpg)

- **Training Results**:
  ![Training Results](runs/detect/train/results.png)

- **Sample Predictions**:
  ![Sample Prediction 1](runs/detect/predict/image1.jpg)  
  ![Sample Prediction 2](runs/detect/predict/image2.jpg)

---

## Acknowledgments

- **Roboflow**: For dataset hosting and management.
- **Ultralytics**: For the YOLO framework.
- **IPython**: For enhanced visualization during training and evaluation.

---

Feel free to customize this README further based on your preferences! Let me know if you need additional sections or tweaks.

