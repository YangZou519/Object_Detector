# Object Detection using YOLOv8 and ONNX

This repository contains the implementation of a real-time object detection system built using the YOLOv8 model and ONNX Runtime. The system is designed to detect and classify multiple objects within images and video streams with high accuracy and speed. Leveraging the power of YOLOv8 and the efficiency of ONNX Runtime, this project offers a robust solution for applications requiring real-time object detection.

## Introduction

Object detection is a fundamental task in computer vision that involves identifying and localizing objects within images or video frames. It is widely used in various domains such as surveillance, autonomous driving, industrial automation, and more. This project implements a real-time object detection system using the YOLOv8 model, a state-of-the-art deep learning model known for its speed and accuracy in detecting multiple objects.

YOLOv8, the latest in the YOLO (You Only Look Once) series, improves upon its predecessors with optimized network architecture and advanced training techniques. Combined with ONNX Runtime, a high-performance inference engine, this project is capable of running efficient and fast object detection on both CPUs and GPUs. The system utilizes OpenCV for image processing, providing a complete pipeline from data acquisition to result visualization.

## Requirements

To run this program, the following dependencies are required:

- **C++11 or later**: The program is written in modern C++.
- **OpenCV**: Used for image processing, video capture, and display functions.
- **ONNX Runtime**: The inference engine used to run the YOLOv8 model efficiently.
- **CMake**: Required to build the project from source.
- **Ultralytics YOLOv8**: The Python package for YOLOv8, needed to export the model to ONNX format.
- 
## Installation

### Environment Setup

To set up the environment for running this project, follow the steps below:

#### 1. Visual Studio

1. **Go to the Visual Studio Downloads page**: [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/).

2. **Click "Free download"** under the **Visual Studio Community** section (or choose the version you need).

3. **Run the downloaded installer** and select the workloads you want.

4. **Click "Install"** to begin the installation.

### 2. Downloading OpenCV:

1. **Go to the OpenCV Releases page**: [OpenCV Releases](https://opencv.org/releases/).

2. **Download the latest Windows version** by clicking on the appropriate link under the "Windows" section.

3. **Extract the downloaded zip file** and follow the included instructions to set up OpenCV on your system.

### 3. Downloading ONNX Runtime:

1. **Go to the ONNX Runtime GitHub page**: [ONNX Runtime GitHub](https://github.com/microsoft/onnxruntime).

2. **Download the latest release** by navigating to the "Releases" section and selecting the appropriate version for your platform (Windows).

3. **Follow the installation instructions** provided in the release to set up ONNX Runtime on your system.
