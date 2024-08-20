# Object Detection using YOLOv8 and ONNX

This repository contains the code for an object detection system based on YOLOv8 and ONNX Runtime. The system is capable of detecting various object classes in static images and video streams with high accuracy and efficiency.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
  - [Environment Setup](#environment-setup)
  - [Build the Project](#build-the-project)
- [Usage](#usage)
  - [Image Detection](#image-detection)
  - [Video Detection](#video-detection)
- [Project Structure](#project-structure)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project implements a real-time object detection system using the YOLOv8 model, leveraging the power of ONNX Runtime for fast inference. The system supports detection on both static images and video streams, and it can identify multiple object categories in real-time.

## Features

- **Real-time Object Detection**: Detects multiple object classes in images and videos.
- **High Performance**: Utilizes ONNX Runtime for efficient inference, especially when using GPU acceleration.
- **Customizable**: Easily modify detection thresholds, skip frames, and more to suit specific needs.

## Installation

### Environment Setup

This project requires several dependencies, including OpenCV, ONNX Runtime, and Ultralytics' YOLOv8. Below are the detailed steps to set up the environment.

#### 1. Install OpenCV

You can either build OpenCV from source or install it using a package manager.

**Option 1: Install OpenCV using a package manager**

For Ubuntu:
```bash
sudo apt-get update
sudo apt-get install libopencv-dev python3-opencv
```

For macOS using Homebrew:
```bash
brew install opencv
```

**Option 2: Build OpenCV from source**

```bash
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules ..
make -j8
sudo make install
```

#### 2. Install ONNX Runtime

You can install ONNX Runtime via pip if you're planning to use it in Python, but since this project is in C++, you'll need to download and build it:

```bash
# Download ONNX Runtime
git clone --recursive https://github.com/microsoft/onnxruntime
cd onnxruntime

# Choose your desired build configuration (CPU/GPU)
# For CPU:
./build.sh --config Release --build_shared_lib --parallel --use_openmp
# For GPU:
./build.sh --config Release --build_shared_lib --parallel --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda

# Install ONNX Runtime
cd build/Linux/Release
sudo make install
```

#### 3. Install Ultralytics YOLOv8

YOLOv8 is provided by Ultralytics and can be installed using pip. However, for C++ integration, you typically download the model and convert it to ONNX format.

**Install Ultralytics YOLOv8 (Python):**
```bash
pip install ultralytics
```

**Export YOLOv8 model to ONNX:**
```python
from ultralytics import YOLO

# Load a pretrained YOLOv8 model (e.g., YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x)
model = YOLO('yolov8m.pt')

# Export the model to ONNX format
model.export(format='onnx')
```

### Build the Project

After setting up the environment, you can build the project using CMake:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repository-name.git
   cd your-repository-name
   ```

2. **Configure and build the project:**
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

## Usage

### Image Detection

To perform object detection on a static image:

```bash
./your-executable-name image_path
```

Example:

```bash
./object_detection ../data/image.jpg
```

### Video Detection

To perform object detection on a video file:

```bash
./your-executable-name video_path
```

Example:

```bash
./object_detection ../data/video.mp4
```

To perform object detection using a webcam (default is the first webcam device):

```bash
./your-executable-name 0
```

## Project Structure

```plaintext
.
├── CMakeLists.txt            # CMake configuration file
├── README.md                 # This file
├── src                       # Source files
│   ├── main.cpp              # Main entry point for the program
│   ├── yolov8_onnx.h         # Header file for YOLOv8 and ONNX integration
│   ├── yolov8_onnx.cpp       # Implementation of YOLOv8 and ONNX integration
│   ├── yolov8_utils.h        # Utility functions and structures
│   └── yolov8_utils.cpp      # Implementation of utility functions
├── data                      # Example images and videos for testing
└── models                    # YOLOv8 ONNX model file
```

## Results

The system demonstrates strong performance in real-time object detection across various test images and videos. Below are some examples of the results:

### Image Detection Example

![Image Detection Result](path/to/image_result.jpg)

### Video Detection Example

![Video Detection Result](path/to/video_result.jpg)

