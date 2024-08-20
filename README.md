# Object Detection using YOLOv8 and ONNX

This repository contains the implementation of a real-time object detection system built using the YOLOv8 model and ONNX Runtime. The system is designed to detect and classify multiple objects within images and video streams with high accuracy and speed. Leveraging the power of YOLOv8 and the efficiency of ONNX Runtime, this project offers a robust solution for applications requiring real-time object detection.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Environment Setup](#environment-setup)
  - [Building the Project](#building-the-project)
- [Usage](#usage)
  - [Image Detection](#image-detection)
  - [Video Detection](#video-detection)
- [Project Structure](#project-structure)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Object detection is a fundamental task in computer vision that involves identifying and localizing objects within images or video frames. It is widely used in various domains such as surveillance, autonomous driving, industrial automation, and more. This project implements a real-time object detection system using the YOLOv8 model, a state-of-the-art deep learning model known for its speed and accuracy in detecting multiple objects.

YOLOv8, the latest in the YOLO (You Only Look Once) series, improves upon its predecessors with optimized network architecture and advanced training techniques. Combined with ONNX Runtime, a high-performance inference engine, this project is capable of running efficient and fast object detection on both CPUs and GPUs. The system utilizes OpenCV for image processing, providing a complete pipeline from data acquisition to result visualization.

## Features

- **Real-Time Object Detection**: Capable of detecting multiple object classes in both static images and live video streams with high speed and accuracy.
- **High Performance**: Utilizes ONNX Runtime for efficient model inference, with support for GPU acceleration to achieve real-time processing.
- **Modular Design**: The system is modular, allowing easy customization and integration of different models or processing techniques.

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

#### 1. Install OpenCV

OpenCV is required for handling image and video processing. You can install it via a package manager or build it from source.

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

ONNX Runtime is the engine used to run the YOLOv8 model. You need to build it from source to use it in a C++ project.

```bash
# Clone the ONNX Runtime repository
git clone --recursive https://github.com/microsoft/onnxruntime
cd onnxruntime

# Build ONNX Runtime
# For CPU:
./build.sh --config Release --build_shared_lib --parallel --use_openmp
# For GPU (CUDA):
./build.sh --config Release --build_shared_lib --parallel --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda

# Install ONNX Runtime
cd build/Linux/Release
sudo make install
```

#### 3. Install Ultralytics YOLOv8

YOLOv8 by Ultralytics is used for training and exporting the model to ONNX format.

**Install Ultralytics YOLOv8:**
```bash
pip install ultralytics
```

**Export YOLOv8 model to ONNX:**
```python
from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO('yolov8m.pt')  # You can choose different sizes like yolov8n, yolov8s, etc.

# Export the model to ONNX format
model.export(format='onnx')
```

### Building the Project

After setting up the environment, you can build the project using CMake as follows:

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

Once the project is built, you can use the executable to perform object detection on images or video streams.

### Image Detection

To perform object detection on a static image:

```bash
./your-executable-name path/to/image.jpg
```

Example:

```bash
./object_detection ../data/image.jpg
```

### Video Detection

To perform object detection on a video file:

```bash
./your-executable-name path/to/video.mp4
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

## Future Work

There are several areas where this project could be extended or improved:

- **Expand Dataset**: Increase detection accuracy by expanding the training dataset with more diverse examples.
- **Enhance Model**: Explore the integration of other models, such as those specialized in object tracking or pose estimation, to broaden the system’s capabilities.
- **Optimize Inference Speed**: Further optimize the system for even faster inference, especially on resource-constrained hardware.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have suggestions for improvements or new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This README provides a detailed overview of the project, including what the program does, how to set up the necessary environment, and how to build and run the project. Feel free to modify it further to suit your project's specifics.
