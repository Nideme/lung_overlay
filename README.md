# lungfibrose-overlay

# Lung Overlay on Webcam Video

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [License](#license)

## Introduction
This project overlays an animation of lungs (`beta.mov`) onto live video captured from a webcam. It utilizes OpenCV for video processing, MediaPipe for pose detection, and NumPy for numerical calculations. The application aims to provide a visual representation of lungs over a person's chest in real-time video.

## Installation
Ensure you have Python 3 or higher, Mediapipe, OpenVc installed on your system. You can then install the required dependencies via pip:
Python 3 Installation:

Most systems come with Python 2 pre-installed. To install Python 3, you can use a package manager like apt (for Ubuntu/Debian) or brew (for macOS).
For Ubuntu/Debian:
sql
Copy code
sudo apt update
sudo apt install python3
For macOS (assuming Homebrew is installed):
sql
Copy code
brew update
brew install python3
NumPy Installation:

NumPy is a popular library for numerical computing in Python.
You can install NumPy via pip, Python's package installer.
Copy code
pip3 install numpy
OpenCV Installation:

OpenCV is a library for computer vision and image processing.
Install OpenCV using pip:
Copy code
pip3 install opencv-python
Mediapipe Installation:

Mediapipe is a framework for building multimodal applied machine learning pipelines.
Similarly, you can install Mediapipe via pip:
Copy code
pip3 install mediapipe
Make sure you have administrative privileges (sudo or administrator rights) to install packages globally. After executing these commands, the respective libraries should be installed on your system and ready to be used in your Python projects.

## Usage
To run the application, navigate to the project directory in your terminal and execute the following command:
python3 app.py 
or 
python3 (name of file) 



## Features
- Real-time overlay of lung animations on a live webcam feed.
- Dynamic adjustment of the overlay based on the person's pose detected using MediaPipe.
- Smooth transition of overlay using a moving average filter.

## Dependencies
- Python 3.x
- NumPy
- OpenCV (opencv-python)
- MediaPipe

## Configuration
No additional configuration is required to run the project as described. However, adjustments can be made within the code to change the overlay's appearance, such as its size and position, based on the person's pose.

## Documentation
The main functionalities are documented within the code through comments, explaining key sections and functions for processing the video, detecting poses, and overlaying the animation.

## Examples
Upon running the application, it will automatically access the default webcam, overlay the `beta.mov` lung animation on the detected person, and display the output in real-time.

## Troubleshooting
- Ensure all dependencies are correctly installed and latest Versions.
- Check your webcam permissions if the application fails to access the video feed.
- For issues related to the overlay not appearing correctly, verify the paths and formats of the input files.

## Contributors
[Hannah Hachemer, Mario Hachemer]

## License
[MIT License.]


