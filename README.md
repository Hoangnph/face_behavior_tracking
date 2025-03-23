# Human Identification and Behavior Tracking System

A lightweight system for human identification and behavior tracking that works in both real-time and offline scenarios using camera input.

## Features

- Face and person detection using MediaPipe and YOLOv8-n
- Person tracking with ByteTrack algorithm
- Pose estimation and behavior analysis
- Real-time processing optimized for CPU
- Support for both live camera and video inputs

## Project Structure

```
├── data/                   # Data files (not included in repository)
├── src/                    # Source code
│   ├── detection/          # Detection models (Face, Person)
│   ├── tracking/           # Person tracking implementations
│   ├── pose/               # Pose estimation and keypoints
│   ├── behavior/           # Behavior analysis components
│   ├── utils/              # Utility functions
│   └── visualization/      # Visualization tools
├── task/                   # Project task tracking and test plans
├── tests/                  # Unit and integration tests
└── docs/                   # Documentation
```

## Setup and Installation

### Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n human_tracking python=3.9
conda activate human_tracking

# Install core dependencies
conda install -c conda-forge opencv numpy matplotlib
conda install -c pytorch pytorch torchvision

# Install additional dependencies
pip install mediapipe onnxruntime ultralytics

# ByteTrack dependencies
pip install cython
pip install 'git+https://github.com/ifzhang/ByteTrack.git'
```

For detailed setup instructions, see [Test Setup](task/test_setup.md).

## Development Process

This project follows a Test-Driven Development (TDD) approach:

1. First, review the test plan for a component
2. Implement the feature to pass those tests
3. Refactor and optimize the implementation

For task tracking and progress, see [Task Overview](task/task_overview.md).

## License

[MIT License](LICENSE) 