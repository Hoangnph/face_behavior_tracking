# Environment Setup Guide

This guide explains how to set up the development environment for the Human Tracking project.

## Prerequisites

- **Python 3.9+**: The project requires Python 3.9 or newer.
- **Conda**: We recommend using Conda for managing the environment and dependencies.
- **GPU (Optional)**: While not required, a GPU with CUDA support will significantly improve performance.

## Automated Setup

We provide automated setup scripts for both Unix-based systems (Linux/macOS) and Windows.

### For Linux/macOS

1. Open a terminal
2. Navigate to the project root directory
3. Run the setup script:

```bash
./scripts/setup.sh
```

### For Windows

1. Open a Command Prompt or PowerShell window
2. Navigate to the project root directory
3. Run the setup script:

```batch
scripts\setup.bat
```

## Manual Setup

If the automated setup doesn't work for your system, you can manually set up the environment:

1. Create a conda environment:

```bash
conda env create -f environment.yml
```

2. Activate the environment:

```bash
conda activate human_tracking
```

3. Install ByteTrack:

```bash
pip install git+https://github.com/ifzhang/ByteTrack.git
```

4. Download the YOLOv8 model (it will automatically download when first imported):

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

5. Verify the environment:

```bash
python scripts/setup_environment.py
```

## Verifying Installation

To verify that your environment is correctly set up, run the test script:

```bash
# Activate the environment first
conda activate human_tracking

# Run the tests
pytest tests/test_environment.py -v
```

## Troubleshooting

### GPU Not Detected

If you have a compatible GPU but it's not being detected:

1. Ensure you have the proper CUDA drivers installed
2. Check that the CUDA version is compatible with PyTorch (11.3 recommended)
3. Try reinstalling PyTorch with GPU support:

```bash
conda install -c pytorch pytorch torchvision cudatoolkit=11.3
```

### MediaPipe Models Failing

MediaPipe can be sensitive to specific versions. If you encounter issues:

```bash
pip uninstall mediapipe
pip install mediapipe==0.10.0  # Try a specific version
```

### Package Conflicts

If you encounter package conflicts, try creating a fresh environment:

```bash
conda deactivate
conda env remove -n human_tracking
conda env create -f environment.yml
``` 