# Test Plan: Project Setup and Dependencies

## Objective
Verify that the development environment is correctly set up with all required dependencies for the human tracking and behavior analysis system.

## Test Components

### 1. Environment Setup Tests
- Verify Python 3.9+ installation
- Test virtual environment creation
- Confirm appropriate CUDA/GPU drivers (if applicable)
- Check OS compatibility (Linux/Windows/macOS)

### 2. Dependency Installation Tests
- Verify installation of core packages:
  - OpenCV (cv2)
  - NumPy
  - MediaPipe
  - ONNX Runtime
  - PyTorch (for YOLOv8 model)
  - ByteTrack dependencies
- Check version compatibility between packages
- Verify optional dependencies (FastAPI, Matplotlib)

### 3. System Resource Validation
- Test available memory allocation
- Check CPU capabilities
- Validate GPU memory and capabilities (if used)
- Measure baseline system performance

### 4. Project Structure Tests
- Verify project directory structure
- Confirm module imports work correctly
- Check configuration file loading
- Test logging system

## Conda Environment Setup

### Create Conda Environment
```bash
# Create a new conda environment with Python 3.9
conda create -n human_tracking python=3.9
conda activate human_tracking

# Install core dependencies
conda install -c conda-forge opencv numpy matplotlib
conda install -c pytorch pytorch torchvision cudatoolkit=11.3  # for GPU support
# or: conda install -c pytorch pytorch torchvision cpuonly  # for CPU only

# Install via pip for packages not available in conda
pip install mediapipe
pip install onnxruntime  # or onnxruntime-gpu if using GPU
pip install ultralytics  # for YOLOv8
pip install fastapi uvicorn  # for API if needed

# ByteTrack dependencies
pip install cython
pip install 'git+https://github.com/ifzhang/ByteTrack.git'
pip install lap  # for Linear Assignment Problem solver
```

### Verify Conda Environment
```bash
# List all installed packages with versions
conda list

# Test import of key packages
python -c "import cv2; import numpy; import mediapipe; import onnxruntime; \
          import torch; print('All core packages imported successfully')"
```

### Environment Export for Reproducibility
```bash
# Export environment specification for sharing with team
conda env export > environment.yml
```

## Test Procedures

### Basic Environment Test
```python
def test_environment():
    # Test Python version
    import sys
    assert sys.version_info >= (3, 8), "Python 3.8+ required"
    
    # Test critical dependencies
    import cv2
    import numpy as np
    import onnxruntime
    
    # Print versions for verification
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"NumPy Version: {np.__version__}")
    print(f"ONNX Runtime Version: {onnxruntime.__version__}")
    
    # Test GPU availability if applicable
    providers = onnxruntime.get_available_providers()
    print(f"Available ONNX providers: {providers}")
    
    return True
```

### MediaPipe Availability Test
```python
def test_mediapipe():
    import mediapipe as mp
    
    # Test MediaPipe models initialization
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    print(f"MediaPipe Version: {mp.__version__}")
    
    return True
```

### YOLOv8 Model Test
```python
def test_yolo_model():
    try:
        from ultralytics import YOLO
        # Test model loading (without actual inference)
        model = YOLO("yolov8n.pt")
        print("YOLOv8 model loaded successfully")
        return True
    except ImportError:
        print("Ultralytics package not found, install with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"Error loading YOLOv8 model: {e}")
        return False
```

## Expected Outcomes
- All dependencies should install without conflicts
- All import tests should pass without exceptions
- System resources should meet minimum requirements
- Project structure tests should confirm proper organization

## Failure Conditions
- Missing or incompatible packages
- Insufficient system resources
- Incorrect environment configuration
- Module import errors 