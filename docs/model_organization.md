# Model Organization

This document describes the organization of machine learning models used in the human tracking and behavior analysis system.

## Directory Structure

All models are stored in the `models/` directory with the following structure:

```
models/
│
├── yolo/                   # YOLO models for object detection
│   ├── yolov8n.pt          # YOLOv8-nano model for person detection
│   └── ...
│
├── mediapipe/              # MediaPipe models (added as needed)
│   └── ...
│
├── face/                   # Face-related models
│   ├── recognition/        # Face recognition models
│   ├── landmarks/          # Facial landmark models
│   └── ...
│
└── pose/                   # Pose estimation models
    └── ...
```

## Model Details

### YOLO Models

- **yolov8n.pt**: YOLOv8-nano model for general object detection, primarily used for person detection in our system.
  - Size: ~6.2MB
  - Classes: 80 (COCO dataset classes)
  - Primary objects of interest: person (class 0)
  - Resolution: Dynamic (usually 640x640)
  - Framework: PyTorch

## Usage in Code

Models should be referenced using relative paths from the project root:

```python
# Example usage in detection module
YOLO_MODEL_PATH = os.path.join("models", "yolo", "yolov8n.pt")
detector = YOLO(YOLO_MODEL_PATH)
```

## Adding New Models

When adding new models to the system:

1. Place the model file in the appropriate subdirectory
2. Document the model in this file with relevant details
3. Update any code references to use the correct path
4. Consider adding a model loading utility if the model requires specific initialization

## Model Versioning

Model files should be versioned and tracked using Git LFS (Large File Storage) when available. For larger models that cannot be stored in the repository, document the exact version and download location in this file. 