# Detection Module Documentation

## Overview

The Detection Module provides human detection capabilities for the tracking system, with support for face detection and full-body person detection. The module is designed to be flexible, efficient, and easy to integrate into the video processing pipeline. This document provides an overview of the module's architecture, components, features, and performance characteristics.

## Architecture

The Detection Module follows a modular design with the following key components:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Detection      │     │  ONNX           │     │  Detection      │
│  Scheduler      │ ──▶ │  Runtime        │ ──▶ │  Results        │
│                 │     │  Integration     │     │  Processing     │
└────────┬────────┘     └─────────────────┘     └─────────────────┘
         │
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Face           │     │  Person         │
│  Detection      │     │  Detection      │
│  (MediaPipe)    │     │  (YOLOv8)       │
└─────────────────┘     └─────────────────┘
```

### Core Components

1. **Detection Scheduler**: Manages multiple detection algorithms and schedules them based on performance requirements. Allows running detectors at different frequencies to optimize performance.

2. **Face Detection**: Implements MediaPipe Face Detection for accurate and efficient face detection.

3. **Person Detection**: Implements YOLOv8 for full-body person detection with high accuracy.

4. **ONNX Runtime Integration**: Provides optimized inference using ONNX Runtime for improved performance.

5. **Base Detector**: Abstract base class defining the common interface for all detectors.

## Features

### Face Detection

- **Model**: MediaPipe Face Detection
- **Resolution**: Optimized for 640x480 input
- **Performance**: 17.41 FPS on CPU
- **Accuracy**: 96.9% detection rate on test dataset
- **Key Features**:
  - Multi-face detection
  - Confidence thresholding
  - BGR to RGB conversion for MediaPipe compatibility
  - Bounding box normalization

### Person Detection

- **Model**: YOLOv8n
- **Resolution**: 640x640 input size
- **Performance**: 10.05 FPS on CPU
- **Accuracy**: 66.7% detection rate on test video
- **Key Features**:
  - Class filtering (person class only)
  - Confidence thresholding (0.15)
  - Aspect ratio-preserving resize
  - Scale factor adjustment (0.75)

### ONNX Runtime Integration

- **Model**: YOLOv8n converted to ONNX format
- **Performance**: 22.38 FPS on CPU (122.7% faster than native YOLOv8)
- **Key Features**:
  - Input size adaptation
  - Dynamic batch size support
  - Error handling for symbolic dimensions
  - Efficient pre- and post-processing

### Detection Scheduler

- **Performance**: 6.87 FPS combined
- **Key Features**:
  - Detector frequency management
  - Named detector selection
  - Result caching for improved performance
  - Unified detection output format

## Implementation Details

### BaseDetector Interface

All detectors implement the `BaseDetector` interface, which provides a common structure:

```python
class BaseDetector:
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        
    def detect(self, frame):
        # To be implemented by subclasses
        pass
        
    def _preprocess_image(self, image):
        # Common preprocessing steps
        pass
```

### Face Detection Implementation

```python
class MediaPipeFaceDetector(BaseDetector):
    def __init__(self, confidence_threshold=0.5, model_selection=1):
        super().__init__(confidence_threshold)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=confidence_threshold
        )
    
    def detect(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.face_detection.process(rgb_frame)
        
        # Extract and return detections
        detections = []
        if results.detections:
            for detection in results.detections:
                # Convert to bounding box format
                # Add to detections list
        
        return detections
```

### Person Detection Implementation

```python
class YOLOPersonDetector(BaseDetector):
    def __init__(self, confidence_threshold=0.15, model_path="yolov8n.pt", size_factor=0.75):
        super().__init__(confidence_threshold)
        self.model = YOLO(model_path)
        self.size_factor = size_factor
        self.input_size = (640, 640)
    
    def detect(self, frame):
        # Resize frame maintaining aspect ratio
        resized_frame = self._resize_with_aspect_ratio(frame)
        
        # Run detection
        results = self.model(resized_frame, conf=self.confidence_threshold)
        
        # Extract person detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 0:  # Person class
                    # Extract and add detection
        
        return detections
        
    def _resize_with_aspect_ratio(self, image):
        # Resize image maintaining aspect ratio
        # Add padding if needed
```

### ONNX Integration Implementation

```python
class ONNXDetector(BaseDetector):
    def __init__(self, model_path, confidence_threshold=0.15, input_size=(640, 640)):
        super().__init__(confidence_threshold)
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        
        # Handle input dimensions
        input_shape = self.session.get_inputs()[0].shape
        self.input_height, self.input_width = self._get_input_dimensions(input_shape)
    
    def detect(self, frame):
        # Preprocess image
        input_tensor = self._preprocess_image(frame)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Process outputs
        detections = self._process_outputs(outputs, frame)
        
        return detections
```

### Detection Scheduler Implementation

```python
class DetectionScheduler:
    def __init__(self, detectors=None, default_detector=None):
        self.detectors = detectors or {}
        self.default_detector = default_detector
        self.frame_count = 0
        self.frequencies = {}
        self.last_results = {}
    
    def detect(self, frame, detector_name=None, force_new_detection=False):
        # Select detector
        detector = self._get_detector(detector_name)
        
        # Check if we need to run detection this frame
        should_detect = self._should_detect(detector_name)
        
        if should_detect or force_new_detection:
            # Run detection
            results = detector.detect(frame)
            self.last_results[detector_name] = results
        else:
            # Return cached results
            results = self.last_results.get(detector_name, [])
        
        return results
    
    def set_frequency(self, detector_name, frequency):
        # Set how often a detector should run
        self.frequencies[detector_name] = frequency
```

## Performance Metrics

| Detector | Resolution | FPS | Detection Rate | Notes |
|----------|------------|-----|----------------|-------|
| MediaPipe Face | 640x480 | 17.41 | 96.9% | Low resource usage |
| YOLOv8 Person | 640x640 | 10.05 | 66.7% | Higher accuracy with lower threshold |
| ONNX Person | 640x640 | 22.38 | 66.7% | Same accuracy as YOLOv8 but faster |
| Detection Scheduler | 640x480 | 6.87 | - | Combined performance |

## Known Limitations

1. **Person Detection Accuracy**: Current accuracy (66.7%) is below the target threshold (70%)
2. **Performance**: Overall performance is below real-time requirements for combined detection
3. **Resource Usage**: High memory usage on resource-constrained devices
4. **Framework Dependencies**: Multiple framework dependencies increase deployment complexity

## Optimization Opportunities

1. **Model Optimization**:
   - Quantization for faster inference
   - Pruning for smaller model size
   - Model distillation for better efficiency

2. **Hardware Acceleration**:
   - GPU acceleration with CUDA or OpenCL
   - Neural Processing Unit (NPU) support
   - Apple Silicon optimizations

3. **Algorithm Improvements**:
   - Adaptive resolution based on scene complexity
   - Motion-based detection scheduling
   - Temporal consistency filtering

## Conclusion

The Detection Module provides a solid foundation for human detection in the tracking system. Face detection exceeds accuracy requirements, while person detection requires further optimization to meet both accuracy and performance targets. The ONNX integration demonstrates significant performance improvements and provides a clear path for further optimization. 