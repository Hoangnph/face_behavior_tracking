# Test Plan: Detection Models Integration

## Objective
Verify the proper integration and performance of face and person detection models (MediaPipe Face Detection and YOLOv8-n) within the system using the project's sample datasets.

## Test Data Sources
- **Face Detection**: Use faces from `data/faces/known_customers` and `data/faces/employees` directories
- **Person Detection**: Use the sample video from `data/videos/sample_2.mp4`
- **Performance Testing**: All benchmarks should be measured on `data/videos/sample_2.mp4`

## Important Note
Please refer to the comprehensive test requirements document for detailed verification procedures and acceptance criteria:
[Updated Test Requirements](../docs/task_requirements/updated_test_plans.md)

These requirements must be followed for all detection module testing, including the use of specific data sources and personal verification procedures according to the @big-project.mdc rule.

## Test Components

### 1. MediaPipe Face Detection Tests
- Test model initialization
- Verify face detection on sample images from `data/faces`
- Test detection confidence thresholds
- Measure detection performance (speed and accuracy)

### 2. YOLOv8-n Person Detection Tests
- Test model loading and initialization
- Verify person detection on `data/videos/sample_2.mp4`
- Test detection with multiple persons
- Measure inference speed and memory usage

### 3. ONNX Runtime Integration Tests
- Test model conversion to ONNX format
- Verify inference using ONNX Runtime
- Compare performance between original and ONNX models on `data/videos/sample_2.mp4`
- Test quantized model performance

### 4. Detection Robustness Tests
- Test detection under varying lighting conditions using sample datasets
- Verify detection with occlusion
- Test detection at different distances
- Measure false positive and false negative rates

## Test Procedures

### MediaPipe Face Detection Test
```python
def test_mediapipe_face_detection(image_path=None):
    import cv2
    import mediapipe as mp
    import time
    import numpy as np
    import glob
    import os
    
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    
    # Default to using data/faces if no image path provided
    if not image_path:
        face_dirs = ["data/faces/known_customers", "data/faces/employees"]
        face_images = []
        for directory in face_dirs:
            if os.path.exists(directory):
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    face_images.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
                    
        if not face_images:
            print("Error: No face images found in the specified directories")
            return False
            
        # Use first image as sample or select random samples
        import random
        image_path = random.choice(face_images)
    
    # Load test image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return False
    
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process image with different confidence thresholds
    thresholds = [0.3, 0.5, 0.7]
    results = {}
    
    for threshold in thresholds:
        with mp_face_detection.FaceDetection(min_detection_confidence=threshold) as face_detection:
            start_time = time.time()
            detection_result = face_detection.process(image_rgb)
            process_time = time.time() - start_time
            
            # Count detections
            face_count = 0
            if detection_result.detections:
                face_count = len(detection_result.detections)
                
            results[threshold] = {
                "faces_detected": face_count,
                "process_time_ms": process_time * 1000
            }
            
            # Save annotated image for verification
            output_dir = "data/output/verification/face_detection"
            os.makedirs(output_dir, exist_ok=True)
            
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, f"threshold_{threshold}_{filename}")
            
            annotated_image = image.copy()
            if detection_result.detections:
                for detection in detection_result.detections:
                    mp_drawing.draw_detection(annotated_image, detection)
            
            cv2.imwrite(output_path, annotated_image)
    
    # Print results
    for threshold, data in results.items():
        print(f"Threshold {threshold}: {data['faces_detected']} faces detected in {data['process_time_ms']:.2f} ms")
    
    print(f"\nTest results saved to data/output/verification/face_detection")
    print("\nPlease manually verify the detection results for confirmation (following @big-project.mdc rule)")
    
    return True
```

### YOLOv8 Person Detection Test
```python
def test_yolo_person_detection(video_path="data/videos/sample_2.mp4"):
    try:
        from ultralytics import YOLO
        import cv2
        import time
        import numpy as np
        import os
        
        # Load YOLOv8 model
        start_time = time.time()
        model = YOLO("yolov8n.pt")
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return False
            
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        print(f"Testing on video: {video_path}")
        print(f"Video properties: {frame_count} frames, {fps} FPS, {duration:.2f} seconds")
        
        # Create output directory
        output_dir = "data/output/verification/person_detection"
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize counters
        processed_frames = 0
        frames_with_detections = 0
        total_persons = 0
        total_inference_time = 0
        
        # Sample frames at regular intervals to test
        sample_interval = max(1, int(frame_count / 20))  # Sample ~20 frames
        
        for frame_idx in range(0, frame_count, sample_interval):
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            processed_frames += 1
            
            # Run inference
            start_time = time.time()
            results = model(frame)
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # Filter for person detections (class 0 in COCO dataset)
            person_detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # Person class
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].tolist()
                        person_detections.append({
                            "confidence": conf,
                            "bbox": xyxy
                        })
            
            # Save annotated images for verification
            has_persons = len(person_detections) > 0
            if has_persons:
                frames_with_detections += 1
                total_persons += len(person_detections)
                
                # Draw detections on image
                annotated_frame = frame.copy()
                for det in person_detections:
                    x1, y1, x2, y2 = map(int, det["bbox"])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"{det['confidence']:.2f}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save frame 
                if processed_frames <= 10:  # Save first 10 frames with detections
                    output_path = os.path.join(output_dir, f"frame_{frame_idx}.jpg")
                    cv2.imwrite(output_path, annotated_frame)
            
            print(f"Frame {frame_idx}: {len(person_detections)} persons detected in {inference_time*1000:.2f} ms")
        
        # Close video
        cap.release()
        
        # Compute statistics
        avg_inference_time = total_inference_time / processed_frames * 1000 if processed_frames > 0 else 0
        detection_rate = frames_with_detections / processed_frames * 100 if processed_frames > 0 else 0
        avg_persons_per_frame = total_persons / frames_with_detections if frames_with_detections > 0 else 0
        
        print("\nYOLOv8 Person Detection Results:")
        print(f"Processed {processed_frames} frames")
        print(f"Frames with person detections: {frames_with_detections} ({detection_rate:.1f}%)")
        print(f"Average persons per frame: {avg_persons_per_frame:.2f}")
        print(f"Average inference time: {avg_inference_time:.2f} ms")
        print(f"Estimated FPS: {1000/avg_inference_time:.2f}")
        print(f"\nSample frames saved to {output_dir}")
        print("\nPlease manually verify the detection results for confirmation (following @big-project.mdc rule)")
        
        return True
    except ImportError:
        print("Error: Ultralytics package not installed")
        return False
    except Exception as e:
        print(f"Error during YOLOv8 detection: {e}")
        return False
```

### ONNX Model Test
```python
def test_onnx_model(model_path="models/onnx/yolov8n.onnx", video_path="data/videos/sample_2.mp4"):
    import onnxruntime as ort
    import numpy as np
    import cv2
    import time
    import os
    
    try:
        # Check if model and video exist
        if not os.path.exists(model_path):
            print(f"Error: ONNX model not found at {model_path}")
            return False
            
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return False
        
        # Create ONNX Runtime session
        start_time = time.time()
        session = ort.InferenceSession(model_path)
        load_time = time.time() - start_time
        print(f"ONNX model loaded in {load_time:.2f} seconds")
        
        # Get model metadata
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"Model input name: {input_name}, shape: {input_shape}")
        
        # Determine input dimensions
        input_height = input_shape[2] if len(input_shape) > 2 else 640
        input_width = input_shape[3] if len(input_shape) > 3 else 640
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False
        
        # Create output directory
        output_dir = "data/output/verification/onnx_inference"
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize counters
        processed_frames = 0
        total_inference_time = 0
        
        # Sample frames at regular intervals
        sample_interval = max(1, int(frame_count / 10))  # Sample ~10 frames
        
        for frame_idx in range(0, frame_count, sample_interval):
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            processed_frames += 1
            
            # Preprocess frame
            input_tensor = cv2.resize(frame, (input_width, input_height))
            input_tensor = input_tensor.transpose(2, 0, 1)  # HWC to CHW
            input_tensor = input_tensor.astype(np.float32) / 255.0  # Normalize to [0,1]
            input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
            
            # Run inference
            start_time = time.time()
            outputs = session.run(None, {input_name: input_tensor})
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            print(f"Frame {frame_idx}: Inference completed in {inference_time*1000:.2f} ms")
            
            # Save original frame for reference
            if processed_frames <= 5:  # Save first 5 frames
                output_path = os.path.join(output_dir, f"frame_{frame_idx}.jpg")
                cv2.imwrite(output_path, frame)
        
        # Close video
        cap.release()
        
        # Compute statistics
        avg_inference_time = total_inference_time / processed_frames * 1000 if processed_frames > 0 else 0
        
        print("\nONNX Inference Results:")
        print(f"Processed {processed_frames} frames")
        print(f"Average inference time: {avg_inference_time:.2f} ms")
        print(f"Estimated FPS: {1000/avg_inference_time:.2f}")
        print(f"\nSample frames saved to {output_dir}")
        print("\nPlease manually verify the inference results for confirmation (following @big-project.mdc rule)")
        
        return True
    except Exception as e:
        print(f"Error during ONNX model testing: {e}")
        return False
```

## Expected Outcomes
- Face detection should identify faces with at least 90% accuracy on the `data/faces` dataset
- Person detection should identify people in `data/videos/sample_2.mp4` with at least 80% accuracy
- ONNX models should run at least 20% faster than original models on `data/videos/sample_2.mp4`
- Detection should work in real-time (minimum 15 FPS) for 640x480 resolution

## Verification Requirements (Following @big-project.mdc rule)
For a task to be considered complete, it must pass both automated tests and personal verification:

1. **Automated Testing**:
   - All unit tests must pass
   - Performance benchmarks must meet specified thresholds
   - No critical errors or exceptions during processing

2. **Personal Verification**:
   - Visual inspection of detection results on sample data
   - Confirmation of detection accuracy in challenging cases
   - Verification of proper bounding box placement
   - Sign-off on performance metrics meeting project requirements

## Test Result Documentation
Test results must be documented in the following format:

```
# Detection Module Test Report

## Test Environment
- Hardware: [Processor, RAM, GPU]
- OS: [Operating System]
- Date: [Test Date]

## Test Results
- MediaPipe Face Detection: [PASS/FAIL]
  - Accuracy: [%]
  - Performance: [FPS]
  
- YOLOv8 Person Detection: [PASS/FAIL]
  - Accuracy: [%]
  - Performance: [FPS]
  
- ONNX Runtime Integration: [PASS/FAIL]
  - Performance improvement: [%]
  - Memory usage reduction: [%]

## Personal Verification
I have personally reviewed the test results and verified that:
- [ ] Face detection correctly identifies faces in the sample dataset
- [ ] Person detection accurately tracks people in the sample video
- [ ] The performance meets project requirements
- [ ] All known issues have been documented

Verified by: [Your Name]
Date: [Verification Date]
```

## Failure Conditions
- Model loading errors or missing model files
- Excessive inference time making real-time processing infeasible (below 10 FPS)
- Poor detection accuracy (below 70% for faces, below 60% for persons)
- Memory leaks during continuous inference 