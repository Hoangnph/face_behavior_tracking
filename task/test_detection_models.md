# Test Plan: Detection Models Integration

## Objective
Verify the proper integration and performance of face and person detection models (MediaPipe Face Detection and YOLOv8-n) within the system.

## Test Components

### 1. MediaPipe Face Detection Tests
- Test model initialization
- Verify face detection on sample images
- Test detection confidence thresholds
- Measure detection performance (speed and accuracy)

### 2. YOLOv8-n Person Detection Tests
- Test model loading and initialization
- Verify person detection on sample images
- Test detection with multiple persons
- Measure inference speed and memory usage

### 3. ONNX Runtime Integration Tests
- Test model conversion to ONNX format
- Verify inference using ONNX Runtime
- Compare performance between original and ONNX models
- Test quantized model performance

### 4. Detection Robustness Tests
- Test detection under varying lighting conditions
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
    
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    
    # Load test image or capture from camera
    if image_path:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return False
    else:
        # Capture from camera
        cap = cv2.VideoCapture(0)
        ret, image = cap.read()
        cap.release()
        
        if not ret:
            print("Error: Could not capture image from camera")
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
            
            # Optional: Draw detections on image for visualization
            # annotated_image = image.copy()
            # if detection_result.detections:
            #     for detection in detection_result.detections:
            #         mp_drawing.draw_detection(annotated_image, detection)
            # cv2.imshow(f"Threshold {threshold}", annotated_image)
            # cv2.waitKey(0)
    
    # Print results
    for threshold, data in results.items():
        print(f"Threshold {threshold}: {data['faces_detected']} faces detected in {data['process_time_ms']:.2f} ms")
    
    # cv2.destroyAllWindows()
    return True
```

### YOLOv8 Person Detection Test
```python
def test_yolo_person_detection(image_path=None):
    try:
        from ultralytics import YOLO
        import cv2
        import time
        import numpy as np
        
        # Load YOLOv8 model
        start_time = time.time()
        model = YOLO("yolov8n.pt")
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Load test image or capture from camera
        if image_path:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image from {image_path}")
                return False
        else:
            # Capture from camera
            cap = cv2.VideoCapture(0)
            ret, image = cap.read()
            cap.release()
            
            if not ret:
                print("Error: Could not capture image from camera")
                return False
        
        # Run inference
        start_time = time.time()
        results = model(image)
        inference_time = time.time() - start_time
        
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
        
        print(f"YOLOv8 detection: {len(person_detections)} persons detected in {inference_time*1000:.2f} ms")
        for i, det in enumerate(person_detections):
            print(f"  Person {i+1}: Confidence {det['confidence']:.2f}, Bbox {det['bbox']}")
        
        # Optional: Draw detections on image for visualization
        # for det in person_detections:
        #     x1, y1, x2, y2 = map(int, det["bbox"])
        #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(image, f"{det['confidence']:.2f}", (x1, y1-10), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # cv2.imshow("YOLOv8 Person Detection", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
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
def test_onnx_model(onnx_model_path):
    import onnxruntime as ort
    import numpy as np
    import cv2
    import time
    
    try:
        # Create ONNX Runtime session
        start_time = time.time()
        session = ort.InferenceSession(onnx_model_path)
        load_time = time.time() - start_time
        print(f"ONNX model loaded in {load_time:.2f} seconds")
        
        # Get model metadata
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"Model input name: {input_name}, shape: {input_shape}")
        
        # Create a dummy input tensor
        input_height = input_shape[2] if len(input_shape) > 2 else 640
        input_width = input_shape[3] if len(input_shape) > 3 else 640
        
        # Prepare a test image
        test_image = np.zeros((input_height, input_width, 3), dtype=np.uint8)
        # Resize and normalize image to match model input requirements
        # This part depends on specific model requirements
        input_tensor = cv2.resize(test_image, (input_width, input_height))
        input_tensor = input_tensor.transpose(2, 0, 1)  # HWC to CHW
        input_tensor = input_tensor.astype(np.float32) / 255.0  # Normalize to [0,1]
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
        
        # Run inference
        start_time = time.time()
        outputs = session.run(None, {input_name: input_tensor})
        inference_time = time.time() - start_time
        
        print(f"ONNX inference completed in {inference_time*1000:.2f} ms")
        print(f"Output shapes: {[output.shape for output in outputs]}")
        
        return True
    except Exception as e:
        print(f"Error during ONNX model testing: {e}")
        return False
```

## Expected Outcomes
- Face detection should identify faces with reasonable accuracy
- Person detection should identify people in various poses
- ONNX models should run efficiently with reduced memory usage
- Detection should work in real-time for standard resolutions

## Failure Conditions
- Model loading errors or missing model files
- Excessive inference time making real-time processing infeasible
- Poor detection accuracy or high false positive rate
- Memory leaks during continuous inference 