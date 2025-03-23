# Test Plan: Camera/Video Input Pipeline

## Objective
Verify that the system can properly capture, process, and manage video streams from various sources (webcam, video files, RTSP streams).

## Test Components

### 1. Video Source Compatibility Tests
- Test webcam access and capture
- Test video file loading (.mp4, .avi, etc.)
- Test RTSP stream connection (if applicable)
- Verify handling of different resolutions

### 2. Frame Processing Tests
- Test frame extraction from video sources
- Verify frame resize operations
- Test frame rate management
- Verify color space conversions (BGR to RGB)

### 3. Video Stream Management Tests
- Test stream initialization and closure
- Verify proper resource allocation/deallocation
- Test stream interruption handling
- Test reconnection logic for network streams

### 4. Performance Tests
- Measure frame processing time
- Test maximum sustainable FPS
- Verify memory usage during stream processing
- Test handling of high-resolution inputs

## Test Procedures

### Basic Camera Access Test
```python
def test_camera_access():
    import cv2
    import time
    
    # Try to open default camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return False
    
    # Read a few frames to ensure stability
    frames_read = 0
    start_time = time.time()
    
    for _ in range(30):  # Try to read 30 frames
        ret, frame = cap.read()
        if not ret:
            break
        frames_read += 1
        
        # Optional: Display the frame
        # cv2.imshow('Camera Test', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    # Release resources
    cap.release()
    # cv2.destroyAllWindows()
    
    duration = time.time() - start_time
    fps = frames_read / duration if duration > 0 else 0
    
    print(f"Camera test: Read {frames_read} frames in {duration:.2f} seconds ({fps:.2f} FPS)")
    
    return frames_read > 0
```

### Video File Test
```python
def test_video_file(video_path):
    import cv2
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {frame_count} frames")
    
    # Sample frames (beginning, middle, end)
    test_frames = [0, frame_count // 2, frame_count - 1]
    
    for frame_idx in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Failed to read frame at position {frame_idx}")
            continue
            
        print(f"Successfully read frame at position {frame_idx}, shape: {frame.shape}")
    
    # Release resources
    cap.release()
    
    return True
```

### Frame Processing Pipeline Test
```python
def test_frame_processing():
    import cv2
    import numpy as np
    import time
    
    # Create a test frame or use camera
    # Option 1: Create synthetic frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw something on the frame
    cv2.rectangle(test_frame, (100, 100), (300, 300), (0, 255, 0), 2)
    
    # Option 2: Capture from camera
    # cap = cv2.VideoCapture(0)
    # ret, test_frame = cap.read()
    # cap.release()
    
    # Test pipeline operations
    start_time = time.time()
    
    # 1. Resize frame
    resized = cv2.resize(test_frame, (320, 240))
    
    # 2. Convert to RGB (for ML models)
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # 3. Apply basic preprocessing
    blurred = cv2.GaussianBlur(rgb_frame, (5, 5), 0)
    
    # 4. Convert back to BGR for display
    processed = cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR)
    
    duration = time.time() - start_time
    
    print(f"Frame processing pipeline completed in {duration*1000:.2f} ms")
    print(f"Original shape: {test_frame.shape}, Processed shape: {processed.shape}")
    
    # Optional: Display the original and processed frames
    # cv2.imshow('Original', test_frame)
    # cv2.imshow('Processed', processed)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return True
```

## Expected Outcomes
- All video sources should be accessible
- Frame extraction should work consistently
- Processing operations should maintain image integrity
- Performance should meet real-time requirements

## Failure Conditions
- Camera access permission issues
- Missing or corrupted video files
- Excessive frame processing time
- Memory leaks during continuous operation 