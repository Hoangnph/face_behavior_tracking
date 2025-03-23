# Test Plan: Person Tracking

## Objective
Verify the correct implementation and performance of the ByteTrack-based person tracking system, ensuring accurate ID assignment and maintenance across frames when processing videos from the project dataset.

## Important Note
Please refer to the comprehensive test requirements document for detailed verification procedures and acceptance criteria:
[Updated Test Requirements](../docs/task_requirements/updated_test_plans.md)

These requirements must be followed for all tracking module testing, including the use of specific data sources (`data/videos/sample_2.mp4`) and personal verification procedures according to the @big-project.mdc rule.

## Test Data Sources
- **Primary Test Video**: Use `data/videos/sample_2.mp4` for all tracking performance tests
- **Tracking Evaluation**: Generate visualizations of tracking results for manual verification
- **Performance Testing**: All benchmarks should be measured on `data/videos/sample_2.mp4`

## Test Components

### 1. ByteTrack Implementation Tests
- Test ByteTrack initialization with different parameters
- Verify tracking performance with real detections from `data/videos/sample_2.mp4`
- Test handling of missed detections and reappearances
- Measure tracking accuracy against manually verified tracks

### 2. Kalman Filter Tests
- Test state prediction accuracy on moving people in `data/videos/sample_2.mp4`
- Verify filter behavior with missing detections
- Test motion model under different movement scenarios
- Measure uncertainty estimation accuracy

### 3. Person Re-identification Tests
- Test ID maintenance during occlusion events in `data/videos/sample_2.mp4`
- Verify ID re-assignment after long disappearance
- Test handling of similar-looking individuals
- Measure ID switch frequency

### 4. Performance Tests
- Measure tracking overhead per frame on `data/videos/sample_2.mp4`
- Test scaling with number of tracked objects
- Verify real-time performance with full pipeline
- Measure memory usage during extended tracking

## Test Procedures

### Tracking Test on Sample Video
```python
def test_tracking_on_sample_video(video_path="data/videos/sample_2.mp4"):
    try:
        import cv2
        import numpy as np
        import os
        import time
        from yolox.tracker.byte_tracker import BYTETracker
        from ultralytics import YOLO
        
        # Create output directory
        output_dir = "data/output/verification/tracking"
        os.makedirs(output_dir, exist_ok=True)
        
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
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps
        
        print(f"Testing on video: {video_path}")
        print(f"Video properties: {frame_count} frames, {fps} FPS, {width}x{height}, {duration:.2f} seconds")
        
        # Initialize YOLOv8 for person detection
        model = YOLO("yolov8n.pt")
        
        # Initialize ByteTrack tracker
        tracker = BYTETracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=fps
        )
        
        # Define colors for tracks (for visualization)
        import random
        random.seed(42)  # For reproducibility
        colors = {}
        
        # Create video writer for output
        output_video_path = os.path.join(output_dir, "tracked_sample.mp4")
        output_video = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        
        # Initialize tracking stats
        processed_frames = 0
        total_detections = 0
        total_tracks = 0
        id_switches = 0
        
        # Track through video
        sample_interval = max(1, int(frame_count / 300))  # Process at most 300 frames
        last_frame_ids = set()
        
        for frame_idx in range(0, frame_count, sample_interval):
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            processed_frames += 1
            
            # Run YOLOv8 detection
            results = model(frame)
            
            # Convert to ByteTrack format
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # Person class
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].tolist()
                        detections.append([*xyxy, conf, cls])
            
            detections = np.array(detections)
            total_detections += len(detections)
            
            # Update tracker
            if len(detections) > 0:
                online_targets = tracker.update(
                    detections,
                    [height, width],
                    [height, width]
                )
                
                # Get current frame IDs
                current_frame_ids = set()
                
                # Draw tracks
                for target in online_targets:
                    track_id = target.track_id
                    current_frame_ids.add(track_id)
                    
                    # Calculate potential ID switches
                    if processed_frames > 1:
                        if track_id not in last_frame_ids and len(online_targets) <= len(last_frame_ids):
                            id_switches += 1
                    
                    # Get track bounding box
                    tlwh = target.tlwh
                    x, y, w, h = map(int, tlwh)
                    
                    # Assign consistent color for this ID
                    if track_id not in colors:
                        colors[track_id] = (
                            random.randint(50, 255),
                            random.randint(50, 255),
                            random.randint(50, 255)
                        )
                    
                    # Draw bounding box and ID
                    color = colors[track_id]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(
                        frame, 
                        f"ID: {track_id}", 
                        (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        color, 
                        2
                    )
                
                total_tracks += len(online_targets)
                last_frame_ids = current_frame_ids
            
            # Write to output video
            output_video.write(frame)
            
            # Save sample frames for report
            if processed_frames <= 5 or processed_frames % 50 == 0:
                output_path = os.path.join(output_dir, f"frame_{frame_idx}_tracking.jpg")
                cv2.imwrite(output_path, frame)
            
            # Print progress
            if processed_frames % 20 == 0:
                print(f"Processed {processed_frames} frames, {len(online_targets) if len(detections) > 0 else 0} active tracks")
        
        # Release resources
        cap.release()
        output_video.release()
        
        # Calculate stats
        avg_detections = total_detections / processed_frames if processed_frames > 0 else 0
        avg_tracks = total_tracks / processed_frames if processed_frames > 0 else 0
        
        # Report results
        print("\nTracking Test Results:")
        print(f"Processed {processed_frames} frames from {video_path}")
        print(f"Average detections per frame: {avg_detections:.2f}")
        print(f"Average tracks per frame: {avg_tracks:.2f}")
        print(f"Estimated ID switches: {id_switches}")
        print(f"\nOutput video saved to: {output_video_path}")
        print(f"Sample frames saved to: {output_dir}")
        print("\nPlease manually verify the tracking results for confirmation (following @big-project.mdc rule)")
        
        return True
    except ImportError as e:
        print(f"Required library not installed: {str(e)}")
        return False
    except Exception as e:
        print(f"Error during tracking test: {str(e)}")
        return False
```

### ByteTrack Basic Tracking Test (Synthetic Data)
```python
def test_bytetrack_basic():
    try:
        import numpy as np
        from yolox.tracker.byte_tracker import BYTETracker
        import time
        import os
        
        # Create output directory
        output_dir = "data/output/verification/tracking_synthetic"
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize tracker
        tracker = BYTETracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=30
        )
        
        # Create synthetic detection sequence
        frame_count = 60
        detection_sequence = []
        
        # Person 1: Moving from left to right
        person1_x = 100
        person1_y = 200
        person1_width = 80
        person1_height = 180
        
        # Person 2: Moving from right to left
        person2_x = 500
        person2_y = 300
        person2_width = 70
        person2_height = 170
        
        # Create detection sequence
        for frame in range(frame_count):
            # Update positions
            person1_x += 5  # Move right
            person2_x -= 4  # Move left
            
            # Create bounding boxes
            detections = []
            
            # Person 1 (always visible)
            x1 = max(0, person1_x - person1_width // 2)
            y1 = max(0, person1_y - person1_height // 2)
            x2 = x1 + person1_width
            y2 = y1 + person1_height
            conf = 0.9
            detections.append([x1, y1, x2, y2, conf, 0])  # Class 0 for person
            
            # Person 2 (occasionally occluded)
            if frame < 20 or (frame >= 30 and frame < 50):
                x1 = max(0, person2_x - person2_width // 2)
                y1 = max(0, person2_y - person2_height // 2)
                x2 = x1 + person2_width
                y2 = y1 + person2_height
                conf = 0.85
                detections.append([x1, y1, x2, y2, conf, 0])
            
            detection_sequence.append(np.array(detections))
        
        # Run tracking on the sequence
        tracking_results = []
        track_ids_by_frame = []
        start_time = time.time()
        
        for frame_idx, dets in enumerate(detection_sequence):
            if len(dets) > 0:
                online_targets = tracker.update(dets, [600, 800], [600, 800])
                online_tlwhs = []
                online_ids = []
                
                for target in online_targets:
                    tlwh = target.tlwh
                    track_id = target.track_id
                    online_tlwhs.append(tlwh)
                    online_ids.append(track_id)
                
                tracking_results.append({
                    'frame': frame_idx,
                    'boxes': online_tlwhs,
                    'ids': online_ids
                })
                track_ids_by_frame.append(set(online_ids))
        
        process_time = time.time() - start_time
        
        # Analyze tracking performance
        unique_ids = set()
        id_switches = 0
        
        for result in tracking_results:
            for track_id in result['ids']:
                unique_ids.add(track_id)
        
        # Check ID consistency
        for i in range(1, len(track_ids_by_frame)):
            prev_ids = track_ids_by_frame[i-1]
            curr_ids = track_ids_by_frame[i]
            
            # If person2 reappears (frame 30), we expect a new ID
            if i == 30:
                                continue
                
            # Otherwise, IDs should be maintained
            for prev_id in prev_ids:
                if prev_id not in curr_ids and i != 20 and i != 50:
                    id_switches += 1
        
        # Save results as text file for verification
        with open(os.path.join(output_dir, "tracking_results.txt"), "w") as f:
            f.write(f"ByteTrack Synthetic Test Results\n")
            f.write(f"Processed {frame_count} frames in {process_time:.3f} seconds\n")
            f.write(f"Total unique track IDs: {len(unique_ids)}\n")
            f.write(f"Total ID switches: {id_switches}\n\n")
            
            f.write("Frame-by-frame tracking:\n")
            for result in tracking_results:
                f.write(f"Frame {result['frame']}: {len(result['ids'])} tracks - IDs: {result['ids']}\n")
        
        # Print summary
        print(f"Processed {frame_count} frames in {process_time:.3f} seconds")
        print(f"Total unique track IDs: {len(unique_ids)}")
        print(f"Total ID switches: {id_switches}")
        print(f"Expected unique IDs: 3 (Person 1 throughout, Person 2 before occlusion, Person 2 after occlusion)")
        print(f"Results saved to {os.path.join(output_dir, 'tracking_results.txt')}")
        print("\nPlease manually verify the tracking results for confirmation (following @big-project.mdc rule)")
        
        return True, len(unique_ids) == 3 and id_switches <= 1
    except ImportError as e:
        print(f"Required library not installed: {str(e)}")
        return False, False
```

## Expected Outcomes
- ByteTrack should maintain consistent IDs for individuals across frames in `data/videos/sample_2.mp4`
- People should keep the same ID even after brief occlusion (< 30 frames)
- ID switches should be minimal (< 5% of total tracks)
- Tracking should operate at real-time speeds (> 20 FPS) for 640x480 resolution

## Verification Requirements (Following @big-project.mdc rule)
For the tracking module to be considered complete, it must pass both automated tests and personal verification:

1. **Automated Testing**:
   - All unit tests must pass successfully
   - Tracking performance must meet the minimum FPS requirement (20 FPS)
   - ID switches must be below 5% of total tracks
   - No critical errors or exceptions during extended tracking

2. **Personal Verification**:
   - Visual inspection of tracking results on `data/videos/sample_2.mp4`
   - Verification of ID consistency during occlusion events
   - Confirmation of proper bounding box tracking
   - Manual validation that individuals maintain consistent IDs
   - Sign-off on tracking performance meeting project requirements

## Test Result Documentation
Test results must be documented in the following format:

```
# Tracking Module Test Report

## Test Environment
- Hardware: [Processor, RAM, GPU]
- OS: [Operating System]
- Date: [Test Date]

## Test Results
- ByteTrack Implementation: [PASS/FAIL]
  - Tracking Accuracy: [%]
  - ID Maintenance: [%]
  - Performance: [FPS]

- Kalman Filter: [PASS/FAIL]
  - Prediction Accuracy: [%]
  - Occlusion Handling: [PASS/FAIL]

- Person Re-identification: [PASS/FAIL]
  - ID Switch Rate: [%]
  - Long-term Re-ID Success: [%]

## Personal Verification
I have personally reviewed the tracking results and verified that:
- [ ] People maintain consistent IDs throughout the video when visible
- [ ] The tracking successfully handles occlusion and reappearance
- [ ] The performance meets real-time requirements
- [ ] All known issues have been documented

Verified by: [Your Name]
Date: [Verification Date]
```

## Failure Conditions
- ID switches > 10% of total tracks
- Tracking FPS < 15 for 640x480 resolution
- Failure to maintain IDs after brief occlusion
- Excessive CPU/memory usage during extended tracking 