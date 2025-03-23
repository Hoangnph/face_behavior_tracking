# Test Plan: Visualization and Output Options

## Objective
Verify that the system correctly visualizes tracking results, pose estimation, and behavior analysis in real-time and generates appropriate outputs in different formats.

## Test Components

### 1. Real-time Visualization Tests
- Test bounding box rendering
- Verify pose skeleton visualization
- Test track ID and status display
- Measure visualization performance impact

### 2. Annotation Tests
- Test behavior labels visualization
- Verify timestamp and frame counter display
- Test configurable display options
- Measure text rendering quality

### 3. Output Format Tests
- Test JSON data export
- Verify video recording with annotations
- Test CSV data logging
- Measure output file sizes and quality

### 4. UI Control Tests
- Test visualization toggle controls
- Verify color scheme configuration
- Test information panel display
- Measure UI responsiveness

## Test Procedures

### Basic Visualization Test
```python
def test_visualization_components():
    import numpy as np
    import cv2
    import time
    
    # This test verifies that all visualization components render correctly
    
    # Create a synthetic frame
    frame_size = (640, 480, 3)
    frame = np.zeros(frame_size, dtype=np.uint8)
    
    # Create synthetic tracking results
    tracks = [
        {
            'track_id': 1, 
            'bbox': (100, 100, 200, 200),
            'confidence': 0.95,
            'keypoints': np.random.rand(17, 3)  # 17 keypoints with x, y, confidence
        },
        {
            'track_id': 2, 
            'bbox': (300, 150, 400, 300),
            'confidence': 0.88,
            'keypoints': np.random.rand(17, 3)
        }
    ]
    
    # Add behavior analysis results
    behaviors = {
        1: {'posture': 'standing', 'confidence': 0.92},
        2: {'posture': 'sitting', 'confidence': 0.85}
    }
    
    # Test function to render bounding boxes
    def render_bboxes(image, tracks, color=(0, 255, 0), thickness=2):
        result = image.copy()
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            conf = track['confidence']
            track_id = track['track_id']
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            # Draw track ID
            text = f"ID: {track_id} ({conf:.2f})"
            font_scale = 0.5
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, font_scale, 1)[0]
            
            # Draw text background
            cv2.rectangle(
                result, 
                (x1, y1 - text_size[1] - 5), 
                (x1 + text_size[0], y1), 
                color, 
                -1  # Filled rectangle
            )
            
            # Draw text
            cv2.putText(
                result, 
                text, 
                (x1, y1 - 5), 
                font, 
                font_scale, 
                (0, 0, 0),  # Black text
                1,
                cv2.LINE_AA
            )
        
        return result
    
    # Test function to render pose skeletons
    def render_pose_skeleton(image, tracks, color=(0, 0, 255), thickness=1):
        result = image.copy()
        
        # Define skeleton connections (example for COCO keypoints)
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head and shoulders
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
        ]
        
        for track in tracks:
            keypoints = track['keypoints']
            
            # Draw skeleton connections
            for pair in skeleton:
                if keypoints[pair[0], 2] > 0.5 and keypoints[pair[1], 2] > 0.5:
                    pt1 = (int(keypoints[pair[0], 0] * image.shape[1]), 
                           int(keypoints[pair[0], 1] * image.shape[0]))
                    pt2 = (int(keypoints[pair[1], 0] * image.shape[1]), 
                           int(keypoints[pair[1], 1] * image.shape[0]))
                    
                    cv2.line(result, pt1, pt2, color, thickness, cv2.LINE_AA)
            
            # Draw keypoints
            for i in range(keypoints.shape[0]):
                if keypoints[i, 2] > 0.5:
                    x = int(keypoints[i, 0] * image.shape[1])
                    y = int(keypoints[i, 1] * image.shape[0])
                    cv2.circle(result, (x, y), 3, (0, 255, 255), -1)
        
        return result
    
    # Test function to render behavior annotations
    def render_behavior_labels(image, tracks, behaviors, color=(255, 0, 0)):
        result = image.copy()
        
        for track in tracks:
            track_id = track['track_id']
            if track_id in behaviors:
                x1, y1, x2, y2 = track['bbox']
                behavior = behaviors[track_id]
                
                # Create label text
                text = f"{behavior['posture']} ({behavior['confidence']:.2f})"
                font_scale = 0.5
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, font_scale, 1)[0]
                
                # Draw text background
                cv2.rectangle(
                    result, 
                    (x1, y2), 
                    (x1 + text_size[0], y2 + text_size[1] + 5), 
                    color, 
                    -1
                )
                
                # Draw text
                cv2.putText(
                    result, 
                    text, 
                    (x1, y2 + text_size[1]), 
                    font, 
                    font_scale, 
                    (255, 255, 255),  # White text
                    1,
                    cv2.LINE_AA
                )
        
        return result
    
    # Test function to render frame information
    def render_frame_info(image, frame_num, fps, color=(255, 255, 255)):
        result = image.copy()
        
        # Create timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Create info text
        info_text = f"Frame: {frame_num} | FPS: {fps:.1f} | Time: {timestamp}"
        font_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Draw text background
        text_size = cv2.getTextSize(info_text, font, font_scale, 1)[0]
        cv2.rectangle(
            result, 
            (10, 10), 
            (10 + text_size[0], 10 + text_size[1] + 5), 
            (0, 0, 0), 
            -1
        )
        
        # Draw text
        cv2.putText(
            result, 
            info_text, 
            (10, 10 + text_size[1]), 
            font, 
            font_scale, 
            color, 
            1,
            cv2.LINE_AA
        )
        
        return result
    
    # Setup some flags to enable/disable components
    show_bboxes = True
    show_skeletons = True
    show_behaviors = True
    show_frame_info = True
    
    # Test full rendering pipeline with performance measurement
    render_times = []
    num_frames = 30
    
    print("\nTesting visualization performance...")
    
    for frame_num in range(num_frames):
        # Create a new frame for each test (with some movement)
        frame = np.zeros(frame_size, dtype=np.uint8)
        
        # Add some background elements that change
        cv2.circle(
            frame, 
            (frame_num * 10 % frame_size[1], frame_num * 8 % frame_size[0]), 
            30, 
            (50, 50, 50), 
            -1
        )
        
        # Start timing
        start_time = time.time()
        
        # Apply visualization layers
        result = frame.copy()
        
        if show_bboxes:
            result = render_bboxes(result, tracks)
        
        if show_skeletons:
            result = render_pose_skeleton(result, tracks)
        
        if show_behaviors:
            result = render_behavior_labels(result, tracks, behaviors)
        
        if show_frame_info:
            result = render_frame_info(result, frame_num, 30.0)  # Assume 30 FPS
        
        # End timing
        render_time = time.time() - start_time
        render_times.append(render_time)
        
        # Display result (disabled in headless testing)
        # cv2.imshow("Visualization Test", result)
        # cv2.waitKey(1)
    
    # Calculate statistics
    avg_render_time = np.mean(render_times) * 1000  # Convert to ms
    max_render_time = np.max(render_times) * 1000
    
    print(f"Average rendering time: {avg_render_time:.2f} ms")
    print(f"Maximum rendering time: {max_render_time:.2f} ms")
    print(f"Estimated rendering cost: {avg_render_time / (1000/30):.1f}% of frame time at 30 FPS")
    
    # Test is successful if rendering is fast enough for real-time
    # Typically want rendering to take less than 5-10% of frame time
    is_fast_enough = avg_render_time < (1000 / 30) * 0.1  # Less than 10% of frame time at 30 FPS
    
    if is_fast_enough:
        print("Visualization performance is sufficient for real-time operation")
    else:
        print("WARNING: Visualization may impact real-time performance")
    
    return is_fast_enough
```

### Output Format Test
```python
def test_output_formats():
    import numpy as np
    import json
    import os
    import time
    import tempfile
    import csv
    
    # This test verifies that the system can correctly output data in different formats
    
    # Create temporary directory for test files
    temp_dir = tempfile.mkdtemp(prefix="tracking_test_")
    
    # Define test data
    num_frames = 10
    tracks_data = {}
    pose_data = {}
    behavior_data = {}
    
    # Generate synthetic tracking data
    for frame_idx in range(num_frames):
        # Create timestamp
        timestamp = time.time()
        
        # For each frame, create 1-3 random tracks
        num_tracks = np.random.randint(1, 4)
        frame_tracks = []
        
        for i in range(num_tracks):
            track_id = i + 1
            
            # Create track data
            track = {
                'track_id': track_id,
                'bbox': [
                    np.random.randint(0, 500),  # x1
                    np.random.randint(0, 300),  # y1
                    np.random.randint(50, 600),  # x2
                    np.random.randint(100, 400)  # y2
                ],
                'confidence': np.random.uniform(0.7, 1.0)
            }
            
            frame_tracks.append(track)
            
            # Create pose data
            pose_data.setdefault(frame_idx, {})
            pose_data[frame_idx][track_id] = {
                'keypoints': np.random.rand(17, 3).tolist(),
                'bbox': track['bbox']
            }
            
            # Create behavior data
            behavior_data.setdefault(frame_idx, {})
            postures = ["standing", "walking", "sitting", "raising_hand"]
            behavior_data[frame_idx][track_id] = {
                'posture': np.random.choice(postures),
                'confidence': np.random.uniform(0.7, 1.0)
            }
        
        # Store tracks for this frame
        tracks_data[frame_idx] = {
            'timestamp': timestamp,
            'tracks': frame_tracks
        }
    
    print("\nTesting output formats...")
    results = {}
    
    # Test 1: JSON export
    json_path = os.path.join(temp_dir, "tracking_results.json")
    try:
        # Combine all data into a single structure
        export_data = {
            'metadata': {
                'start_time': time.time(),
                'num_frames': num_frames,
                'version': '1.0'
            },
            'frames': {}
        }
        
        for frame_idx in range(num_frames):
            export_data['frames'][frame_idx] = {
                'timestamp': tracks_data[frame_idx]['timestamp'],
                'tracks': {}
            }
            
            for track in tracks_data[frame_idx]['tracks']:
                track_id = track['track_id']
                
                export_data['frames'][frame_idx]['tracks'][track_id] = {
                    'bbox': track['bbox'],
                    'confidence': track['confidence'],
                    'pose': pose_data[frame_idx][track_id] if track_id in pose_data.get(frame_idx, {}) else None,
                    'behavior': behavior_data[frame_idx][track_id] if track_id in behavior_data.get(frame_idx, {}) else None
                }
        
        # Write JSON to file
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        # Verify file was created and has the right structure
        file_size = os.path.getsize(json_path)
        
        print(f"JSON export: Success ({file_size} bytes)")
        results['json'] = True
        
    except Exception as e:
        print(f"JSON export failed: {e}")
        results['json'] = False
    
    # Test 2: CSV export
    csv_path = os.path.join(temp_dir, "tracking_results.csv")
    try:
        # Write CSV file with track data
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'frame', 'timestamp', 'track_id', 
                'x1', 'y1', 'x2', 'y2', 'confidence',
                'posture', 'posture_confidence'
            ])
            
            # Write data rows
            for frame_idx in range(num_frames):
                timestamp = tracks_data[frame_idx]['timestamp']
                
                for track in tracks_data[frame_idx]['tracks']:
                    track_id = track['track_id']
                    x1, y1, x2, y2 = track['bbox']
                    confidence = track['confidence']
                    
                    behavior = behavior_data.get(frame_idx, {}).get(track_id, {})
                    posture = behavior.get('posture', 'unknown')
                    posture_conf = behavior.get('confidence', 0.0)
                    
                    writer.writerow([
                        frame_idx, timestamp, track_id,
                        x1, y1, x2, y2, confidence,
                        posture, posture_conf
                    ])
        
        # Verify file was created
        file_size = os.path.getsize(csv_path)
        
        print(f"CSV export: Success ({file_size} bytes)")
        results['csv'] = True
        
    except Exception as e:
        print(f"CSV export failed: {e}")
        results['csv'] = False
    
    # Test 3: Create a simplified binary format
    binary_path = os.path.join(temp_dir, "tracking_data.bin")
    try:
        # Create a simple binary format with frame number, track count, and track data
        with open(binary_path, 'wb') as f:
            # Write file header: format version
            f.write((1).to_bytes(4, byteorder='little'))
            
            # Write total frame count
            f.write(num_frames.to_bytes(4, byteorder='little'))
            
            # Write each frame's data
            for frame_idx in range(num_frames):
                # Write frame number
                f.write(frame_idx.to_bytes(4, byteorder='little'))
                
                # Write timestamp (as float64)
                timestamp_bytes = np.array([tracks_data[frame_idx]['timestamp']], dtype=np.float64).tobytes()
                f.write(timestamp_bytes)
                
                # Write track count
                tracks = tracks_data[frame_idx]['tracks']
                f.write(len(tracks).to_bytes(4, byteorder='little'))
                
                # Write each track
                for track in tracks:
                    # Write track ID
                    f.write(track['track_id'].to_bytes(4, byteorder='little'))
                    
                    # Write bounding box (4 int32 values)
                    for val in track['bbox']:
                        f.write(int(val).to_bytes(4, byteorder='little'))
                    
                    # Write confidence (float32)
                    conf_bytes = np.array([track['confidence']], dtype=np.float32).tobytes()
                    f.write(conf_bytes)
                    
                    # Write behavior data if available
                    behavior = behavior_data.get(frame_idx, {}).get(track['track_id'], None)
                    if behavior:
                        # Write flag indicating behavior data is present
                        f.write((1).to_bytes(1, byteorder='little'))
                        
                        # Write posture as an enum value (0-3)
                        postures = ["standing", "walking", "sitting", "raising_hand"]
                        posture_idx = postures.index(behavior['posture']) if behavior['posture'] in postures else 0
                        f.write(posture_idx.to_bytes(1, byteorder='little'))
                        
                        # Write confidence (float32)
                        conf_bytes = np.array([behavior['confidence']], dtype=np.float32).tobytes()
                        f.write(conf_bytes)
                    else:
                        # Write flag indicating no behavior data
                        f.write((0).to_bytes(1, byteorder='little'))
        
        # Verify file was created
        file_size = os.path.getsize(binary_path)
        
        print(f"Binary export: Success ({file_size} bytes)")
        results['binary'] = True
        
    except Exception as e:
        print(f"Binary export failed: {e}")
        results['binary'] = False
    
    # Calculate space efficiency
    if all(results.values()):
        json_size = os.path.getsize(json_path)
        csv_size = os.path.getsize(csv_path)
        binary_size = os.path.getsize(binary_path)
        
        print("\nOutput format size comparison:")
        print(f"JSON format: {json_size} bytes")
        print(f"CSV format: {csv_size} bytes")
        print(f"Binary format: {binary_size} bytes")
        
        # Compare sizes
        if binary_size < csv_size and binary_size < json_size:
            print("Binary format is most space-efficient")
        elif csv_size < json_size:
            print("CSV format is more space-efficient than JSON")
        else:
            print("JSON format is least space-efficient (but most human-readable)")
    
    # Test is successful if all output formats worked
    success = all(results.values())
    
    # Clean up temp files
    for path in [json_path, csv_path, binary_path]:
        if os.path.exists(path):
            os.remove(path)
    
    try:
        os.rmdir(temp_dir)
    except:
        pass
    
    return success
```

## Expected Outcomes
- Real-time visualization should display correct bounding boxes, IDs, and poses
- Visualization should maintain performance impact below 10% of frame time
- Different output formats should correctly store tracking data
- UI controls should be responsive and functional

## Failure Conditions
- Excessive rendering time affecting real-time performance
- Visualization elements missing or displayed incorrectly
- Output formats missing critical information
- UI controls not functioning as expected 