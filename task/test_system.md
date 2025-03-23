# Test Plan: System Testing and Evaluation

## Objective
Evaluate the complete system's functionality, accuracy, and performance with real-world data, ensuring it meets all requirements and can operate reliably in production environments.

## Test Components

### 1. End-to-End Tests
- Test full system pipeline with real video inputs
- Verify detection, tracking, pose estimation, and behavior analysis
- Test multiple persons in frame
- Measure overall system accuracy

### 2. Robustness Tests
- Test operation in challenging lighting conditions
- Verify performance with occlusions and crowded scenes
- Test with different camera angles and positions
- Measure system reliability under stress

### 3. Extended Operation Tests
- Test system stability over long periods (4+ hours)
- Verify no memory leaks or performance degradation
- Test recovery from failures
- Measure resource usage during extended operation

### 4. User Acceptance Tests
- Test usability of controls and interfaces
- Verify output formats meet user requirements
- Test installation and setup procedures
- Measure user satisfaction

## Test Procedures

### End-to-End System Test
```python
def test_end_to_end_system(video_path=None, duration=300):
    """
    Test the complete system pipeline with real video inputs.
    
    Args:
        video_path: Path to test video file. If None, use camera input.
        duration: Test duration in seconds for camera input.
    
    Returns:
        True if the test passes, False otherwise.
    """
    import time
    import os
    import numpy as np
    import cv2
    
    # Import your actual system components
    try:
        # Try to import system components
        # Replace with actual imports for your system
        from detection import PersonDetector
        from tracking import PersonTracker
        from pose_estimation import PoseEstimator
        from behavior_analysis import BehaviorAnalyzer
        has_modules = True
    except ImportError:
        print("Could not import system modules. Using mocks for testing.")
        has_modules = False
    
    # Create mock classes if real modules aren't available
    if not has_modules:
        class PersonDetector:
            def __init__(self, model_type="yolov8n", conf_threshold=0.5):
                self.model_type = model_type
                self.conf_threshold = conf_threshold
                print(f"Initialized mock PersonDetector with {model_type}")
                
            def detect(self, frame):
                # Mock detection - generate 1-3 random detections
                h, w = frame.shape[:2]
                num_detections = np.random.randint(1, 4)
                detections = []
                
                for _ in range(num_detections):
                    x1 = np.random.randint(0, w-100)
                    y1 = np.random.randint(0, h-200)
                    x2 = x1 + np.random.randint(50, 150)
                    y2 = y1 + np.random.randint(100, 200)
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': np.random.uniform(0.5, 1.0),
                        'class_id': 0  # Person class
                    })
                
                return detections
        
        class PersonTracker:
            def __init__(self):
                self.tracks = {}
                self.next_id = 1
                print("Initialized mock PersonTracker")
            
            def update(self, detections, frame):
                # Mock tracking - assign IDs to detections
                tracks = []
                
                for det in detections:
                    # Assign a track ID
                    track_id = self.next_id
                    self.next_id += 1
                    
                    tracks.append({
                        'track_id': track_id,
                        'bbox': det['bbox'],
                        'confidence': det['confidence']
                    })
                
                return tracks
        
        class PoseEstimator:
            def __init__(self):
                print("Initialized mock PoseEstimator")
            
            def estimate_pose(self, frame, bbox):
                # Mock pose estimation - generate random keypoints
                keypoints = np.random.rand(17, 3)  # 17 keypoints with (x, y, confidence)
                
                return {
                    'keypoints': keypoints,
                    'bbox': bbox
                }
        
        class BehaviorAnalyzer:
            def __init__(self):
                self.history = {}
                print("Initialized mock BehaviorAnalyzer")
            
            def analyze(self, track_id, pose, frame=None):
                # Mock behavior analysis
                if track_id not in self.history:
                    self.history[track_id] = []
                
                # Add pose to history (limit to 10 frames)
                self.history[track_id].append(pose)
                if len(self.history[track_id]) > 10:
                    self.history[track_id].pop(0)
                
                # Generate random behavior
                postures = ["standing", "walking", "sitting", "raising_hand"]
                posture = np.random.choice(postures)
                
                return {
                    'posture': posture,
                    'confidence': np.random.uniform(0.7, 1.0)
                }
    
    # Initialize system components
    print("\n==== End-to-End System Test ====\n")
    print("Initializing system components...")
    
    detector = PersonDetector()
    tracker = PersonTracker()
    pose_estimator = PoseEstimator()
    behavior_analyzer = BehaviorAnalyzer()
    
    # Performance tracking variables
    fps_history = []
    detection_times = []
    tracking_times = []
    pose_times = []
    behavior_times = []
    frame_count = 0
    
    # Initialize video source
    if video_path and os.path.exists(video_path):
        print(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = total_frames
        print(f"Video properties: {source_fps:.1f} FPS, {total_frames} frames")
    else:
        print("Opening camera input")
        cap = cv2.VideoCapture(0)  # Use default camera
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return False
            
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        max_frames = int(duration * source_fps)
        print(f"Camera properties: {source_fps:.1f} FPS, test duration: {duration}s")
    
    # Initialize visualization
    window_name = "End-to-End System Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Initialize results tracking
    detection_counts = []
    tracking_counts = []
    
    # Main processing loop
    start_time = time.time()
    
    try:
        while cap.isOpened() and frame_count < max_frames:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start_time = time.time()
            
            # 1. Detection
            det_start = time.time()
            detections = detector.detect(frame)
            det_time = time.time() - det_start
            detection_times.append(det_time)
            detection_counts.append(len(detections))
            
            # 2. Tracking
            track_start = time.time()
            tracks = tracker.update(detections, frame)
            track_time = time.time() - track_start
            tracking_times.append(track_time)
            tracking_counts.append(len(tracks))
            
            # 3. Pose Estimation
            pose_start = time.time()
            poses = {}
            for track in tracks:
                track_id = track['track_id']
                bbox = track['bbox']
                pose = pose_estimator.estimate_pose(frame, bbox)
                poses[track_id] = pose
            pose_time = time.time() - pose_start
            pose_times.append(pose_time)
            
            # 4. Behavior Analysis
            behavior_start = time.time()
            behaviors = {}
            for track_id, pose in poses.items():
                behavior = behavior_analyzer.analyze(track_id, pose, frame)
                behaviors[track_id] = behavior
            behavior_time = time.time() - behavior_start
            behavior_times.append(behavior_time)
            
            # Calculate FPS
            frame_time = time.time() - frame_start_time
            instantaneous_fps = 1.0 / frame_time if frame_time > 0 else 0
            fps_history.append(instantaneous_fps)
            
            # Visualization (simple version for testing)
            result = frame.copy()
            
            # Draw bounding boxes and tracks
            for track in tracks:
                track_id = track['track_id']
                x1, y1, x2, y2 = track['bbox']
                
                # Draw bounding box
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw track ID
                cv2.putText(
                    result, 
                    f"ID: {track_id}", 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    1
                )
                
                # Draw behavior if available
                if track_id in behaviors:
                    behavior = behaviors[track_id]
                    cv2.putText(
                        result, 
                        f"{behavior['posture']}", 
                        (x1, y2 + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 0, 0), 
                        1
                    )
            
            # Draw FPS
            cv2.putText(
                result, 
                f"FPS: {instantaneous_fps:.1f}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 255), 
                2
            )
            
            # Display result
            cv2.imshow(window_name, result)
            
            # Process keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                break
            
            frame_count += 1
            
            # Print progress periodically
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"Processed {frame_count} frames, Avg FPS: {avg_fps:.1f}")
    
    except KeyboardInterrupt:
        print("Test interrupted by user")
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        # Calculate and print results
        test_duration = time.time() - start_time
        
        print("\n==== End-to-End Test Results ====")
        print(f"Total frames processed: {frame_count}")
        print(f"Test duration: {test_duration:.1f} seconds")
        
        if fps_history:
            avg_fps = np.mean(fps_history)
            min_fps = np.min(fps_history)
            max_fps = np.max(fps_history)
            print(f"FPS: {avg_fps:.1f} avg, {min_fps:.1f} min, {max_fps:.1f} max")
        
        if detection_times:
            avg_det_time = np.mean(detection_times) * 1000
            print(f"Detection time: {avg_det_time:.1f} ms avg")
            print(f"Average detections per frame: {np.mean(detection_counts):.1f}")
        
        if tracking_times:
            avg_track_time = np.mean(tracking_times) * 1000
            print(f"Tracking time: {avg_track_time:.1f} ms avg")
            print(f"Average tracks per frame: {np.mean(tracking_counts):.1f}")
        
        if pose_times:
            avg_pose_time = np.mean(pose_times) * 1000
            print(f"Pose estimation time: {avg_pose_time:.1f} ms avg")
        
        if behavior_times:
            avg_behavior_time = np.mean(behavior_times) * 1000
            print(f"Behavior analysis time: {avg_behavior_time:.1f} ms avg")
        
        # Test is successful if:
        # 1. System maintained reasonable FPS (depends on hardware)
        # 2. No crashes occurred during processing
        # We can't automatically check accuracy without ground truth
        
        # Determine reasonable FPS threshold based on hardware
        # This is a simplified example - adjust based on your requirements
        is_realtime = avg_fps >= 15  # Consider 15 FPS as minimum for "real-time"
        
        if is_realtime:
            print("\nTest PASSED: System maintained real-time performance")
        else:
            print("\nTest WARNING: System did not maintain real-time performance")
            print("Consider optimizing detection or pose estimation")
        
        return is_realtime
```

### Robustness Test
```python
def test_system_robustness(test_sequences_dir=None):
    """
    Test system performance across challenging conditions.
    
    Args:
        test_sequences_dir: Directory containing test video sequences.
                          If None, use predefined test cases.
    
    Returns:
        True if the system meets robustness criteria, False otherwise.
    """
    import os
    import cv2
    import numpy as np
    
    # This is a placeholder for a more comprehensive robustness test
    # In a real scenario, you would test with multiple videos showing
    # different challenging conditions
    
    print("\n==== System Robustness Test ====\n")
    
    # Define test cases and criteria
    test_cases = {
        "low_light": {
            "description": "Test detection in low light conditions",
            "video": os.path.join(test_sequences_dir, "low_light.mp4") if test_sequences_dir else None,
            "threshold": 0.7,  # Minimum detection rate
            "passed": False
        },
        "occlusion": {
            "description": "Test tracking through occlusions",
            "video": os.path.join(test_sequences_dir, "occlusion.mp4") if test_sequences_dir else None,
            "threshold": 0.8,  # Minimum track continuity
            "passed": False
        },
        "crowded": {
            "description": "Test system in crowded scenes",
            "video": os.path.join(test_sequences_dir, "crowded.mp4") if test_sequences_dir else None,
            "threshold": 0.6,  # Minimum accuracy
            "passed": False
        },
        "angle": {
            "description": "Test with unusual camera angles",
            "video": os.path.join(test_sequences_dir, "angle.mp4") if test_sequences_dir else None,
            "threshold": 0.7,  # Minimum detection rate
            "passed": False
        }
    }
    
    # Check if test videos exist
    available_tests = []
    for case_name, case_info in test_cases.items():
        if case_info["video"] and os.path.exists(case_info["video"]):
            available_tests.append(case_name)
    
    if not available_tests:
        print("No test videos available. Creating synthetic test cases.")
        
        # Create synthetic test video for low light
        low_light_path = "synthetic_low_light.mp4"
        create_synthetic_test_video(low_light_path, "low_light")
        test_cases["low_light"]["video"] = low_light_path
        available_tests.append("low_light")
        
        # Create synthetic test video for occlusion
        occlusion_path = "synthetic_occlusion.mp4"
        create_synthetic_test_video(occlusion_path, "occlusion")
        test_cases["occlusion"]["video"] = occlusion_path
        available_tests.append("occlusion")
    
    # Run available tests
    for case_name in available_tests:
        case = test_cases[case_name]
        print(f"\nRunning test: {case_name} - {case['description']}")
        
        # Run a simplified end-to-end test on this video
        # In a real test, you would use your actual system and
        # measure specific metrics for each test case
        case["passed"] = run_robustness_test_case(case["video"], case_name, case["threshold"])
        
        print(f"Test {case_name}: {'PASSED' if case['passed'] else 'FAILED'}")
    
    # Clean up synthetic test videos
    for case_name, case_info in test_cases.items():
        if case_info["video"] and "synthetic" in case_info["video"] and os.path.exists(case_info["video"]):
            try:
                os.remove(case_info["video"])
            except:
                pass
    
    # Calculate overall result
    if available_tests:
        passed_tests = sum(1 for case_name in available_tests if test_cases[case_name]["passed"])
        pass_rate = passed_tests / len(available_tests)
        overall_passed = pass_rate >= 0.75  # Pass if at least 75% of tests passed
        
        print(f"\nOverall robustness test result: {passed_tests}/{len(available_tests)} tests passed")
        print(f"Test {'PASSED' if overall_passed else 'FAILED'}")
        
        return overall_passed
    else:
        print("No tests were executed. Check test video paths.")
        return False

def create_synthetic_test_video(output_path, test_type, duration=3, fps=20):
    """Create a synthetic test video for robustness testing."""
    import cv2
    import numpy as np
    
    # Create video writer
    frame_size = (640, 480)
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        frame_size
    )
    
    # Generate frames based on test type
    frames = int(duration * fps)
    
    for i in range(frames):
        # Create base frame
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        
        if test_type == "low_light":
            # Dark background with dim figure
            frame = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 30
            
            # Add a person silhouette
            x, y = 320 + int(50 * np.sin(i/10)), 240
            cv2.rectangle(frame, (x-40, y-100), (x+40, y+100), (60, 60, 60), -1)
            
        elif test_type == "occlusion":
            # Normal background with moving figure and occlusion
            frame = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 120
            
            # Add a person moving horizontally
            x = int(i * (frame_size[0] - 100) / frames)
            cv2.rectangle(frame, (x, 190), (x+80, 390), (200, 180, 160), -1)
            
            # Add occluding object
            if i > frames // 3 and i < 2 * frames // 3:
                cv2.rectangle(frame, (frame_size[0]//2-40, 0), (frame_size[0]//2+40, frame_size[1]), 
                             (80, 80, 80), -1)
        
        # Write frame
        out.write(frame)
    
    # Release video writer
    out.release()

def run_robustness_test_case(video_path, test_type, threshold):
    """
    Run a robustness test case on a specific video.
    
    This is a simplified mock implementation. In a real test,
    you would use your actual system components and measure
    appropriate metrics based on the test type.
    """
    import cv2
    import numpy as np
    import time
    
    if not video_path or not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return False
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {fps:.1f} FPS, {total_frames} frames")
    
    # Initialize metrics based on test type
    if test_type == "low_light":
        # For low light test, track detection success rate
        detections = []
    elif test_type == "occlusion":
        # For occlusion test, track ID persistence
        tracks_before = {}
        tracks_after = {}
        occluded = False
    elif test_type == "crowded":
        # For crowded scenes, track detection count and stability
        detection_counts = []
    elif test_type == "angle":
        # For unusual angles, track pose estimation success
        pose_confidences = []
    
    # Process video frames
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Simple mock processing based on test type
        if test_type == "low_light":
            # For low light test, attempt to detect persons in the frame
            # In this mock, we just check if the frame has enough brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            # Simulate detection based on brightness
            detected = mean_brightness > 20
            detections.append(1 if detected else 0)
            
        elif test_type == "occlusion":
            # For occlusion test, track objects before and after occlusion
            h, w = frame.shape[:2]
            
            # Determine if we're in the occlusion phase (middle third of video)
            if frame_idx < total_frames // 3:
                # Before occlusion phase
                occluded = False
                
                # Find movement in the frame (simplified)
                if frame_idx > 0:
                    diff = cv2.absdiff(frame, prev_frame)
                    motion = np.mean(diff) > 5
                    
                    if motion:
                        # Assign a mock tracking ID (1 for this test)
                        tracks_before[1] = True
                
            elif frame_idx < 2 * total_frames // 3:
                # During occlusion phase
                occluded = True
                
            else:
                # After occlusion phase
                occluded = False
                
                # Find movement in the frame (simplified)
                if frame_idx > 0:
                    diff = cv2.absdiff(frame, prev_frame)
                    motion = np.mean(diff) > 5
                    
                    if motion:
                        # Assign a mock tracking ID (should be same as before)
                        tracks_after[1] = True
        
        elif test_type == "crowded":
            # For crowded scenes, count potential detections
            # In this mock, we use simple edge detection to find shapes
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Count connected components as potential detections
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if cv2.contourArea(c) > 500]
            detection_counts.append(len(valid_contours))
        
        elif test_type == "angle":
            # For unusual angles, estimate pose quality
            # In this mock, we just use a heuristic based on frame position
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            # Generate a confidence value based on frame index
            # Start high, then decrease, then increase again
            relative_idx = frame_idx / total_frames
            mock_confidence = 0.9 - 0.4 * np.sin(relative_idx * np.pi)
            pose_confidences.append(mock_confidence)
        
        # Save previous frame for motion detection
        prev_frame = frame.copy()
        frame_idx += 1
        
        # Display progress
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
    
    # Calculate results based on test type
    if test_type == "low_light":
        detection_rate = np.mean(detections) if detections else 0
        print(f"Low light detection rate: {detection_rate:.2f}")
        return detection_rate >= threshold
    
    elif test_type == "occlusion":
        # Check if tracks after occlusion match tracks before
        track_persistence = (1 in tracks_after) and (1 in tracks_before)
        print(f"Track persistence through occlusion: {track_persistence}")
        return track_persistence
    
    elif test_type == "crowded":
        # Check stability of detection counts
        if detection_counts:
            mean_detections = np.mean(detection_counts)
            std_detections = np.std(detection_counts)
            stability = 1.0 - (std_detections / (mean_detections + 1e-5))
            print(f"Crowded scene detection stability: {stability:.2f}")
            return stability >= threshold
        return False
    
    elif test_type == "angle":
        # Check average pose confidence
        avg_confidence = np.mean(pose_confidences) if pose_confidences else 0
        print(f"Unusual angle pose confidence: {avg_confidence:.2f}")
        return avg_confidence >= threshold
    
    return False
```

### Extended Operation Test
```python
def test_extended_operation(duration=3600):  # 1 hour by default
    """
    Test system stability over an extended period.
    
    Args:
        duration: Test duration in seconds.
    
    Returns:
        True if the system is stable over the test period, False otherwise.
    """
    import time
    import threading
    import queue
    import numpy as np
    import gc
    
    try:
        import psutil
        has_psutil = True
    except ImportError:
        has_psutil = False
        print("psutil not available. Some resource monitoring will be limited.")
    
    print(f"\n==== Extended Operation Test ({duration/3600:.1f} hours) ====\n")
    
    # Set up a simplified mock processing pipeline
    # In a real test, use your actual system components
    
    # Flag to signal when to stop
    stop_signal = threading.Event()
    
    # Thread-safe queues
    frame_queue = queue.Queue(maxsize=30)
    result_queue = queue.Queue()
    
    # Resource monitoring data
    memory_usage = []
    cpu_usage = []
    frame_times = []
    processed_frames = 0
    error_count = 0
    
    # Resource monitor thread
    def monitor_resources():
        if not has_psutil:
            return
            
        process = psutil.Process()
        
        while not stop_signal.is_set():
            # Get CPU and memory usage
            try:
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                
                cpu_usage.append(cpu_percent)
                memory_usage.append(memory_info.rss)
            except:
                pass
                
            time.sleep(5)  # Sample every 5 seconds
    
    # Frame producer thread (generates synthetic frames)
    def frame_producer():
        nonlocal processed_frames
        frame_size = (640, 480, 3)
        frame_idx = 0
        
        start_time = time.time()
        
        while not stop_signal.is_set():
            # Create synthetic frame
            frame = np.random.randint(0, 256, frame_size, dtype=np.uint8)
            
            # Add to queue, skip if full
            try:
                frame_queue.put((frame_idx, frame), block=False)
                frame_idx += 1
            except queue.Full:
                pass
            
            # Report status periodically
            if frame_idx % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_idx / elapsed if elapsed > 0 else 0
                print(f"Producer: Generated {frame_idx} frames ({fps:.1f} FPS)")
            
            # Simulate camera frame rate
            time.sleep(1/30)  # 30 FPS
    
    # Frame processing thread
    def frame_processor():
        nonlocal processed_frames, error_count
        
        while not stop_signal.is_set():
            try:
                # Get frame from queue
                frame_idx, frame = frame_queue.get(timeout=0.5)
                frame_start = time.time()
                
                try:
                    # Apply mock processing pipeline
                    
                    # 1. Detection (simulate computationally intensive operation)
                    time.sleep(0.01)  # Simulate detection time
                    detections = []
                    for _ in range(np.random.randint(0, 4)):
                        h, w = frame.shape[:2]
                        x1 = np.random.randint(0, w-100)
                        y1 = np.random.randint(0, h-200)
                        x2 = x1 + np.random.randint(50, 150)
                        y2 = y1 + np.random.randint(100, 200)
                        
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': np.random.uniform(0.5, 1.0)
                        })
                    
                    # 2. Tracking
                    time.sleep(0.005)  # Simulate tracking time
                    tracks = []
                    for i, det in enumerate(detections):
                        tracks.append({
                            'track_id': i + 1,
                            'bbox': det['bbox'],
                            'confidence': det['confidence']
                        })
                    
                    # 3. Pose Estimation
                    time.sleep(0.015)  # Simulate pose estimation time
                    poses = {}
                    for track in tracks:
                        poses[track['track_id']] = {
                            'keypoints': np.random.rand(17, 3)
                        }
                    
                    # 4. Behavior Analysis
                    time.sleep(0.005)  # Simulate behavior analysis time
                    behaviors = {}
                    for track_id in poses:
                        behaviors[track_id] = {
                            'posture': np.random.choice(["standing", "walking", "sitting"]),
                            'confidence': np.random.uniform(0.7, 1.0)
                        }
                    
                    # Record processing time
                    frame_time = time.time() - frame_start
                    frame_times.append(frame_time)
                    
                    # Put result in output queue
                    result_queue.put({
                        'frame_idx': frame_idx,
                        'tracks': tracks,
                        'behaviors': behaviors
                    })
                    
                    processed_frames += 1
                    
                except Exception as e:
                    error_count += 1
                    print(f"Error processing frame {frame_idx}: {e}")
                
                # Mark task as done
                frame_queue.task_done()
                
                # Forced memory cleanup every 1000 frames
                if processed_frames % 1000 == 0:
                    gc.collect()
                
            except queue.Empty:
                continue
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_resources)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Start producer and processor threads
    producer_thread = threading.Thread(target=frame_producer)
    processor_thread = threading.Thread(target=frame_processor)
    
    producer_thread.start()
    processor_thread.start()
    
    # Run for specified duration
    print(f"Running extended operation test for {duration} seconds...")
    
    try:
        time.sleep(duration)
    except KeyboardInterrupt:
        print("Test interrupted by user")
    
    # Stop all threads
    stop_signal.set()
    producer_thread.join(timeout=2.0)
    processor_thread.join(timeout=2.0)
    
    # Analyze results
    print("\n==== Extended Operation Test Results ====")
    print(f"Total frames processed: {processed_frames}")
    print(f"Error count: {error_count}")
    
    test_passed = True
    
    # Analyze frame processing times for stability
    if frame_times:
        avg_time = np.mean(frame_times)
        std_time = np.std(frame_times)
        cv_time = std_time / avg_time if avg_time > 0 else 0
        
        print(f"Average processing time: {avg_time*1000:.1f} ms")
        print(f"Processing time variability: {cv_time*100:.1f}%")
        
        time_stable = cv_time < 0.3  # Less than 30% variability
        if not time_stable:
            test_passed = False
            print("WARNING: Processing time showed high variability")
    
    # Analyze memory usage for leaks
    if has_psutil and memory_usage:
        # Convert bytes to MB
        memory_mb = [m / (1024*1024) for m in memory_usage]
        
        initial_memory = memory_mb[0]
        final_memory = memory_mb[-1]
        max_memory = max(memory_mb)
        
        growth = final_memory - initial_memory
        growth_pct = 100 * growth / initial_memory if initial_memory > 0 else 0
        
        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"Final memory: {final_memory:.1f} MB")
        print(f"Maximum memory: {max_memory:.1f} MB")
        print(f"Memory growth: {growth:.1f} MB ({growth_pct:.1f}%)")
        
        # Check for significant memory growth
        memory_stable = growth_pct < 20  # Less than 20% growth
        if not memory_stable:
            test_passed = False
            print("WARNING: Significant memory growth detected (possible leak)")
    
    # Analyze CPU usage
    if has_psutil and cpu_usage:
        avg_cpu = np.mean(cpu_usage)
        max_cpu = max(cpu_usage)
        
        print(f"Average CPU usage: {avg_cpu:.1f}%")
        print(f"Peak CPU usage: {max_cpu:.1f}%")
    
    # Check error rate
    error_rate = error_count / processed_frames if processed_frames > 0 else 1.0
    print(f"Error rate: {error_rate*100:.2f}%")
    
    errors_acceptable = error_rate < 0.001  # Less than 0.1% errors
    if not errors_acceptable:
        test_passed = False
        print("WARNING: Error rate too high")
    
    # Overall test result
    if test_passed:
        print("\nExtended operation test PASSED")
    else:
        print("\nExtended operation test FAILED")
    
    return test_passed
```

## Expected Outcomes
- System successfully processes videos with acceptable frame rate
- Detection, tracking, pose estimation, and behavior analysis work correctly
- System handles challenging conditions appropriately
- System remains stable over extended periods

## Failure Conditions
- System fails to maintain acceptable frame rate
- Detection or tracking fails under challenging conditions
- Memory leaks or resource exhaustion during extended operation
- Excessive error rate during processing 