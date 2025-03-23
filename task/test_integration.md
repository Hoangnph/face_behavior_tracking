# Test Plan: System Integration

## Objective
Verify that all system components work together correctly, with proper data flow between modules and efficient overall operation.

## Test Components

### 1. Pipeline Integration Tests
- Test data flow between detection, tracking, and behavior analysis
- Verify proper module interface implementations
- Test error handling across module boundaries
- Measure end-to-end pipeline performance

### 2. Threading Model Tests
- Test frame producer-consumer pattern
- Verify worker thread pool efficiency
- Test thread synchronization mechanisms
- Measure threading overhead and performance

### 3. Buffer Management Tests
- Test frame buffer allocation and release
- Verify adaptive frame skipping under load
- Test memory usage during extended operation
- Measure impact of buffer size on performance

### 4. Error Handling and Recovery Tests
- Test graceful failure handling across modules
- Verify system recovery from camera disconnection
- Test behavior with corrupted or invalid frame data
- Measure system stability during extended operation

## Test Procedures

### End-to-End Pipeline Test
```python
def test_pipeline_integration(video_path=None, duration=10):
    import cv2
    import time
    import numpy as np
    import queue
    import threading
    
    # This test verifies that all modules work together in an end-to-end pipeline
    
    # Mock implementation of the detection module
    class DetectionModule:
        def __init__(self, detection_confidence=0.5):
            self.confidence = detection_confidence
            print("Initializing Detection Module...")
            
            # For this mock, we'll simulate loading a model
            time.sleep(0.5)
            
            self.initialized = True
            
        def process_frame(self, frame):
            # Simulate detection processing time
            time.sleep(0.02)
            
            # For this mock, we'll randomly generate some detections
            height, width = frame.shape[:2]
            
            # Random number of detections (0-3)
            num_detections = np.random.randint(0, 4)
            
            detections = []
            for _ in range(num_detections):
                # Random detection box
                x1 = np.random.randint(0, width - 100)
                y1 = np.random.randint(0, height - 200)
                w = np.random.randint(50, 150)
                h = np.random.randint(100, 200)
                x2 = x1 + w
                y2 = y1 + h
                
                confidence = np.random.uniform(self.confidence, 1.0)
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class_id': 0  # Assume 0 = person
                })
            
            return detections
    
    # Mock implementation of the tracking module
    class TrackingModule:
        def __init__(self):
            self.next_track_id = 1
            self.tracks = {}  # track_id -> track_info
            self.lost_tracks = {}  # track_id -> frames_lost
            self.max_lost_frames = 30
            
            print("Initializing Tracking Module...")
            self.initialized = True
        
        def update(self, detections, frame):
            # Simulate tracking processing time
            time.sleep(0.01)
            
            # Simple tracking logic (this is a much simplified version)
            tracked_objects = []
            
            # First, mark all existing tracks as not updated
            for track_id in self.tracks:
                self.tracks[track_id]['updated'] = False
            
            # Process each detection
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                confidence = det['confidence']
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Try to match with existing tracks based on position
                matched = False
                
                for track_id, track_info in self.tracks.items():
                    if track_info['updated']:
                        continue
                        
                    track_x, track_y = track_info['position']
                    distance = np.sqrt((center_x - track_x)**2 + (center_y - track_y)**2)
                    
                    # If close enough, update the track
                    if distance < 50:  # Simple distance threshold
                        self.tracks[track_id].update({
                            'position': (center_x, center_y),
                            'bbox': det['bbox'],
                            'confidence': confidence,
                            'updated': True,
                            'frames_visible': self.tracks[track_id]['frames_visible'] + 1
                        })
                        
                        # Remove from lost tracks if it was there
                        self.lost_tracks.pop(track_id, None)
                        
                        tracked_objects.append({
                            'track_id': track_id,
                            'bbox': det['bbox'],
                            'position': (center_x, center_y),
                            'confidence': confidence,
                            'frames_visible': self.tracks[track_id]['frames_visible']
                        })
                        
                        matched = True
                        break
                
                # If no match found, create a new track
                if not matched:
                    new_track_id = self.next_track_id
                    self.next_track_id += 1
                    
                    self.tracks[new_track_id] = {
                        'position': (center_x, center_y),
                        'bbox': det['bbox'],
                        'confidence': confidence,
                        'updated': True,
                        'frames_visible': 1
                    }
                    
                    tracked_objects.append({
                        'track_id': new_track_id,
                        'bbox': det['bbox'],
                        'position': (center_x, center_y),
                        'confidence': confidence,
                        'frames_visible': 1
                    })
            
            # Handle tracks that weren't updated
            for track_id, track_info in list(self.tracks.items()):
                if not track_info['updated']:
                    # Move to lost tracks
                    if track_id not in self.lost_tracks:
                        self.lost_tracks[track_id] = 1
                    else:
                        self.lost_tracks[track_id] += 1
                    
                    # If lost for too long, remove the track
                    if self.lost_tracks[track_id] > self.max_lost_frames:
                        self.tracks.pop(track_id, None)
                        self.lost_tracks.pop(track_id, None)
            
            return tracked_objects
    
    # Mock implementation of the pose estimation module
    class PoseEstimationModule:
        def __init__(self):
            print("Initializing Pose Estimation Module...")
            time.sleep(0.5)
            self.initialized = True
        
        def process_detection(self, frame, bbox):
            # Simulate pose estimation processing time
            time.sleep(0.03)
            
            # Extract the detection ROI
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(int(x2), frame.shape[1]), min(int(y2), frame.shape[0])
            
            # Skip if invalid bbox
            if x1 >= x2 or y1 >= y2:
                return None
                
            roi = frame[y1:y2, x1:x2]
            
            # For this mock, we'll randomly generate some keypoints
            # Simplified keypoints (just 10 main ones instead of full 33)
            keypoints = np.zeros((10, 3))  # (x, y, confidence)
            
            # Randomly generate keypoints within the ROI
            for i in range(10):
                kp_x = np.random.uniform(0, roi.shape[1])
                kp_y = np.random.uniform(0, roi.shape[0])
                conf = np.random.uniform(0.5, 1.0)
                
                # Convert back to original frame coordinates
                keypoints[i] = [kp_x + x1, kp_y + y1, conf]
            
            return {
                'keypoints': keypoints,
                'bbox': bbox
            }
    
    # Mock implementation of the behavior analysis module
    class BehaviorAnalysisModule:
        def __init__(self):
            print("Initializing Behavior Analysis Module...")
            self.posture_history = {}  # track_id -> list of postures
            self.initialized = True
        
        def analyze(self, track_id, pose_data, position, frames_visible):
            # Simulate behavior analysis processing time
            time.sleep(0.01)
            
            if pose_data is None:
                return None
                
            # Simple random classification for this mock
            postures = ["standing", "walking", "sitting", "hand_raised"]
            posture = np.random.choice(postures, p=[0.5, 0.3, 0.15, 0.05])
            
            # Keep track of posture history
            if track_id not in self.posture_history:
                self.posture_history[track_id] = []
            
            self.posture_history[track_id].append(posture)
            
            # If we have enough history, detect transitions
            detected_behaviors = []
            
            if len(self.posture_history[track_id]) > 5:
                # Check for patterns
                recent = self.posture_history[track_id][-5:]
                
                if "sitting" in recent and frames_visible > 300:
                    detected_behaviors.append("prolonged_sitting")
                
                if all(p == "standing" for p in recent[-3:]):
                    detected_behaviors.append("stationary")
                
                if "hand_raised" in recent:
                    detected_behaviors.append("attention_gesture")
            
            return {
                'posture': posture,
                'behaviors': detected_behaviors,
                'confidence': np.random.uniform(0.7, 1.0)
            }
    
    # Set up the test pipeline
    print("\n==== Starting Pipeline Integration Test ====\n")
    
    # Initialize the modules
    detection_module = DetectionModule()
    tracking_module = TrackingModule()
    pose_module = PoseEstimationModule()
    behavior_module = BehaviorAnalysisModule()
    
    # Set up video source
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video at {video_path}")
            return False
    else:
        cap = cv2.VideoCapture(0)  # Use default camera
        if not cap.isOpened():
            print("Error: Could not open camera")
            return False
    
    # Thread-safe queues for frame processing
    frame_queue = queue.Queue(maxsize=30)
    result_queue = queue.Queue()
    
    # Flag to signal threads to stop
    stop_signal = threading.Event()
    
    # Function for the frame producer thread
    def frame_producer():
        frame_count = 0
        start_time = time.time()
        
        while not stop_signal.is_set():
            ret, frame = cap.read()
            
            if not ret:
                print("Error reading frame")
                break
                
            # Try to add to queue, skip if full
            try:
                frame_queue.put((frame_count, frame), block=False)
                frame_count += 1
            except queue.Full:
                pass
                
            # Stop after duration (if specified)
            if duration and time.time() - start_time > duration:
                break
                
            # Simulate real-time frame rate
            time.sleep(0.03)  # ~30 FPS
    
    # Function for the processing thread
    def frame_processor():
        while not stop_signal.is_set():
            try:
                # Get frame from queue
                frame_idx, frame = frame_queue.get(timeout=1.0)
                
                # Process through the pipeline
                processing_start = time.time()
                
                # 1. Detection
                detections = detection_module.process_frame(frame)
                
                # 2. Tracking
                tracked_objects = tracking_module.update(detections, frame)
                
                # 3. Pose Estimation
                poses = {}
                for obj in tracked_objects:
                    pose_data = pose_module.process_detection(frame, obj['bbox'])
                    if pose_data:
                        poses[obj['track_id']] = pose_data
                
                # 4. Behavior Analysis
                behaviors = {}
                for obj in tracked_objects:
                    if obj['track_id'] in poses:
                        behavior = behavior_module.analyze(
                            obj['track_id'], 
                            poses[obj['track_id']], 
                            obj['position'],
                            obj['frames_visible']
                        )
                        if behavior:
                            behaviors[obj['track_id']] = behavior
                
                # 5. Combine results
                results = {
                    'frame_idx': frame_idx,
                    'processing_time': time.time() - processing_start,
                    'detection_count': len(detections),
                    'tracking_count': len(tracked_objects),
                    'pose_count': len(poses),
                    'behavior_count': len(behaviors),
                    'tracks': tracked_objects,
                    'behaviors': behaviors
                }
                
                # Add to result queue
                result_queue.put(results)
                
                # Mark as done
                frame_queue.task_done()
                
            except queue.Empty:
                continue
    
    # Start threads
    producer_thread = threading.Thread(target=frame_producer)
    processor_thread = threading.Thread(target=frame_processor)
    
    producer_thread.start()
    processor_thread.start()
    
    # Process results
    results = []
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration + 2:  # Add buffer time
            try:
                result = result_queue.get(timeout=0.5)
                results.append(result)
                result_queue.task_done()
                
                # Print some stats
                print(f"Frame {result['frame_idx']}: "
                     f"Detected {result['detection_count']} objects, "
                     f"Tracked {result['tracking_count']} objects, "
                     f"Processed in {result['processing_time']*1000:.1f} ms")
                
            except queue.Empty:
                # If we have processed frames and no more come in, we might be done
                if len(results) > 0 and frame_queue.empty():
                    break
    
    finally:
        # Clean up
        stop_signal.set()
        
        producer_thread.join(timeout=1.0)
        processor_thread.join(timeout=1.0)
        
        cap.release()
    
    # Analyze results
    total_frames = len(results)
    avg_processing_time = np.mean([r['processing_time'] for r in results]) if results else 0
    max_processing_time = np.max([r['processing_time'] for r in results]) if results else 0
    avg_detections = np.mean([r['detection_count'] for r in results]) if results else 0
    avg_tracks = np.mean([r['tracking_count'] for r in results]) if results else 0
    
    # Print summary
    print("\n==== Pipeline Integration Test Results ====")
    print(f"Processed {total_frames} frames")
    print(f"Average processing time: {avg_processing_time*1000:.2f} ms per frame")
    print(f"Maximum processing time: {max_processing_time*1000:.2f} ms")
    print(f"Average detections per frame: {avg_detections:.2f}")
    print(f"Average tracks per frame: {avg_tracks:.2f}")
    print(f"Effective FPS: {1/avg_processing_time:.2f}" if avg_processing_time > 0 else "Inf")
    
    # Test is successful if average processing time is reasonable for real-time
    # operation (less than 50ms per frame for ~20 FPS)
    return avg_processing_time < 0.05
```

### Threading Model Test
```python
def test_threading_model():
    import threading
    import queue
    import time
    import numpy as np
    
    # This test evaluates the effectiveness of the threaded processing model
    
    # Test parameters
    num_frames = 100
    frame_size = (640, 480, 3)  # 640x480 RGB
    num_workers = [1, 2, 4, 8]  # Test different numbers of worker threads
    
    # Create synthetic frames
    frames = [np.random.randint(0, 256, frame_size, dtype=np.uint8) 
             for _ in range(num_frames)]
    
    # Define a task that simulates processing a frame
    def process_frame(frame):
        # Simulate detection and tracking
        time.sleep(0.02)  # 20ms for detection
        
        # Simulate pose estimation (heavier)
        time.sleep(0.03)  # 30ms for pose
        
        # Simulate behavior analysis
        time.sleep(0.01)  # 10ms for behavior
        
        # Total: ~60ms per frame
        return {"result": "processed"}
    
    # Test function for single-threaded processing
    def test_single_threaded():
        start_time = time.time()
        results = []
        
        for frame in frames:
            result = process_frame(frame)
            results.append(result)
            
        duration = time.time() - start_time
        fps = num_frames / duration
        
        return {
            "duration": duration,
            "fps": fps,
            "results": len(results)
        }
    
    # Test function for multi-threaded processing
    def test_multi_threaded(num_worker_threads):
        # Create queues
        input_queue = queue.Queue()
        output_queue = queue.Queue()
        
        # Flag to signal worker threads to stop
        stop_signal = threading.Event()
        
        # Worker thread function
        def worker():
            while not stop_signal.is_set():
                try:
                    frame_idx, frame = input_queue.get(timeout=0.1)
                    result = process_frame(frame)
                    output_queue.put((frame_idx, result))
                    input_queue.task_done()
                except queue.Empty:
                    continue
        
        # Start timing
        start_time = time.time()
        
        # Start worker threads
        workers = []
        for _ in range(num_worker_threads):
            t = threading.Thread(target=worker)
            t.daemon = True
            t.start()
            workers.append(t)
        
        # Add frames to the queue
        for i, frame in enumerate(frames):
            input_queue.put((i, frame))
        
        # Wait for all tasks to be processed
        input_queue.join()
        
        # Stop workers
        stop_signal.set()
        for w in workers:
            w.join(timeout=0.5)
        
        # Collect results
        results = []
        while not output_queue.empty():
            results.append(output_queue.get())
        
        duration = time.time() - start_time
        fps = num_frames / duration
        
        return {
            "duration": duration,
            "fps": fps,
            "results": len(results)
        }
    
    # Run tests
    print("\n==== Threading Model Test ====\n")
    
    # Run single-threaded test
    print("Running single-threaded test...")
    single_result = test_single_threaded()
    print(f"Single-threaded: {single_result['fps']:.2f} FPS, "
          f"Processed {single_result['results']} frames in {single_result['duration']:.2f} seconds")
    
    # Run multi-threaded tests
    multi_results = {}
    for n in num_workers:
        print(f"Running {n}-worker test...")
        multi_results[n] = test_multi_threaded(n)
        print(f"{n} workers: {multi_results[n]['fps']:.2f} FPS, "
              f"Processed {multi_results[n]['results']} frames in {multi_results[n]['duration']:.2f} seconds")
    
    # Calculate scaling efficiency
    max_workers = max(num_workers)
    scaling_efficiency = multi_results[max_workers]['fps'] / single_result['fps']
    
    print("\n==== Threading Model Test Results ====")
    print(f"Single-threaded baseline: {single_result['fps']:.2f} FPS")
    for n in num_workers:
        speedup = multi_results[n]['fps'] / single_result['fps']
        efficiency = speedup / n
        print(f"{n} workers: {multi_results[n]['fps']:.2f} FPS, "
              f"Speedup: {speedup:.2f}x, "
              f"Efficiency: {efficiency:.2f}")
    
    print(f"\nScaling efficiency with {max_workers} workers: {scaling_efficiency:.2f}x")
    
    # Test is successful if we get significant speedup with multiple threads
    return scaling_efficiency > 1.5  # At least 1.5x speedup with max workers
```

### Buffer Management Test
```python
def test_buffer_management():
    import queue
    import threading
    import time
    import numpy as np
    import psutil
    
    # This test evaluates the frame buffer management capabilities
    
    # Create a simulator class for the buffer management test
    class FrameBufferSimulator:
        def __init__(self, buffer_size=30, frame_size=(640, 480, 3),
                    producer_fps=30, consumer_fps=15):
            self.buffer_size = buffer_size
            self.frame_size = frame_size
            self.producer_fps = producer_fps
            self.consumer_fps = consumer_fps
            
            # Create the buffer queue
            self.frame_queue = queue.Queue(maxsize=buffer_size)
            
            # Stats
            self.frames_produced = 0
            self.frames_consumed = 0
            self.frames_dropped = 0
            self.memory_usage = []
            
            # Signal to stop threads
            self.stop_signal = threading.Event()
            
            # Setup memory monitoring
            self.process = psutil.Process()
            self.base_memory = self.process.memory_info().rss
        
        def producer(self, duration=5):
            """Simulate camera producing frames at a fixed rate"""
            start_time = time.time()
            frame_idx = 0
            
            while not self.stop_signal.is_set():
                # Create a synthetic frame
                frame = np.zeros(self.frame_size, dtype=np.uint8)
                
                # Try to add to the queue, drop if full
                try:
                    self.frame_queue.put((frame_idx, frame), block=False)
                    self.frames_produced += 1
                except queue.Full:
                    self.frames_dropped += 1
                
                frame_idx += 1
                
                # Record memory usage
                mem_info = self.process.memory_info().rss
                self.memory_usage.append(mem_info - self.base_memory)
                
                # Wait for next frame time
                elapsed = time.time() - start_time
                if duration and elapsed >= duration:
                    break
                    
                # Sleep to maintain producer FPS
                target_time = start_time + (frame_idx / self.producer_fps)
                sleep_time = max(0, target_time - time.time())
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        def consumer(self):
            """Simulate processing frames at a variable rate"""
            while not self.stop_signal.is_set():
                try:
                    frame_idx, frame = self.frame_queue.get(timeout=0.5)
                    
                    # Simulate processing time (inverse of consumer FPS)
                    process_time = 1.0 / self.consumer_fps
                    time.sleep(process_time)
                    
                    self.frames_consumed += 1
                    self.frame_queue.task_done()
                    
                except queue.Empty:
                    # If the queue is empty and we've been running for a while, we might be done
                    if self.frames_produced > 0 and self.frames_consumed > 0:
                        continue
        
        def run_simulation(self, duration=5):
            """Run the producer-consumer simulation"""
            # Start threads
            producer_thread = threading.Thread(target=self.producer, args=(duration,))
            consumer_thread = threading.Thread(target=self.consumer)
            
            producer_thread.start()
            consumer_thread.start()
            
            # Let simulation run
            time.sleep(duration + 1)  # Add buffer time
            
            # Stop threads
            self.stop_signal.set()
            
            producer_thread.join(timeout=1.0)
            consumer_thread.join(timeout=1.0)
            
            # Calculate metrics
            frame_drop_rate = self.frames_dropped / max(1, self.frames_produced + self.frames_dropped)
            buffer_utilization = self.frames_consumed / max(1, self.frames_produced)
            max_memory = max(self.memory_usage) if self.memory_usage else 0
            
            return {
                "frames_produced": self.frames_produced,
                "frames_consumed": self.frames_consumed,
                "frames_dropped": self.frames_dropped,
                "drop_rate": frame_drop_rate,
                "buffer_utilization": buffer_utilization,
                "max_memory_bytes": max_memory
            }
    
    # Test different buffer sizes
    print("\n==== Buffer Management Test ====\n")
    
    buffer_sizes = [5, 15, 30, 60]
    scenarios = [
        {"name": "Balanced (30/30)", "producer_fps": 30, "consumer_fps": 30},
        {"name": "Producer faster (30/15)", "producer_fps": 30, "consumer_fps": 15},
        {"name": "Consumer faster (15/30)", "producer_fps": 15, "consumer_fps": 30}
    ]
    
    all_results = {}
    
    for scenario in scenarios:
        print(f"\nTesting scenario: {scenario['name']}")
        scenario_results = {}
        
        for size in buffer_sizes:
            print(f"Testing buffer size: {size}...")
            simulator = FrameBufferSimulator(
                buffer_size=size,
                producer_fps=scenario['producer_fps'],
                consumer_fps=scenario['consumer_fps']
            )
            
            results = simulator.run_simulation(duration=3)
            scenario_results[size] = results
            
            print(f"  Produced: {results['frames_produced']}, "
                 f"Consumed: {results['frames_consumed']}, "
                 f"Dropped: {results['frames_dropped']} "
                 f"({results['drop_rate']*100:.2f}%)")
        
        all_results[scenario['name']] = scenario_results
    
    # Print summary
    print("\n==== Buffer Management Test Results ====")
    
    for scenario, results in all_results.items():
        print(f"\nScenario: {scenario}")
        print("-" * 50)
        print(f"{'Buffer Size':<12} {'Produced':<10} {'Consumed':<10} {'Dropped':<10} {'Drop Rate':<12} {'Memory (MB)':<12}")
        print("-" * 50)
        for size, data in results.items():
            print(f"{size:<12} {data['frames_produced']:<10} {data['frames_consumed']:<10} "
                 f"{data['frames_dropped']:<10} {data['drop_rate']*100:<12.2f}% "
                 f"{data['max_memory_bytes']/1024/1024:<12.2f}")
    
    # Test is successful if the best buffer configuration has a low drop rate
    # in the "Producer faster" scenario
    best_size = min(buffer_sizes, 
                  key=lambda s: all_results["Producer faster (30/15)"][s]['drop_rate'])
    drop_rate = all_results["Producer faster (30/15)"][best_size]['drop_rate']
    
    print(f"\nBest buffer size for 'Producer faster' scenario: {best_size} (Drop rate: {drop_rate*100:.2f}%)")
    
    return drop_rate < 0.1  # Less than 10% drop rate is acceptable
```

## Expected Outcomes
- End-to-end pipeline should process frames in real-time
- Multi-threaded implementation should show significant speedup over single-threaded
- Buffer management should minimize frame drops under varying load
- System should maintain stable memory usage during extended operation

## Failure Conditions
- Pipeline bottlenecks causing significant frame drops
- Threading overhead exceeding performance benefits
- Memory leaks during continuous operation
- High latency making real-time analysis infeasible 