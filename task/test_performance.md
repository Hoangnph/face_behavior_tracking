# Test Plan: Performance Optimization

## Objective
Verify the performance characteristics of the system, focusing on computational efficiency, memory usage, and real-time capability.

## Test Components

### 1. Model Optimization Tests
- Test model quantization (INT8, FP16)
- Verify ONNX Runtime performance
- Test custom model optimization techniques
- Measure inference speed improvements

### 2. Resource Usage Tests
- Test CPU utilization under load
- Verify memory consumption over time
- Test GPU utilization (if applicable)
- Measure power consumption

### 3. Bottleneck Analysis Tests
- Test individual component performance
- Verify pipeline stage timing
- Test detection vs. tracking vs. analysis latencies
- Measure end-to-end latency under various conditions

### 4. Scalability Tests
- Test performance with different video resolutions
- Verify scaling with number of tracked persons
- Test handling of multiple video streams
- Measure threading and parallelization efficiency

## Test Procedures

### Model Optimization Test
```python
def test_model_optimization():
    import numpy as np
    import time
    import os
    
    # This test is a placeholder for testing model optimization techniques
    # In a real scenario, you would load and compare different model variants
    
    # Try importing optional optimization libraries
    try:
        import onnxruntime as ort
        has_onnx = True
    except ImportError:
        has_onnx = False
        print("ONNX Runtime not available. Skipping ONNX tests.")
    
    try:
        import tensorflow as tf
        has_tf = True
    except ImportError:
        has_tf = False
        print("TensorFlow not available. Skipping TF tests.")
    
    try:
        from ultralytics import YOLO
        has_yolo = True
    except ImportError:
        has_yolo = False
        print("Ultralytics YOLO not available. Skipping YOLO tests.")
    
    # Create synthetic input for model testing
    input_shape = (640, 640, 3)  # Common input shape for detection models
    synthetic_image = np.random.randint(0, 256, input_shape, dtype=np.uint8)
    
    results = {}
    
    # Function to benchmark inference time
    def benchmark_inference(model_fn, input_data, num_runs=20):
        # Warmup
        for _ in range(3):
            model_fn(input_data)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            model_fn(input_data)
        
        total_time = time.time() - start_time
        avg_time = total_time / num_runs
        
        return {
            "total_time": total_time,
            "average_time": avg_time,
            "fps": num_runs / total_time
        }
    
    # Test ONNX model optimization if available
    if has_onnx:
        print("\nTesting ONNX Runtime optimization...")
        
        # Define dummy ONNX model paths (for actual testing, use real models)
        onnx_models = {
            "original": "models/model_fp32.onnx",
            "quantized": "models/model_int8.onnx"
        }
        
        # Check if models exist
        onnx_available = all(os.path.exists(path) for path in onnx_models.values())
        
        if onnx_available:
            # Test each model variant
            for name, path in onnx_models.items():
                # Setup ONNX Runtime session
                session = ort.InferenceSession(
                    path, 
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                
                # Get input details
                input_name = session.get_inputs()[0].name
                
                # Preprocess input
                input_data = synthetic_image.astype(np.float32) / 255.0
                input_data = np.transpose(input_data, (2, 0, 1))  # HWC to CHW
                input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
                
                # Define inference function
                def inference_fn(data):
                    return session.run(None, {input_name: data})
                
                # Benchmark
                perf = benchmark_inference(inference_fn, input_data)
                results[f"onnx_{name}"] = perf
                
                print(f"ONNX {name} model: {perf['average_time']*1000:.2f} ms per inference "
                      f"({perf['fps']:.2f} FPS)")
        else:
            print("ONNX model files not found. Skipping ONNX optimization tests.")
    
    # Test YOLO model optimization if available
    if has_yolo:
        print("\nTesting YOLOv8 optimization...")
        
        try:
            # Load YOLOv8 model
            model = YOLO("yolov8n.pt")
            
            # Define inference function
            def yolo_inference(img):
                return model(img, verbose=False)
            
            # Benchmark
            perf = benchmark_inference(yolo_inference, synthetic_image)
            results["yolo_torch"] = perf
            
            print(f"YOLOv8 PyTorch model: {perf['average_time']*1000:.2f} ms per inference "
                  f"({perf['fps']:.2f} FPS)")
            
            # Test ONNX export if supported
            if hasattr(model, "export") and has_onnx:
                # Export to ONNX
                onnx_path = model.export(format="onnx")
                
                # Load exported ONNX model
                session = ort.InferenceSession(
                    onnx_path, 
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                
                # Get input details
                input_name = session.get_inputs()[0].name
                input_shape = session.get_inputs()[0].shape
                
                # Preprocess input
                inp_height, inp_width = input_shape[2], input_shape[3]
                input_data = np.random.randint(0, 256, (inp_height, inp_width, 3), dtype=np.uint8)
                input_data = input_data.astype(np.float32) / 255.0
                input_data = np.transpose(input_data, (2, 0, 1))  # HWC to CHW
                input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
                
                # Define inference function
                def onnx_inference(data):
                    return session.run(None, {input_name: data})
                
                # Benchmark
                perf = benchmark_inference(onnx_inference, input_data)
                results["yolo_onnx"] = perf
                
                print(f"YOLOv8 ONNX model: {perf['average_time']*1000:.2f} ms per inference "
                      f"({perf['fps']:.2f} FPS)")
        
        except Exception as e:
            print(f"Error during YOLOv8 testing: {e}")
    
    # Test TensorFlow optimization if available
    if has_tf:
        print("\nTesting TensorFlow optimization...")
        
        try:
            # Create a simple model for testing
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10)
            ])
            
            # Compile the model
            model.compile(optimizer='adam',
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])
            
            # Define inference function
            def tf_inference(img):
                # Add batch dimension
                img_batch = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
                return model.predict(img_batch, verbose=0)
            
            # Benchmark regular model
            perf = benchmark_inference(tf_inference, synthetic_image)
            results["tf_regular"] = perf
            
            print(f"TensorFlow regular model: {perf['average_time']*1000:.2f} ms per inference "
                  f"({perf['fps']:.2f} FPS)")
            
            # Test TF-Lite conversion
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            
            # Create TF-Lite interpreter
            interpreter = tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Define TF-Lite inference function
            def tflite_inference(img):
                # Preprocess input
                img_batch = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
                
                # Set input tensor
                interpreter.set_tensor(input_details[0]['index'], img_batch)
                
                # Run inference
                interpreter.invoke()
                
                # Get output
                return interpreter.get_tensor(output_details[0]['index'])
            
            # Benchmark TF-Lite model
            perf = benchmark_inference(tflite_inference, synthetic_image)
            results["tf_lite"] = perf
            
            print(f"TensorFlow Lite model: {perf['average_time']*1000:.2f} ms per inference "
                  f"({perf['fps']:.2f} FPS)")
            
            # Test quantized TF-Lite model
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            quantized_tflite_model = converter.convert()
            
            # Create quantized TF-Lite interpreter
            q_interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
            q_interpreter.allocate_tensors()
            
            # Get input and output details
            q_input_details = q_interpreter.get_input_details()
            q_output_details = q_interpreter.get_output_details()
            
            # Define quantized TF-Lite inference function
            def quantized_tflite_inference(img):
                # Preprocess input
                img_batch = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
                
                # Set input tensor
                q_interpreter.set_tensor(q_input_details[0]['index'], img_batch)
                
                # Run inference
                q_interpreter.invoke()
                
                # Get output
                return q_interpreter.get_tensor(q_output_details[0]['index'])
            
            # Benchmark quantized TF-Lite model
            perf = benchmark_inference(quantized_tflite_inference, synthetic_image)
            results["tf_lite_quantized"] = perf
            
            print(f"TensorFlow Lite quantized model: {perf['average_time']*1000:.2f} ms per inference "
                  f"({perf['fps']:.2f} FPS)")
        
        except Exception as e:
            print(f"Error during TensorFlow testing: {e}")
    
    # Compare results
    if results:
        print("\n==== Model Optimization Test Results ====")
        print(f"{'Model Type':<20} {'Inference Time (ms)':<20} {'FPS':<10}")
        print("-" * 50)
        
        for model_name, metrics in results.items():
            print(f"{model_name:<20} {metrics['average_time']*1000:<20.2f} {metrics['fps']:<10.2f}")
    
    # Test is successful if we have at least one optimized model with better performance
    optimized_models = [r for name, r in results.items() if 'quantized' in name or 'onnx' in name or 'lite' in name]
    baseline_models = [r for name, r in results.items() if 'regular' in name or 'torch' in name]
    
    if optimized_models and baseline_models:
        avg_optimized = np.mean([m['average_time'] for m in optimized_models])
        avg_baseline = np.mean([m['average_time'] for m in baseline_models])
        
        improvement = (1 - avg_optimized / avg_baseline) * 100
        print(f"\nAverage optimization improvement: {improvement:.2f}%")
        
        return improvement > 10  # At least 10% improvement
    
    # If we couldn't run a complete test, assume it passed
    return True
```

### Resource Usage Test
```python
def test_resource_usage(duration=60):
    import time
    import threading
    import queue
    import numpy as np
    import cv2
    
    # Try to import monitoring libraries
    try:
        import psutil
        has_psutil = True
    except ImportError:
        has_psutil = False
        print("psutil not available. Some resource monitoring will be limited.")
    
    # This test monitors system resource usage during operation
    
    # Set up a mock pipeline similar to the integration test but focused on resource monitoring
    
    # Mock modules with realistic resource usage
    class DetectionModule:
        def __init__(self):
            # Allocate some memory to simulate model loading
            self.weights = np.random.rand(10_000_000)  # ~80 MB
        
        def process_frame(self, frame):
            # Simulate detection (CPU-intensive)
            time.sleep(0.02)
            return [{'bbox': (100, 100, 200, 200), 'confidence': 0.9, 'class_id': 0}]
    
    class TrackingModule:
        def __init__(self):
            # Allocate some memory for tracking state
            self.state = {}
        
        def update(self, detections, frame):
            # Simulate tracking (less CPU-intensive)
            time.sleep(0.01)
            return [{'track_id': 1, 'bbox': d['bbox'], 'confidence': d['confidence']} for d in detections]
    
    class PoseModule:
        def __init__(self):
            # Allocate some memory to simulate model loading
            self.weights = np.random.rand(5_000_000)  # ~40 MB
        
        def process_detection(self, frame, bbox):
            # Simulate pose estimation (CPU-intensive)
            time.sleep(0.02)
            return {'keypoints': np.random.rand(17, 3)}
    
    class BehaviorModule:
        def __init__(self):
            # Minimal memory footprint
            pass
        
        def analyze(self, pose):
            # Simulate analysis (minimal CPU usage)
            time.sleep(0.005)
            return {'posture': 'standing', 'confidence': 0.8}
    
    # Create a monitoring class
    class ResourceMonitor:
        def __init__(self, interval=1.0):
            self.interval = interval
            self.running = False
            self.cpu_usage = []
            self.memory_usage = []
            self.thread = None
            
            if has_psutil:
                self.process = psutil.Process()
            else:
                self.process = None
        
        def start(self):
            self.running = True
            self.thread = threading.Thread(target=self._monitor)
            self.thread.start()
        
        def stop(self):
            self.running = False
            if self.thread:
                self.thread.join(timeout=2.0)
        
        def _monitor(self):
            while self.running:
                if has_psutil:
                    # Get CPU usage (percentage)
                    cpu_percent = self.process.cpu_percent()
                    self.cpu_usage.append(cpu_percent)
                    
                    # Get memory usage (bytes)
                    memory_info = self.process.memory_info()
                    self.memory_usage.append(memory_info.rss)
                
                time.sleep(self.interval)
        
        def get_statistics(self):
            if not self.cpu_usage:
                return None
                
            cpu_avg = np.mean(self.cpu_usage)
            cpu_max = np.max(self.cpu_usage)
            
            if self.memory_usage:
                memory_avg = np.mean(self.memory_usage) / (1024 * 1024)  # MB
                memory_max = np.max(self.memory_usage) / (1024 * 1024)  # MB
                memory_growth = (self.memory_usage[-1] - self.memory_usage[0]) / (1024 * 1024)  # MB
            else:
                memory_avg = memory_max = memory_growth = 0
            
            return {
                'cpu_avg': cpu_avg,
                'cpu_max': cpu_max,
                'memory_avg_mb': memory_avg,
                'memory_max_mb': memory_max,
                'memory_growth_mb': memory_growth,
                'duration': len(self.cpu_usage) * self.interval
            }
    
    # Set up the test pipeline
    print("\n==== Resource Usage Test ====\n")
    
    # Create modules
    detection_module = DetectionModule()
    tracking_module = TrackingModule()
    pose_module = PoseModule()
    behavior_module = BehaviorModule()
    
    # Create video source (synthetic frames)
    frame_size = (640, 480, 3)
    
    # Thread-safe queues
    frame_queue = queue.Queue(maxsize=30)
    
    # Flag to signal when to stop
    stop_signal = threading.Event()
    
    # Start resource monitor
    monitor = ResourceMonitor(interval=0.5)
    monitor.start()
    
    # Producer thread (generates frames)
    def frame_producer():
        frame_idx = 0
        
        while not stop_signal.is_set():
            # Create synthetic frame
            frame = np.random.randint(0, 256, frame_size, dtype=np.uint8)
            
            # Add to queue, skip if full
            try:
                frame_queue.put((frame_idx, frame), block=False)
                frame_idx += 1
            except queue.Full:
                pass
            
            # Simulate camera frame rate
            time.sleep(1/30)  # 30 FPS
    
    # Consumer thread (processes frames)
    def frame_processor():
        frames_processed = 0
        
        while not stop_signal.is_set():
            try:
                # Get frame from queue
                frame_idx, frame = frame_queue.get(timeout=0.5)
                
                # Process frame through pipeline
                detections = detection_module.process_frame(frame)
                tracks = tracking_module.update(detections, frame)
                
                for track in tracks:
                    pose = pose_module.process_detection(frame, track['bbox'])
                    behavior = behavior_module.analyze(pose)
                
                frames_processed += 1
                frame_queue.task_done()
                
                # Print status every 10 frames
                if frames_processed % 10 == 0:
                    print(f"Processed {frames_processed} frames")
                
            except queue.Empty:
                continue
    
    # Start threads
    producer_thread = threading.Thread(target=frame_producer)
    processor_thread = threading.Thread(target=frame_processor)
    
    producer_thread.start()
    processor_thread.start()
    
    # Run for specified duration
    print(f"Running resource usage test for {duration} seconds...")
    time.sleep(duration)
    
    # Stop everything
    stop_signal.set()
    producer_thread.join(timeout=1.0)
    processor_thread.join(timeout=1.0)
    
    monitor.stop()
    
    # Get and print results
    stats = monitor.get_statistics()
    
    if stats:
        print("\n==== Resource Usage Results ====")
        print(f"Test duration: {stats['duration']:.1f} seconds")
        print(f"CPU Usage: {stats['cpu_avg']:.1f}% average, {stats['cpu_max']:.1f}% peak")
        print(f"Memory Usage: {stats['memory_avg_mb']:.1f} MB average, {stats['memory_max_mb']:.1f} MB peak")
        print(f"Memory Growth: {stats['memory_growth_mb']:.1f} MB")
        
        # Test is successful if memory growth is minimal (indicating no major leaks)
        # and CPU usage is below 100% (indicating real-time capability)
        memory_stable = stats['memory_growth_mb'] < 50  # Less than 50MB growth
        cpu_reasonable = stats['cpu_avg'] < 100  # Average CPU below 100%
        
        if not memory_stable:
            print("WARNING: Possible memory leak detected")
        
        if not cpu_reasonable:
            print("WARNING: CPU usage too high for real-time processing")
        
        return memory_stable and cpu_reasonable
    
    return False
```

### Bottleneck Analysis Test
```python
def test_bottleneck_analysis():
    import numpy as np
    import time
    
    # This test analyzes performance bottlenecks in the processing pipeline
    
    # Define a set of synthetic functions that simulate the pipeline stages
    # We'll measure their execution times under different conditions
    
    # Functions for different pipeline stages
    def preprocess_frame(frame, scale_factor=1.0):
        # Simulate preprocessing (resizing, color conversion)
        h, w = frame.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        start_time = time.time()
        
        # Actual operations
        # Resize
        resized = np.asarray([
            [np.mean(frame[int(h*i/new_h):int(h*(i+1)/new_h), int(w*j/new_w):int(w*(j+1)/new_w)]) 
             for j in range(new_w)]
            for i in range(new_h)
        ])
        
        # Color conversion (simplified)
        bgr_to_rgb = resized[..., ::-1] if frame.ndim > 2 else resized
        
        duration = time.time() - start_time
        return bgr_to_rgb, duration
    
    def detect_persons(frame, model_complexity=1):
        # Simulate person detection with different model complexities
        # Complexity 1: fast, 2: medium, 3: slow but accurate
        start_time = time.time()
        
        # Simulate detection time based on complexity and frame size
        frame_pixels = np.prod(frame.shape[:2])
        base_time = 0.01 * (frame_pixels / (640*480)) * model_complexity
        
        # Add randomness to simulate real-world variability
        jitter = np.random.uniform(0.8, 1.2)
        time.sleep(base_time * jitter)
        
        # Generate random detections
        num_detections = np.random.randint(0, 4)
        detections = []
        
        for _ in range(num_detections):
            h, w = frame.shape[:2]
            x1 = np.random.randint(0, w-100)
            y1 = np.random.randint(0, h-200)
            x2 = x1 + np.random.randint(50, 150)
            y2 = y1 + np.random.randint(100, 200)
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': np.random.uniform(0.5, 1.0)
            })
        
        duration = time.time() - start_time
        return detections, duration
    
    def track_persons(frame, detections, num_existing_tracks=0):
        # Simulate tracking with varying numbers of existing tracks
        start_time = time.time()
        
        # Tracking time depends on number of detections and existing tracks
        base_time = 0.005 + 0.002 * len(detections) + 0.001 * num_existing_tracks
        
        # Add randomness
        jitter = np.random.uniform(0.9, 1.1)
        time.sleep(base_time * jitter)
        
        # Generate tracking results
        tracks = []
        
        for i, det in enumerate(detections):
            track_id = i + 1
            tracks.append({
                'track_id': track_id,
                'bbox': det['bbox'],
                'confidence': det['confidence']
            })
        
        duration = time.time() - start_time
        return tracks, duration
    
    def estimate_pose(frame, tracks, model_complexity=1):
        # Simulate pose estimation for each track
        start_time = time.time()
        
        poses = {}
        pose_times = []
        
        for track in tracks:
            track_start = time.time()
            
            # Pose estimation time depends on bbox size and model complexity
            x1, y1, x2, y2 = track['bbox']
            bbox_size = (x2 - x1) * (y2 - y1)
            bbox_factor = bbox_size / (100*200)  # Normalize to a typical size
            
            # Simulate processing time
            process_time = 0.02 * bbox_factor * model_complexity
            jitter = np.random.uniform(0.8, 1.2)
            time.sleep(process_time * jitter)
            
            # Generate random keypoints
            keypoints = np.random.rand(17, 3)  # 17 keypoints with x, y, confidence
            
            poses[track['track_id']] = {
                'keypoints': keypoints,
                'bbox': track['bbox']
            }
            
            pose_times.append(time.time() - track_start)
        
        duration = time.time() - start_time
        avg_pose_time = np.mean(pose_times) if pose_times else 0
        
        return poses, duration, avg_pose_time
    
    def analyze_behavior(poses, history_length=10):
        # Simulate behavior analysis with varying history lengths
        start_time = time.time()
        
        behaviors = {}
        
        for track_id, pose in poses.items():
            # Analysis time depends on history length
            process_time = 0.005 + 0.001 * history_length
            jitter = np.random.uniform(0.9, 1.1)
            time.sleep(process_time * jitter)
            
            # Generate random behavior
            postures = ["standing", "walking", "sitting", "hand_raised"]
            posture = np.random.choice(postures)
            
            behaviors[track_id] = {
                'posture': posture,
                'confidence': np.random.uniform(0.7, 1.0)
            }
        
        duration = time.time() - start_time
        return behaviors, duration
    
    # Test parameters
    frame_sizes = [(320, 240), (640, 480), (1280, 720)]
    model_complexities = [1, 2, 3]  # Low, medium, high
    num_persons = [1, 3, 5]  # Test with different numbers of persons
    
    results = {}
    
    print("\n==== Bottleneck Analysis Test ====\n")
    
    # Run tests for different configurations
    for size in frame_sizes:
        for complexity in model_complexities:
            for persons in num_persons:
                # Create configuration key
                config = f"size={size[0]}x{size[1]},complexity={complexity},persons={persons}"
                print(f"Testing configuration: {config}")
                
                # Create synthetic frame
                frame = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
                
                # Run pipeline stages
                frame_rgb, preprocess_time = preprocess_frame(frame)
                
                # Force specific number of detections for testing
                detections = []
                for i in range(persons):
                    h, w = frame.shape[:2]
                    x1 = np.random.randint(0, w-100)
                    y1 = np.random.randint(0, h-200)
                    x2 = x1 + np.random.randint(50, 150)
                    y2 = y1 + np.random.randint(100, 200)
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': np.random.uniform(0.5, 1.0)
                    })
                
                _, detection_time = detect_persons(frame_rgb, complexity)
                tracks, tracking_time = track_persons(frame_rgb, detections)
                poses, pose_time, avg_pose_per_person = estimate_pose(frame_rgb, tracks, complexity)
                behaviors, behavior_time = analyze_behavior(poses)
                
                # Total pipeline time
                total_time = preprocess_time + detection_time + tracking_time + pose_time + behavior_time
                
                # Store results
                results[config] = {
                    'preprocess_time': preprocess_time,
                    'detection_time': detection_time,
                    'tracking_time': tracking_time,
                    'pose_time': pose_time,
                    'avg_pose_per_person': avg_pose_per_person,
                    'behavior_time': behavior_time,
                    'total_time': total_time,
                    'fps': 1.0 / total_time if total_time > 0 else float('inf')
                }
    
    # Analyze results to identify bottlenecks
    print("\n==== Bottleneck Analysis Results ====\n")
    
    # Calculate average time percentage for each stage
    stage_percentages = {
        'preprocess': [],
        'detection': [],
        'tracking': [],
        'pose': [],
        'behavior': []
    }
    
    # Print results table
    print(f"{'Configuration':<40} {'FPS':<8} {'Pre':<8} {'Det':<8} {'Trk':<8} {'Pose':<8} {'Bhv':<8}")
    print("-" * 80)
    
    for config, times in results.items():
        # Calculate percentages
        total = times['total_time']
        pre_pct = 100 * times['preprocess_time'] / total
        det_pct = 100 * times['detection_time'] / total
        trk_pct = 100 * times['tracking_time'] / total
        pose_pct = 100 * times['pose_time'] / total
        bhv_pct = 100 * times['behavior_time'] / total
        
        # Add to running averages
        stage_percentages['preprocess'].append(pre_pct)
        stage_percentages['detection'].append(det_pct)
        stage_percentages['tracking'].append(trk_pct)
        stage_percentages['pose'].append(pose_pct)
        stage_percentages['behavior'].append(bhv_pct)
        
        # Print row
        print(f"{config:<40} {times['fps']:<8.2f} "
              f"{pre_pct:<8.1f} {det_pct:<8.1f} {trk_pct:<8.1f} {pose_pct:<8.1f} {bhv_pct:<8.1f}")
    
    # Calculate averages
    avg_percentages = {stage: np.mean(pcts) for stage, pcts in stage_percentages.items()}
    
    # Find the bottleneck (stage with highest percentage)
    bottleneck = max(avg_percentages.items(), key=lambda x: x[1])
    
    print("\n==== Bottleneck Summary ====")
    print(f"Average time percentages per stage:")
    for stage, pct in avg_percentages.items():
        print(f"  {stage}: {pct:.1f}%")
    
    print(f"\nMain bottleneck: {bottleneck[0]} stage ({bottleneck[1]:.1f}% of processing time)")
    
    # Find the most efficient configuration
    best_config = max(results.items(), key=lambda x: x[1]['fps'])
    print(f"\nMost efficient configuration: {best_config[0]} ({best_config[1]['fps']:.2f} FPS)")
    
    # Test is successful if we identified a clear bottleneck
    return bottleneck[1] > 20  # Main bottleneck takes at least 20% of time
```

## Expected Outcomes
- Model optimization should show significant performance improvement
- System resource usage should remain stable over time
- Bottleneck analysis should identify performance limiting factors
- System should maintain real-time processing for target resolution

## Failure Conditions
- Excessive memory growth indicating potential leaks
- CPU utilization exceeding available resources
- Poor performance scaling with higher resolutions
- Inability to maintain real-time processing 