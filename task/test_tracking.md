# Test Plan: Person Tracking

## Objective
Verify the correct implementation and performance of the ByteTrack-based person tracking system, ensuring accurate ID assignment and maintenance across frames.

## Test Components

### 1. ByteTrack Implementation Tests
- Test ByteTrack initialization with different parameters
- Verify tracking performance with simulated detections
- Test handling of missed detections and reappearances
- Measure tracking accuracy against ground truth

### 2. Kalman Filter Tests
- Test state prediction accuracy
- Verify filter behavior with missing detections
- Test motion model under different movement scenarios
- Measure uncertainty estimation accuracy

### 3. Person Re-identification Tests
- Test ID maintenance during occlusion
- Verify ID re-assignment after long disappearance
- Test handling of similar-looking individuals
- Measure ID switch frequency

### 4. Performance Tests
- Measure tracking overhead per frame
- Test scaling with number of tracked objects
- Verify real-time performance with full pipeline
- Measure memory usage during extended tracking

## Test Procedures

### ByteTrack Basic Tracking Test
```python
def test_bytetrack_basic():
    try:
        import numpy as np
        from yolox.tracker.byte_tracker import BYTETracker
        import time
        
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
        start_time = time.time()
        
        for frame_idx, dets in enumerate(detection_sequence):
            if len(dets) > 0:
                online_targets = tracker.update(dets, [1000, 1000], [800, 800])
                online_tlwhs = []
                online_ids = []
                online_scores = []
                
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                
                tracking_results.append({
                    'frame': frame_idx,
                    'tlwhs': online_tlwhs,
                    'ids': online_ids,
                    'scores': online_scores
                })
            else:
                tracking_results.append({
                    'frame': frame_idx,
                    'tlwhs': [],
                    'ids': [],
                    'scores': []
                })
        
        processing_time = time.time() - start_time
        
        # Analyze results
        id_consistency = {}
        for res in tracking_results:
            for i, track_id in enumerate(res['ids']):
                if track_id not in id_consistency:
                    id_consistency[track_id] = {
                        'first_seen': res['frame'],
                        'last_seen': res['frame'],
                        'appearance_count': 1
                    }
                else:
                    id_consistency[track_id]['last_seen'] = res['frame']
                    id_consistency[track_id]['appearance_count'] += 1
        
        # Print results
        print(f"ByteTrack processed {frame_count} frames in {processing_time:.4f} seconds "
              f"({frame_count/processing_time:.1f} FPS)")
        print(f"Detected {len(id_consistency)} unique IDs")
        
        for track_id, stats in id_consistency.items():
            print(f"ID {track_id}: First seen at frame {stats['first_seen']}, "
                  f"Last seen at frame {stats['last_seen']}, "
                  f"Tracked for {stats['appearance_count']} frames")
        
        # Evaluate ID consistency (ideally should have exactly 2 IDs)
        test_passed = len(id_consistency) == 2
        if not test_passed:
            print("WARNING: Expected 2 tracked objects, but found", len(id_consistency))
        
        return test_passed
    
    except ImportError as e:
        print(f"Error: ByteTrack dependencies not installed. {e}")
        return False
    except Exception as e:
        print(f"Error during ByteTrack testing: {e}")
        return False
```

### Tracking with Detection Gaps Test
```python
def test_tracking_with_gaps():
    import numpy as np
    import cv2
    import time
    
    try:
        # You can use your actual tracker implementation here
        # This is a simplified mock-up for the test procedure
        class SimpleTracker:
            def __init__(self):
                self.tracks = {}
                self.next_id = 1
            
            def update(self, detections):
                # Simple tracking logic
                results = []
                if not self.tracks:  # First frame
                    for det in detections:
                        self.tracks[self.next_id] = {
                            'bbox': det[:4],
                            'last_seen': 0,
                            'lost': 0
                        }
                        results.append((self.next_id, det))
                        self.next_id += 1
                else:
                    # Associate detections with existing tracks
                    # (Simplified association based on center distance)
                    for det in detections:
                        det_center = ((det[0] + det[2]) / 2, (det[1] + det[3]) / 2)
                        best_id = None
                        best_dist = float('inf')
                        
                        for track_id, track in self.tracks.items():
                            if track['lost'] > 30:  # Remove if lost for too long
                                continue
                                
                            track_center = ((track['bbox'][0] + track['bbox'][2]) / 2, 
                                           (track['bbox'][1] + track['bbox'][3]) / 2)
                            dist = np.sqrt((det_center[0] - track_center[0])**2 + 
                                          (det_center[1] - track_center[1])**2)
                            
                            if dist < best_dist and dist < 100:  # Threshold
                                best_dist = dist
                                best_id = track_id
                        
                        if best_id is not None:
                            self.tracks[best_id]['bbox'] = det[:4]
                            self.tracks[best_id]['last_seen'] += 1
                            self.tracks[best_id]['lost'] = 0
                            results.append((best_id, det))
                        else:
                            # New track
                            self.tracks[self.next_id] = {
                                'bbox': det[:4],
                                'last_seen': 0,
                                'lost': 0
                            }
                            results.append((self.next_id, det))
                            self.next_id += 1
                
                # Update lost count for missing tracks
                matched_ids = [r[0] for r in results]
                for track_id in self.tracks:
                    if track_id not in matched_ids:
                        self.tracks[track_id]['lost'] += 1
                
                return results
        
        # Create a synthetic video with tracking gaps
        width, height = 800, 600
        frame_count = 100
        
        # Define object trajectory with disappearance
        obj_x = width // 4
        obj_y = height // 2
        obj_width = 60
        obj_height = 120
        
        # Initialize tracker
        tracker = SimpleTracker()
        
        # Run tracking simulation
        track_history = []
        processing_times = []
        
        for frame_idx in range(frame_count):
            # Create a blank frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Update object position
            obj_x += 5
            
            # Object disappears between frames 30-50 and 70-80
            has_detection = True
            if (30 <= frame_idx < 50) or (70 <= frame_idx < 80):
                has_detection = False
            
            detections = []
            if has_detection:
                # Draw the object
                x1 = max(0, obj_x - obj_width // 2)
                y1 = max(0, obj_y - obj_height // 2)
                x2 = min(width, x1 + obj_width)
                y2 = min(height, y1 + obj_height)
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Add to detections
                detections.append([x1, y1, x2, y2, 0.9])
            
            # Track the detections
            start_time = time.time()
            results = tracker.update(detections)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Draw tracking results
            for track_id, det in results:
                x1, y1, x2, y2 = map(int, det[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Save tracking history
            track_history.append({
                'frame': frame_idx,
                'detections': len(detections),
                'tracks': len(results),
                'track_ids': [r[0] for r in results]
            })
            
            # Optional: Display the frame
            # cv2.imshow('Tracking Test', frame)
            # cv2.waitKey(1)
        
        # cv2.destroyAllWindows()
        
        # Analyze tracking consistency through gaps
        unique_ids = set()
        for record in track_history:
            unique_ids.update(record['track_ids'])
        
        # Ideally, we should have a single track ID despite gaps
        print(f"Tracking gaps test: Found {len(unique_ids)} unique track IDs")
        print(f"Average processing time: {np.mean(processing_times)*1000:.2f} ms per frame")
        
        # Analyze ID consistency through gaps
        id_spans = {}
        for record in track_history:
            for track_id in record['track_ids']:
                if track_id not in id_spans:
                    id_spans[track_id] = {'first': record['frame'], 'last': record['frame']}
                else:
                    id_spans[track_id]['last'] = record['frame']
        
        for track_id, span in id_spans.items():
            print(f"ID {track_id}: Tracked from frame {span['first']} to {span['last']} "
                  f"(duration: {span['last'] - span['first'] + 1})")
        
        # Test is successful if we have minimal ID switches
        return len(unique_ids) <= 3  # Allow for some ID switches during long gaps
    
    except Exception as e:
        print(f"Error during tracking with gaps test: {e}")
        return False
```

### Kalman Filter Prediction Test
```python
def test_kalman_filter_prediction():
    import numpy as np
    import matplotlib.pyplot as plt
    from filterpy.kalman import KalmanFilter
    import time
    
    try:
        # Initialize Kalman filter for 2D tracking (x, y, vx, vy)
        kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (constant velocity model)
        dt = 1.0  # time step
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only measure position, not velocity)
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement noise
        kf.R = np.eye(2) * 5.0
        
        # Process noise
        q = 0.1  # process noise
        kf.Q = np.array([
            [q*dt**4/4, 0, q*dt**3/2, 0],
            [0, q*dt**4/4, 0, q*dt**3/2],
            [q*dt**3/2, 0, q*dt**2, 0],
            [0, q*dt**3/2, 0, q*dt**2]
        ])
        
        # Initial state and covariance
        kf.x = np.array([0., 0., 0., 0.])  # initial state
        kf.P = np.eye(4) * 1000.0  # initial uncertainty
        
        # Create a test track (sine wave + noise)
        track_length = 100
        true_x = np.arange(track_length)
        true_y = 100 + 50 * np.sin(0.1 * true_x)
        
        # Add noise to create measurements
        np.random.seed(42)  # for reproducibility
        noise_x = np.random.normal(0, 2, track_length)
        noise_y = np.random.normal(0, 5, track_length)
        meas_x = true_x + noise_x
        meas_y = true_y + noise_y
        
        # Create missing measurements
        missing_idx = set(np.random.choice(range(20, 80), 15, replace=False))
        
        # Run Kalman filter
        filtered_x = []
        filtered_y = []
        predicted_x = []
        predicted_y = []
        
        start_time = time.time()
        for i in range(track_length):
            # Predict
            kf.predict()
            
            # Store prediction (before update)
            predicted_x.append(kf.x[0])
            predicted_y.append(kf.x[1])
            
            # Update if measurement available
            if i not in missing_idx:
                z = np.array([meas_x[i], meas_y[i]])
                kf.update(z)
            
            # Store filtered state
            filtered_x.append(kf.x[0])
            filtered_y.append(kf.x[1])
        
        processing_time = time.time() - start_time
        
        # Calculate errors
        mse_filtered = np.mean((np.array(filtered_x) - true_x)**2 + (np.array(filtered_y) - true_y)**2)
        mse_predicted = np.mean((np.array(predicted_x) - true_x)**2 + (np.array(predicted_y) - true_y)**2)
        mse_raw = np.mean((meas_x - true_x)**2 + (meas_y - true_y)**2)
        
        # Print results
        print(f"Kalman filter processed {track_length} steps in {processing_time:.4f} seconds")
        print(f"Mean Squared Error (MSE):")
        print(f"  Raw measurements: {mse_raw:.2f}")
        print(f"  Filtered measurements: {mse_filtered:.2f}")
        print(f"  Predictions: {mse_predicted:.2f}")
        
        # Optional: Plot results
        # plt.figure(figsize=(12, 6))
        # plt.plot(true_x, true_y, 'k-', label='Ground Truth')
        # plt.plot(meas_x, meas_y, 'ro', alpha=0.3, label='Measurements')
        # plt.plot(filtered_x, filtered_y, 'b-', label='Kalman Filter')
        # plt.legend()
        # plt.title('Kalman Filter Performance')
        # plt.savefig('kalman_filter_test.png')
        
        # Success if filtered MSE is less than raw MSE
        return mse_filtered < mse_raw
    
    except ImportError:
        print("Error: FilterPy package not installed. Install with: pip install filterpy")
        return False
    except Exception as e:
        print(f"Error during Kalman filter test: {e}")
        return False
```

## Expected Outcomes
- ByteTrack should maintain consistent IDs across frames
- Tracking should continue through short occlusions
- Kalman filter should provide accurate predictions of movement
- Re-identification should minimize ID switches

## Failure Conditions
- Excessive ID switches during tracking
- Lost tracks during brief occlusions
- Poor prediction accuracy during gaps
- High computational overhead making real-time tracking infeasible 