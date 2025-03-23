# Test Plan: Pose Estimation and Keypoint Extraction

## Objective
Verify the correct implementation and performance of pose estimation for extracting human keypoints, focusing on accuracy, robustness, and computational efficiency.

## Test Components

### 1. MediaPipe Pose Tests
- Test pose model initialization
- Verify keypoint detection on various poses
- Test detection confidence thresholds
- Measure inference speed and accuracy

### 2. Face Mesh Integration Tests
- Test face mesh model initialization
- Verify facial landmark detection
- Test detection robustness with different face orientations
- Measure face mesh extraction performance

### 3. Keypoint Data Processing Tests
- Test extraction of joint angles
- Verify normalization of keypoint coordinates
- Test handling of missing/low-confidence keypoints
- Verify temporal consistency of keypoints

### 4. Performance Tests
- Measure inference time per frame
- Test scaling with multiple subjects
- Verify memory usage during extended operation
- Measure detection rate at different resolutions

## Test Procedures

### MediaPipe Pose Detection Test
```python
def test_mediapipe_pose(image_path=None):
    import cv2
    import mediapipe as mp
    import time
    import numpy as np
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
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
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,  # 0, 1, or 2
            enable_segmentation=False,
            min_detection_confidence=threshold) as pose:
                
            start_time = time.time()
            pose_result = pose.process(image_rgb)
            process_time = time.time() - start_time
            
            # Check if pose was detected
            pose_detected = pose_result.pose_landmarks is not None
            
            # Count visible landmarks if pose detected
            landmark_count = 0
            if pose_detected:
                landmarks = pose_result.pose_landmarks.landmark
                visible_landmarks = [lm for lm in landmarks if lm.visibility > 0.5]
                landmark_count = len(visible_landmarks)
                
            results[threshold] = {
                "pose_detected": pose_detected,
                "visible_landmarks": landmark_count,
                "process_time_ms": process_time * 1000
            }
            
            # Optional: Draw landmarks on image for visualization
            # if pose_detected:
            #     annotated_image = image.copy()
            #     mp_drawing.draw_landmarks(
            #         annotated_image,
            #         pose_result.pose_landmarks,
            #         mp_pose.POSE_CONNECTIONS,
            #         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            #     cv2.imshow(f"Threshold {threshold}", annotated_image)
            #     cv2.waitKey(0)
    
    # Print results
    for threshold, data in results.items():
        pose_status = "Detected" if data["pose_detected"] else "Not detected"
        print(f"Threshold {threshold}: Pose {pose_status} with {data['visible_landmarks']} "
              f"visible landmarks in {data['process_time_ms']:.2f} ms")
    
    # cv2.destroyAllWindows()
    
    # Test is successful if at least one threshold detected the pose
    return any(data["pose_detected"] for data in results.values())
```

### Face Mesh Detection Test
```python
def test_mediapipe_face_mesh(image_path=None):
    import cv2
    import mediapipe as mp
    import time
    import numpy as np
    
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
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
    
    # Process image
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
            
        start_time = time.time()
        face_result = face_mesh.process(image_rgb)
        process_time = time.time() - start_time
        
        # Check if face mesh was detected
        face_detected = face_result.multi_face_landmarks is not None
        
        # Count landmarks if face detected
        landmark_count = 0
        if face_detected:
            landmark_count = len(face_result.multi_face_landmarks[0].landmark)
        
        # Optional: Draw landmarks on image for visualization
        # if face_detected:
        #     annotated_image = image.copy()
        #     for face_landmarks in face_result.multi_face_landmarks:
        #         mp_drawing.draw_landmarks(
        #             image=annotated_image,
        #             landmark_list=face_landmarks,
        #             connections=mp_face_mesh.FACEMESH_TESSELATION,
        #             landmark_drawing_spec=None,
        #             connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
        #     cv2.imshow("Face Mesh", annotated_image)
        #     cv2.waitKey(0)
    
    # Print results
    face_status = "Detected" if face_detected else "Not detected"
    print(f"Face Mesh: {face_status} with {landmark_count} landmarks in {process_time*1000:.2f} ms")
    
    # cv2.destroyAllWindows()
    
    return face_detected
```

### Pose Feature Extraction Test
```python
def test_pose_feature_extraction():
    import numpy as np
    import time
    
    # This function tests extraction of meaningful features from pose keypoints
    # We'll create synthetic pose data to test various extraction methods
    
    # Create synthetic pose data (similar to MediaPipe output format)
    # 33 landmarks with x, y, z, visibility
    np.random.seed(42)  # for reproducibility
    
    # Create a "standing" pose
    synthetic_pose = np.zeros((33, 4))  # MediaPipe pose has 33 keypoints
    
    # Set basic body positions (simplified)
    # Note: MediaPipe uses normalized coordinates [0..1]
    
    # Head (landmark 0)
    synthetic_pose[0] = [0.5, 0.2, 0, 0.9]  # x, y, z, visibility
    
    # Shoulders (landmarks 11, 12)
    synthetic_pose[11] = [0.4, 0.3, 0, 0.9]  # left shoulder
    synthetic_pose[12] = [0.6, 0.3, 0, 0.9]  # right shoulder
    
    # Elbows (landmarks 13, 14)
    synthetic_pose[13] = [0.3, 0.4, 0, 0.9]  # left elbow
    synthetic_pose[14] = [0.7, 0.4, 0, 0.9]  # right elbow
    
    # Wrists (landmarks 15, 16)
    synthetic_pose[15] = [0.25, 0.5, 0, 0.9]  # left wrist
    synthetic_pose[16] = [0.75, 0.5, 0, 0.9]  # right wrist
    
    # Hips (landmarks 23, 24)
    synthetic_pose[23] = [0.45, 0.6, 0, 0.9]  # left hip
    synthetic_pose[24] = [0.55, 0.6, 0, 0.9]  # right hip
    
    # Knees (landmarks 25, 26)
    synthetic_pose[25] = [0.45, 0.75, 0, 0.9]  # left knee
    synthetic_pose[26] = [0.55, 0.75, 0, 0.9]  # right knee
    
    # Ankles (landmarks 27, 28)
    synthetic_pose[27] = [0.45, 0.9, 0, 0.9]  # left ankle
    synthetic_pose[28] = [0.55, 0.9, 0, 0.9]  # right ankle
    
    # Test various feature extraction functions
    start_time = time.time()
    
    # 1. Calculate joint angles
    def calculate_angle(a, b, c):
        # Calculate angle between three points
        a = np.array(a[:2])  # Convert to numpy array, use only x,y
        b = np.array(b[:2])
        c = np.array(c[:2])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    # Calculate some key angles
    left_elbow_angle = calculate_angle(
        synthetic_pose[11],  # left shoulder
        synthetic_pose[13],  # left elbow
        synthetic_pose[15]   # left wrist
    )
    
    right_elbow_angle = calculate_angle(
        synthetic_pose[12],  # right shoulder
        synthetic_pose[14],  # right elbow
        synthetic_pose[16]   # right wrist
    )
    
    left_knee_angle = calculate_angle(
        synthetic_pose[23],  # left hip
        synthetic_pose[25],  # left knee
        synthetic_pose[27]   # left ankle
    )
    
    right_knee_angle = calculate_angle(
        synthetic_pose[24],  # right hip
        synthetic_pose[26],  # right knee
        synthetic_pose[28]   # right ankle
    )
    
    # 2. Calculate relative positions (normalized by height)
    body_height = synthetic_pose[28, 1] - synthetic_pose[0, 1]  # ankle to head
    
    # Normalize positions relative to body height and center
    normalized_pose = synthetic_pose.copy()
    center = (synthetic_pose[23, :2] + synthetic_pose[24, :2]) / 2  # Center at hips
    
    for i in range(synthetic_pose.shape[0]):
        normalized_pose[i, 0] = (synthetic_pose[i, 0] - center[0]) / body_height
        normalized_pose[i, 1] = (synthetic_pose[i, 1] - center[1]) / body_height
    
    # 3. Extract stance width
    stance_width = np.abs(synthetic_pose[27, 0] - synthetic_pose[28, 0])  # Distance between ankles
    
    process_time = time.time() - start_time
    
    # Print results
    print(f"Pose feature extraction completed in {process_time*1000:.2f} ms")
    print(f"Joint Angles:")
    print(f"  Left Elbow: {left_elbow_angle:.1f} degrees")
    print(f"  Right Elbow: {right_elbow_angle:.1f} degrees")
    print(f"  Left Knee: {left_knee_angle:.1f} degrees")
    print(f"  Right Knee: {right_knee_angle:.1f} degrees")
    print(f"Stance Width (normalized): {stance_width:.3f}")
    
    # Verify that extracted features make sense for the synthetic pose
    # Angles should be around 90-120 degrees for a standing pose with arms slightly bent
    test_passed = (
        60 < left_elbow_angle < 170 and 
        60 < right_elbow_angle < 170 and
        120 < left_knee_angle < 180 and
        120 < right_knee_angle < 180 and
        stance_width > 0
    )
    
    if not test_passed:
        print("WARNING: Extracted features don't match expected values for standing pose")
    
    return test_passed
```

### Keypoint Temporal Consistency Test
```python
def test_keypoint_temporal_consistency():
    import numpy as np
    import time
    
    # This test verifies if temporal filtering improves keypoint stability
    # We'll create a synthetic sequence with noise and apply filtering
    
    # Generate a synthetic walking sequence (simplified)
    np.random.seed(42)
    frame_count = 30
    keypoint_count = 33  # MediaPipe pose keypoints
    
    # Create synthetic keypoint sequence with sinusoidal movement and noise
    sequence = []
    
    # Parameters for sinusoidal movement
    amplitude = 0.05  # Movement amplitude
    frequency = 0.2   # Movement frequency
    
    # Noise parameters
    noise_level = 0.01  # Standard deviation of noise
    
    # Generate sequence
    for frame in range(frame_count):
        # Base pose similar to our previous test
        pose = np.zeros((keypoint_count, 4))  # x, y, z, visibility
        
        # Head (landmark 0)
        pose[0] = [0.5, 0.2, 0, 0.9]
        
        # Shoulders (landmarks 11, 12)
        pose[11] = [0.4, 0.3, 0, 0.9]  # left shoulder
        pose[12] = [0.6, 0.3, 0, 0.9]  # right shoulder
        
        # Add sinusoidal movement to arms
        phase = 2 * np.pi * frequency * frame
        
        # Left arm moves up and down
        pose[13] = [0.3, 0.4 + amplitude * np.sin(phase), 0, 0.9]  # left elbow
        pose[15] = [0.25, 0.5 + amplitude * np.sin(phase), 0, 0.9]  # left wrist
        
        # Right arm moves in opposite phase
        pose[14] = [0.7, 0.4 + amplitude * np.sin(phase + np.pi), 0, 0.9]  # right elbow
        pose[16] = [0.75, 0.5 + amplitude * np.sin(phase + np.pi), 0, 0.9]  # right wrist
        
        # Legs (simplified walking motion)
        # Left leg
        pose[23] = [0.45, 0.6, 0, 0.9]  # left hip
        pose[25] = [0.45 + 0.03 * np.sin(phase), 0.75, 0, 0.9]  # left knee
        pose[27] = [0.45 + 0.06 * np.sin(phase), 0.9, 0, 0.9]  # left ankle
        
        # Right leg (opposite phase)
        pose[24] = [0.55, 0.6, 0, 0.9]  # right hip
        pose[26] = [0.55 + 0.03 * np.sin(phase + np.pi), 0.75, 0, 0.9]  # right knee
        pose[28] = [0.55 + 0.06 * np.sin(phase + np.pi), 0.9, 0, 0.9]  # right ankle
        
        # Add random noise
        noise = np.random.normal(0, noise_level, pose.shape[:2])
        pose[:, :2] += noise  # Add noise to x, y coordinates
        
        sequence.append(pose)
    
    # Convert to numpy array
    sequence = np.array(sequence)
    
    # Apply temporal filtering (simple moving average)
    def moving_average_filter(sequence, window_size=3):
        filtered = np.zeros_like(sequence)
        padded = np.pad(sequence, ((window_size//2, window_size//2), (0, 0), (0, 0)), 
                        mode='edge')
        
        for i in range(sequence.shape[0]):
            filtered[i] = np.mean(padded[i:i+window_size], axis=0)
            
        return filtered
    
    # Start timing
    start_time = time.time()
    
    # Apply filter
    filtered_sequence = moving_average_filter(sequence, window_size=5)
    
    # Calculate metrics to evaluate filtering
    original_velocity = np.diff(sequence[:, :, :2], axis=0)  # Velocity of each keypoint
    filtered_velocity = np.diff(filtered_sequence[:, :, :2], axis=0)
    
    # Calculate average velocity magnitude
    original_vel_mag = np.sqrt(np.sum(original_velocity**2, axis=2))
    filtered_vel_mag = np.sqrt(np.sum(filtered_velocity**2, axis=2))
    
    # Calculate jerk (derivative of acceleration)
    original_jerk = np.diff(np.diff(sequence[:, :, :2], axis=0), axis=0)
    filtered_jerk = np.diff(np.diff(filtered_sequence[:, :, :2], axis=0), axis=0)
    
    # Calculate average jerk magnitude
    original_jerk_mag = np.sqrt(np.sum(original_jerk**2, axis=2))
    filtered_jerk_mag = np.sqrt(np.sum(filtered_jerk**2, axis=2))
    
    process_time = time.time() - start_time
    
    # Compute average metrics
    avg_original_jerk = np.mean(original_jerk_mag)
    avg_filtered_jerk = np.mean(filtered_jerk_mag)
    
    # Print results
    print(f"Temporal consistency processing completed in {process_time*1000:.2f} ms")
    print(f"Original sequence jerk: {avg_original_jerk:.6f}")
    print(f"Filtered sequence jerk: {avg_filtered_jerk:.6f}")
    print(f"Jerk reduction: {100 * (1 - avg_filtered_jerk / avg_original_jerk):.1f}%")
    
    # Test is successful if filtering reduces jerk
    return avg_filtered_jerk < avg_original_jerk
```

## Expected Outcomes
- Pose detection should work with reasonable accuracy and speed
- Feature extraction should provide meaningful joint angles and positions
- Temporal filtering should improve keypoint stability
- Face mesh should extract facial landmarks accurately

## Failure Conditions
- Poor pose detection accuracy
- Excessive inference time making real-time processing infeasible
- Instability in extracted keypoints
- Inadequate feature extraction for behavior analysis 