# Test Plan: Behavior Analysis

## Objective
Verify the correct implementation and performance of the behavior analysis system, ensuring accurate classification of human actions and gestures.

## Test Components

### 1. Action Classification Tests
- Test basic posture recognition (standing, sitting, etc.)
- Verify walking/running detection
- Test hand gesture recognition
- Measure classification accuracy on known samples

### 2. Temporal Pattern Tests
- Test activity duration measurement
- Verify action sequence detection
- Test transition recognition between actions
- Measure temporal consistency of classifications

### 3. Rule-based System Tests
- Test rule triggering conditions
- Verify rule priority handling
- Test complex conditional rules
- Measure rule evaluation performance

### 4. Performance and Integration Tests
- Measure classification latency
- Test real-time response capabilities
- Verify integration with detection and tracking modules
- Measure system resource usage during operation

## Test Procedures

### Basic Posture Recognition Test
```python
def test_posture_recognition():
    import numpy as np
    import time
    
    # Test function for basic posture classification
    # We'll create synthetic pose data for different postures and test classification
    
    # Create synthetic pose keypoints for different postures
    def create_standing_pose():
        # Similar to our pose estimation test, but with a standing posture
        pose = np.zeros((33, 4))  # MediaPipe pose format
        
        # Head and torso
        pose[0] = [0.5, 0.1, 0, 0.9]  # nose
        pose[11] = [0.4, 0.2, 0, 0.9]  # left shoulder
        pose[12] = [0.6, 0.2, 0, 0.9]  # right shoulder
        pose[23] = [0.45, 0.5, 0, 0.9]  # left hip
        pose[24] = [0.55, 0.5, 0, 0.9]  # right hip
        
        # Arms straight down
        pose[13] = [0.35, 0.35, 0, 0.9]  # left elbow
        pose[14] = [0.65, 0.35, 0, 0.9]  # right elbow
        pose[15] = [0.3, 0.5, 0, 0.9]   # left wrist
        pose[16] = [0.7, 0.5, 0, 0.9]   # right wrist
        
        # Straight legs
        pose[25] = [0.45, 0.7, 0, 0.9]  # left knee
        pose[26] = [0.55, 0.7, 0, 0.9]  # right knee
        pose[27] = [0.45, 0.9, 0, 0.9]  # left ankle
        pose[28] = [0.55, 0.9, 0, 0.9]  # right ankle
        
        return pose
    
    def create_sitting_pose():
        # Create a sitting posture
        pose = np.zeros((33, 4))
        
        # Head and torso (higher up in the frame)
        pose[0] = [0.5, 0.2, 0, 0.9]   # nose
        pose[11] = [0.4, 0.3, 0, 0.9]  # left shoulder
        pose[12] = [0.6, 0.3, 0, 0.9]  # right shoulder
        pose[23] = [0.45, 0.6, 0, 0.9]  # left hip
        pose[24] = [0.55, 0.6, 0, 0.9]  # right hip
        
        # Arms
        pose[13] = [0.35, 0.45, 0, 0.9]  # left elbow
        pose[14] = [0.65, 0.45, 0, 0.9]  # right elbow
        pose[15] = [0.3, 0.6, 0, 0.9]   # left wrist
        pose[16] = [0.7, 0.6, 0, 0.9]   # right wrist
        
        # Bent legs (knees at same height as hips, ankles closer to camera)
        pose[25] = [0.4, 0.6, 0, 0.9]   # left knee
        pose[26] = [0.6, 0.6, 0, 0.9]   # right knee
        pose[27] = [0.4, 0.7, 0, 0.9]   # left ankle
        pose[28] = [0.6, 0.7, 0, 0.9]   # right ankle
        
        return pose
    
    def create_walking_pose():
        # Create a walking posture (mid-stride)
        pose = np.zeros((33, 4))
        
        # Head and torso
        pose[0] = [0.5, 0.1, 0, 0.9]    # nose
        pose[11] = [0.4, 0.2, 0, 0.9]   # left shoulder
        pose[12] = [0.6, 0.2, 0, 0.9]   # right shoulder
        pose[23] = [0.45, 0.5, 0, 0.9]  # left hip
        pose[24] = [0.55, 0.5, 0, 0.9]  # right hip
        
        # Arms swinging (opposite to legs)
        pose[13] = [0.45, 0.35, 0, 0.9]  # left elbow (forward)
        pose[14] = [0.5, 0.35, 0, 0.9]   # right elbow (back)
        pose[15] = [0.5, 0.45, 0, 0.9]   # left wrist (forward)
        pose[16] = [0.4, 0.45, 0, 0.9]   # right wrist (back)
        
        # Legs in stride
        pose[25] = [0.55, 0.7, 0, 0.9]   # left knee (back)
        pose[26] = [0.45, 0.7, 0, 0.9]   # right knee (forward)
        pose[27] = [0.6, 0.85, 0, 0.9]   # left ankle (back)
        pose[28] = [0.35, 0.85, 0, 0.9]  # right ankle (forward)
        
        return pose
    
    def create_raising_hand_pose():
        # Create a pose with raised hand
        pose = np.zeros((33, 4))
        
        # Head and torso
        pose[0] = [0.5, 0.1, 0, 0.9]    # nose
        pose[11] = [0.4, 0.2, 0, 0.9]   # left shoulder
        pose[12] = [0.6, 0.2, 0, 0.9]   # right shoulder
        pose[23] = [0.45, 0.5, 0, 0.9]  # left hip
        pose[24] = [0.55, 0.5, 0, 0.9]  # right hip
        
        # Right arm down, left arm raised
        pose[13] = [0.4, 0.1, 0, 0.9]   # left elbow (raised)
        pose[14] = [0.65, 0.35, 0, 0.9]  # right elbow
        pose[15] = [0.5, 0.05, 0, 0.9]  # left wrist (raised high)
        pose[16] = [0.7, 0.5, 0, 0.9]   # right wrist
        
        # Standing legs
        pose[25] = [0.45, 0.7, 0, 0.9]  # left knee
        pose[26] = [0.55, 0.7, 0, 0.9]  # right knee
        pose[27] = [0.45, 0.9, 0, 0.9]  # left ankle
        pose[28] = [0.55, 0.9, 0, 0.9]  # right ankle
        
        return pose
    
    # Create test poses
    test_poses = {
        "standing": create_standing_pose(),
        "sitting": create_sitting_pose(),
        "walking": create_walking_pose(),
        "raising_hand": create_raising_hand_pose()
    }
    
    # Define simple classification functions
    
    def is_standing(pose):
        """Check if the pose represents a standing person"""
        # Key indicators: 
        # 1. Vertical alignment of ankles, knees, and hips
        # 2. Knees not significantly bent
        
        # Get relevant keypoints
        left_hip = pose[23, :2]
        right_hip = pose[24, :2]
        left_knee = pose[25, :2]
        right_knee = pose[26, :2]
        left_ankle = pose[27, :2]
        right_ankle = pose[28, :2]
        
        # Check vertical alignment
        hip_y = (left_hip[1] + right_hip[1]) / 2
        knee_y = (left_knee[1] + right_knee[1]) / 2
        ankle_y = (left_ankle[1] + right_ankle[1]) / 2
        
        # Vertical alignment check (y-coordinates should increase from hip to ankle)
        aligned = hip_y < knee_y < ankle_y
        
        # Check if knees are straight (hip-knee-ankle angle close to 180 degrees)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        # Knees should be relatively straight for standing (>150 degrees)
        straight_knees = left_knee_angle > 150 and right_knee_angle > 150
        
        return aligned and straight_knees
    
    def is_sitting(pose):
        """Check if the pose represents a sitting person"""
        # Key indicators:
        # 1. Knees bent significantly
        # 2. Hips and knees at similar height
        
        # Get relevant keypoints
        left_hip = pose[23, :2]
        right_hip = pose[24, :2]
        left_knee = pose[25, :2]
        right_knee = pose[26, :2]
        left_ankle = pose[27, :2]
        right_ankle = pose[28, :2]
        
        # Calculate knee angles
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        # For sitting, knees should be bent (<120 degrees)
        bent_knees = left_knee_angle < 120 and right_knee_angle < 120
        
        # Check if hips and knees are at similar height
        hip_y = (left_hip[1] + right_hip[1]) / 2
        knee_y = (left_knee[1] + right_knee[1]) / 2
        
        knees_near_hips = abs(hip_y - knee_y) < 0.15  # Threshold for similarity
        
        return bent_knees and knees_near_hips
    
    def is_walking(pose):
        """Check if the pose represents a walking person"""
        # Key indicators:
        # 1. Asymmetric leg positions (stride)
        # 2. Arms typically in counter-swing to legs
        
        # Get relevant keypoints for legs
        left_hip = pose[23, :2]
        right_hip = pose[24, :2]
        left_knee = pose[25, :2]
        right_knee = pose[26, :2]
        left_ankle = pose[27, :2]
        right_ankle = pose[28, :2]
        
        # Get relevant keypoints for arms
        left_shoulder = pose[11, :2]
        right_shoulder = pose[12, :2]
        left_wrist = pose[15, :2]
        right_wrist = pose[16, :2]
        
        # Check for stride - significant difference in horizontal position of ankles
        ankle_x_diff = abs(left_ankle[0] - right_ankle[0])
        stride_detected = ankle_x_diff > 0.15  # Threshold for stride width
        
        # Check for counter-swing - arms usually move opposite to legs
        # This is simplified; real walking detection would use sequential frames
        left_arm_forward = left_wrist[0] > left_shoulder[0]
        right_arm_forward = right_wrist[0] > right_shoulder[0]
        arm_asymmetry = (left_arm_forward and not right_arm_forward) or (not left_arm_forward and right_arm_forward)
        
        return stride_detected and arm_asymmetry
    
    def is_raising_hand(pose):
        """Check if the pose represents someone raising a hand"""
        # Key indicators:
        # 1. One wrist significantly higher than shoulder
        # 2. Elbow bent upward
        
        # Get relevant keypoints
        left_shoulder = pose[11, :2]
        right_shoulder = pose[12, :2]
        left_elbow = pose[13, :2]
        right_elbow = pose[14, :2]
        left_wrist = pose[15, :2]
        right_wrist = pose[16, :2]
        
        # Check if either wrist is higher than its shoulder
        left_hand_raised = left_wrist[1] < left_shoulder[1] - 0.1  # Y decreases upward
        right_hand_raised = right_wrist[1] < right_shoulder[1] - 0.1
        
        # Check if elbows are bent appropriately for raised hand
        left_elbow_positioned = left_elbow[1] < left_shoulder[1]
        right_elbow_positioned = right_elbow[1] < right_shoulder[1]
        
        return (left_hand_raised and left_elbow_positioned) or (right_hand_raised and right_elbow_positioned)
    
    def calculate_angle(a, b, c):
        """Calculate angle between three points"""
        import numpy as np
        
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    # Run posture classification tests
    start_time = time.time()
    
    classification_results = {}
    for posture, pose in test_poses.items():
        # Apply all classifiers to each pose
        results = {
            "standing": is_standing(pose),
            "sitting": is_sitting(pose),
            "walking": is_walking(pose),
            "raising_hand": is_raising_hand(pose)
        }
        classification_results[posture] = results
    
    process_time = time.time() - start_time
    
    # Evaluate classification accuracy
    correct_classifications = 0
    for posture, results in classification_results.items():
        # A pose should be classified as its corresponding posture
        if results[posture]:
            correct_classifications += 1
    
    accuracy = correct_classifications / len(test_poses) * 100
    
    # Print results
    print(f"Posture classification completed in {process_time*1000:.2f} ms")
    print(f"Classification accuracy: {accuracy:.1f}%")
    
    for posture, results in classification_results.items():
        print(f"\nClassification results for {posture} pose:")
        for classifier, result in results.items():
            marker = "✓" if (classifier == posture and result) or (classifier != posture and not result) else "✗"
            print(f"  {classifier}: {result} {marker}")
    
    # Test is successful if accuracy is at least 75%
    return accuracy >= 75.0
```

### Activity Sequence Detection Test
```python
def test_activity_sequence_detection():
    import numpy as np
    import time
    
    # Test for detecting sequences of activities over time
    # We'll create a synthetic sequence of postures and detect patterns
    
    # Define activity codes
    STANDING = 0
    SITTING = 1
    WALKING = 2
    HAND_RAISED = 3
    
    # Create a synthetic sequence of activities
    # e.g., Standing -> Walking -> Standing -> Sitting -> Hand raised -> Standing
    activity_sequence = np.array([
        # Standing for a while (20 frames)
        *([STANDING] * 20),
        
        # Walking (15 frames)
        *([WALKING] * 15),
        
        # Standing again (10 frames)
        *([STANDING] * 10),
        
        # Sitting down (25 frames)
        *([SITTING] * 25),
        
        # Raising hand while sitting (10 frames)
        *([HAND_RAISED] * 10),
        
        # Standing up again (20 frames)
        *([STANDING] * 20)
    ])
    
    # Define patterns to detect
    patterns = {
        "sit_down": [STANDING, SITTING],
        "stand_up": [SITTING, STANDING],
        "start_walking": [STANDING, WALKING],
        "stop_walking": [WALKING, STANDING],
        "raise_hand": [SITTING, HAND_RAISED]
    }
    
    # Function to detect activity transitions
    def detect_transitions(activity_array, min_duration=5):
        activities = []
        transitions = []
        
        current_activity = activity_array[0]
        start_idx = 0
        
        # Find segments of consistent activity
        for i in range(1, len(activity_array)):
            if activity_array[i] != current_activity:
                # We found a transition
                duration = i - start_idx
                
                if duration >= min_duration:
                    activities.append({
                        "activity": current_activity,
                        "start": start_idx,
                        "end": i - 1,
                        "duration": duration
                    })
                    
                start_idx = i
                current_activity = activity_array[i]
        
        # Add the last activity
        duration = len(activity_array) - start_idx
        if duration >= min_duration:
            activities.append({
                "activity": current_activity,
                "start": start_idx,
                "end": len(activity_array) - 1,
                "duration": duration
            })
        
        # Detect transitions between activities
        for i in range(1, len(activities)):
            prev_activity = activities[i-1]["activity"]
            curr_activity = activities[i]["activity"]
            
            transitions.append({
                "from": prev_activity,
                "to": curr_activity,
                "frame": activities[i]["start"]
            })
        
        return activities, transitions
    
    # Function to detect known patterns in transitions
    def detect_patterns(transitions, patterns):
        detected_patterns = []
        
        for t in transitions:
            transition_pair = [t["from"], t["to"]]
            
            for pattern_name, pattern in patterns.items():
                if transition_pair == pattern:
                    detected_patterns.append({
                        "pattern": pattern_name,
                        "frame": t["frame"]
                    })
        
        return detected_patterns
    
    # Start timing
    start_time = time.time()
    
    # Detect activities and transitions
    activities, transitions = detect_transitions(activity_sequence)
    
    # Detect patterns
    detected_patterns = detect_patterns(transitions, patterns)
    
    process_time = time.time() - start_time
    
    # Print results
    print(f"Activity sequence detection completed in {process_time*1000:.2f} ms")
    
    print("\nDetected Activities:")
    activity_names = ["STANDING", "SITTING", "WALKING", "HAND_RAISED"]
    for a in activities:
        activity_name = activity_names[a["activity"]]
        print(f"  {activity_name}: frames {a['start']}-{a['end']} (duration: {a['duration']})")
    
    print("\nDetected Transitions:")
    for t in transitions:
        from_name = activity_names[t["from"]]
        to_name = activity_names[t["to"]]
        print(f"  {from_name} -> {to_name} at frame {t['frame']}")
    
    print("\nDetected Patterns:")
    for p in detected_patterns:
        print(f"  {p['pattern']} at frame {p['frame']}")
    
    # Expected number of patterns to detect
    expected_patterns = 5  # sit_down, stand_up, start_walking, stop_walking, raise_hand
    
    # Test is successful if we detect the correct number of patterns
    return len(detected_patterns) == expected_patterns
```

### Rule-based Behavior Test
```python
def test_rule_based_behavior():
    import numpy as np
    import time
    
    # Test rule-based behavior detection system
    # We'll implement a simple rule engine and test it on synthetic data
    
    # Define a simple rule engine
    class BehaviorRuleEngine:
        def __init__(self):
            self.rules = []
        
        def add_rule(self, name, condition_fn, priority=0):
            """Add a rule with a name, condition function, and priority"""
            self.rules.append({
                "name": name,
                "condition": condition_fn,
                "priority": priority
            })
            # Sort rules by priority (higher priority first)
            self.rules.sort(key=lambda x: -x["priority"])
        
        def evaluate(self, data):
            """Evaluate all rules on the given data, return first matching rule"""
            for rule in self.rules:
                if rule["condition"](data):
                    return rule["name"]
            return None
    
    # Create test data
    class FakeDetection:
        """Simple class to simulate detection data"""
        def __init__(self, persons=[], duration=0, frame_rate=30):
            self.persons = persons
            self.duration = duration
            self.frame_rate = frame_rate
    
    class FakePerson:
        """Simple class to simulate a tracked person"""
        def __init__(self, pose=None, face=None, track_id=0, position=(0, 0), 
                    velocity=(0, 0), time_visible=0):
            self.track_id = track_id
            self.position = position  # (x, y) center
            self.velocity = velocity  # (vx, vy) pixels per frame
            self.time_visible = time_visible  # seconds
            self.pose = pose
            self.face = face
    
    # Define pose landmarks (simplified)
    class FakePose:
        """Simple class to simulate pose data"""
        def __init__(self, keypoints=None, posture="standing", 
                    joint_angles=None, is_moving=False):
            self.keypoints = keypoints or {}
            self.posture = posture
            self.joint_angles = joint_angles or {}
            self.is_moving = is_moving
    
    # Define some rule condition functions
    def is_person_stationary(data):
        """Check if any person has been stationary for > 10 seconds"""
        for person in data.persons:
            if (abs(person.velocity[0]) < 0.1 and 
                abs(person.velocity[1]) < 0.1 and
                person.time_visible > 10):
                return True
        return False
    
    def is_person_running(data):
        """Check if any person is running (high velocity)"""
        for person in data.persons:
            velocity_mag = (person.velocity[0]**2 + person.velocity[1]**2)**0.5
            if velocity_mag > 5.0 and person.pose and person.pose.is_moving:
                return True
        return False
    
    def is_person_sitting_long(data):
        """Check if any person has been sitting for > 5 minutes"""
        for person in data.persons:
            if (person.pose and 
                person.pose.posture == "sitting" and
                person.time_visible > 300):
                return True
        return False
    
    def is_hand_raised(data):
        """Check if any person has a raised hand"""
        for person in data.persons:
            if person.pose and person.pose.posture == "hand_raised":
                return True
        return False
    
    def is_multiple_people(data):
        """Check if there are multiple people"""
        return len(data.persons) > 1
    
    # Define test cases
    test_cases = [
        # Person sitting for a long time
        FakeDetection(
            persons=[
                FakePerson(
                    track_id=1,
                    position=(300, 400),
                    velocity=(0, 0),
                    time_visible=360,  # 6 minutes
                    pose=FakePose(posture="sitting")
                )
            ],
            duration=360
        ),
        
        # Person running
        FakeDetection(
            persons=[
                FakePerson(
                    track_id=1,
                    position=(300, 400),
                    velocity=(10, 2),
                    time_visible=5,
                    pose=FakePose(posture="standing", is_moving=True)
                )
            ],
            duration=5
        ),
        
        # Person with raised hand
        FakeDetection(
            persons=[
                FakePerson(
                    track_id=1,
                    position=(300, 400),
                    velocity=(0, 0),
                    time_visible=15,
                    pose=FakePose(posture="hand_raised")
                )
            ],
            duration=15
        ),
        
        # Multiple people, one stationary
        FakeDetection(
            persons=[
                FakePerson(
                    track_id=1,
                    position=(300, 400),
                    velocity=(0, 0),
                    time_visible=20,
                    pose=FakePose(posture="standing")
                ),
                FakePerson(
                    track_id=2,
                    position=(500, 400),
                    velocity=(0.5, 0.2),
                    time_visible=15,
                    pose=FakePose(posture="standing", is_moving=True)
                )
            ],
            duration=20
        )
    ]
    
    # Expected results for each test case
    expected_behaviors = [
        "sitting_too_long",
        "running",
        "hand_raised",
        "stationary_person"
    ]
    
    # Start timing
    start_time = time.time()
    
    # Create and configure rule engine
    engine = BehaviorRuleEngine()
    
    # Add rules with priorities
    engine.add_rule("running", is_person_running, priority=10)
    engine.add_rule("hand_raised", is_hand_raised, priority=5)
    engine.add_rule("sitting_too_long", is_person_sitting_long, priority=3)
    engine.add_rule("stationary_person", is_person_stationary, priority=1)
    engine.add_rule("multiple_people", is_multiple_people, priority=0)
    
    # Evaluate rules on test cases
    results = []
    for test_case in test_cases:
        behavior = engine.evaluate(test_case)
        results.append(behavior)
    
    process_time = time.time() - start_time
    
    # Print results
    print(f"Rule-based behavior detection completed in {process_time*1000:.2f} ms")
    
    print("\nRule Evaluation Results:")
    for i, (behavior, expected) in enumerate(zip(results, expected_behaviors)):
        match = behavior == expected
        marker = "✓" if match else "✗"
        print(f"  Test {i+1}: Detected '{behavior}', Expected '{expected}' {marker}")
    
    # Calculate accuracy
    correct = sum(1 for r, e in zip(results, expected_behaviors) if r == e)
    accuracy = correct / len(test_cases) * 100
    
    print(f"\nAccuracy: {accuracy:.1f}%")
    
    # Test is successful if accuracy is 100%
    return accuracy == 100.0
```

## Expected Outcomes
- Posture recognition should correctly classify basic poses
- Activity sequence detection should identify transitions between activities
- Rule-based behavior detection should trigger appropriate rules
- Classification should happen with minimal latency

## Failure Conditions
- Poor posture classification accuracy
- Missed transitions in activity sequences
- Incorrect rule prioritization
- Excessive processing time making real-time analysis infeasible 