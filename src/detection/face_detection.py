"""
Face detection module using MediaPipe face detection.
"""
from typing import List, Union, Dict, Any, Optional, Tuple
import numpy as np
import cv2
import mediapipe as mp
from .base_detection import BaseDetector, Detection, BoundingBox


class FaceDetection(Detection):
    """
    Face detection result with optional landmarks.
    Extends the base Detection class with additional face-specific information.
    """
    def __init__(self, 
                 bbox: BoundingBox, 
                 confidence: float, 
                 label: str = "face",
                 landmarks: Optional[Dict[str, Tuple[int, int]]] = None):
        """
        Initialize face detection.
        
        Args:
            bbox: Bounding box of the detected face.
            confidence: Detection confidence score.
            label: Detection label, defaults to "face".
            landmarks: Dictionary of facial landmarks if available.
        """
        super().__init__(bbox=bbox, confidence=confidence, label=label)
        self.landmarks = landmarks or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert face detection to dictionary format for serialization."""
        data = super().to_dict()
        data["landmarks"] = self.landmarks
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FaceDetection':
        """Create a FaceDetection from dictionary format."""
        detection = super().from_dict(data)
        return cls(
            bbox=detection.bbox,
            confidence=detection.confidence,
            label=detection.label,
            landmarks=data.get("landmarks", {})
        )


class MediaPipeFaceDetector(BaseDetector):
    """
    Face detector using MediaPipe Face Detection.
    """
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 model_selection: int = 1):
        """
        Initialize MediaPipe face detector.
        
        Args:
            min_detection_confidence: Minimum confidence value for face detection.
            model_selection: Model selection - 0 for short-range (2m) or 1 for full-range (5m).
        """
        super().__init__(confidence_threshold=min_detection_confidence)
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        self.face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
    
    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in the input image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV).
            
        Returns:
            List of FaceDetection objects.
        """
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and return the detection results
        with self.face_detection.FaceDetection(
            min_detection_confidence=self.min_detection_confidence,
            model_selection=self.model_selection
        ) as face_detector:
            results = face_detector.process(image_rgb)
        
        # Process detection results
        return self._process_detections(results, image_rgb)
    
    def _process_detections(self, results, image: np.ndarray) -> List[FaceDetection]:
        """
        Process MediaPipe detection results into FaceDetection objects.
        
        Args:
            results: MediaPipe detection results.
            image: Input image as numpy array.
            
        Returns:
            List of FaceDetection objects.
        """
        height, width, _ = image.shape
        detections = []
        
        if results.detections:
            for detection in results.detections:
                # Get score
                score = detection.score[0] if detection.score else 0
                
                # Skip detections below threshold
                if score < self.confidence_threshold:
                    continue
                
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * width)
                y1 = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                # Create detection
                face_detection = FaceDetection(
                    bbox=BoundingBox(x1=x1, y1=y1, width=w, height=h),
                    confidence=score,
                    landmarks=self._extract_landmarks(detection, width, height)
                )
                
                detections.append(face_detection)
        
        return detections
    
    def _extract_landmarks(self, detection, width: int, height: int) -> Dict[str, Tuple[int, int]]:
        """
        Extract facial landmarks from MediaPipe detection.
        
        Args:
            detection: MediaPipe detection result.
            width: Image width.
            height: Image height.
            
        Returns:
            Dictionary of landmarks with normalized coordinates.
        """
        landmarks = {}
        
        # MediaPipe provides 6 landmarks:
        # 0: right eye
        # 1: left eye
        # 2: nose tip
        # 3: mouth center
        # 4: right ear tragion
        # 5: left ear tragion
        landmark_names = ["right_eye", "left_eye", "nose_tip", "mouth_center", 
                          "right_ear_tragion", "left_ear_tragion"]
        
        for idx, name in enumerate(landmark_names):
            try:
                if idx < len(detection.location_data.relative_keypoints):
                    keypoint = detection.location_data.relative_keypoints[idx]
                    landmarks[name] = (int(keypoint.x * width), int(keypoint.y * height))
            except (IndexError, AttributeError):
                # Skip if landmark is not available
                pass
        
        return landmarks
    
    def visualize(self, image: np.ndarray, detections: List[FaceDetection]) -> np.ndarray:
        """
        Visualize face detections on the image.
        
        Args:
            image: Input image as numpy array.
            detections: List of FaceDetection objects.
            
        Returns:
            Image with visualized detections.
        """
        # Create a copy of the image
        vis_image = image.copy()
        
        for detection in detections:
            # Draw bounding box
            x1, y1, x2, y2 = detection.bbox.to_xyxy()
            color = (0, 255, 0)  # Green color for face
            
            # Draw rectangle with confidence score
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            text = f"Face: {detection.confidence:.2f}"
            cv2.putText(vis_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw landmarks
            for name, (x, y) in detection.landmarks.items():
                cv2.circle(vis_image, (x, y), 2, (255, 0, 0), -1)
        
        return vis_image 