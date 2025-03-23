"""
Person detection module using YOLOv8 object detection.
"""
from typing import List, Union, Dict, Any, Optional, Tuple
import os
import numpy as np
import cv2
from ultralytics import YOLO
from .base_detection import BaseDetector, Detection, BoundingBox


class PersonDetection(Detection):
    """
    Person detection result.
    Extends the base Detection class with additional person-specific information.
    """
    def __init__(self, 
                 bbox: BoundingBox, 
                 confidence: float, 
                 label: str = "person",
                 track_id: Optional[int] = None):
        """
        Initialize person detection.
        
        Args:
            bbox: Bounding box of the detected person.
            confidence: Detection confidence score.
            label: Detection label, defaults to "person".
            track_id: Tracking ID if available.
        """
        super().__init__(bbox=bbox, confidence=confidence, label=label)
        self.track_id = track_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert person detection to dictionary format for serialization."""
        data = super().to_dict()
        if self.track_id is not None:
            data["track_id"] = self.track_id
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonDetection':
        """Create a PersonDetection from dictionary format."""
        detection = super().from_dict(data)
        return cls(
            bbox=detection.bbox,
            confidence=detection.confidence,
            label=detection.label,
            track_id=data.get("track_id")
        )


class YOLOPersonDetector(BaseDetector):
    """
    Person detector using YOLOv8 object detection.
    """
    def __init__(self, 
                 model_path: str = "models/yolo/yolov8n.pt",
                 confidence_threshold: float = 0.5,
                 device: Optional[str] = None):
        """
        Initialize YOLOv8 person detector.
        
        Args:
            model_path: Path to YOLOv8 model file.
            confidence_threshold: Minimum confidence threshold for detections.
            device: Device to run inference on ('cpu', 'cuda', 'mps', etc.).
                    If None, will use the best available device.
        """
        super().__init__(confidence_threshold=confidence_threshold)
        
        # Ensure model directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Load model
        self.model = YOLO(model_path)
        
        # Set device if provided
        if device:
            self.model.to(device)
    
    def detect(self, image: np.ndarray) -> List[PersonDetection]:
        """
        Detect persons in the input image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV).
            
        Returns:
            List of PersonDetection objects.
        """
        # Run inference
        results = self.model(image, verbose=False)
        
        # Process results
        return self._process_results(results, image)
    
    def _process_results(self, results, image: np.ndarray) -> List[PersonDetection]:
        """
        Process YOLOv8 results into PersonDetection objects.
        
        Args:
            results: YOLOv8 results object.
            image: Input image.
            
        Returns:
            List of PersonDetection objects.
        """
        height, width = image.shape[:2]
        detections = []
        
        for result in results:
            # Get boxes
            boxes = result.boxes
            
            # Process each detection
            for i, box in enumerate(boxes):
                # Get class ID and filter for persons (class 0 in COCO dataset)
                cls_id = int(box.cls[0].item() if len(box.cls) > 0 else -1)
                if cls_id != 0:  # Not a person
                    continue
                
                # Get confidence
                conf = float(box.conf[0].item() if len(box.conf) > 0 else 0)
                
                # Skip detections below threshold
                if conf < self.confidence_threshold:
                    continue
                
                # Get bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Create detection
                person_detection = PersonDetection(
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=conf
                )
                
                detections.append(person_detection)
        
        return detections
    
    def visualize(self, image: np.ndarray, detections: List[PersonDetection]) -> np.ndarray:
        """
        Visualize person detections on the image.
        
        Args:
            image: Input image as numpy array.
            detections: List of PersonDetection objects.
            
        Returns:
            Image with visualized detections.
        """
        # Create a copy of the image
        vis_image = image.copy()
        
        for detection in detections:
            # Draw bounding box
            x1, y1, x2, y2 = detection.bbox.to_xyxy()
            color = (0, 0, 255)  # Red color for person
            
            # Draw rectangle with confidence score
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Add track ID if available
            if detection.track_id is not None:
                text = f"Person #{detection.track_id}: {detection.confidence:.2f}"
            else:
                text = f"Person: {detection.confidence:.2f}"
                
            cv2.putText(vis_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_image 