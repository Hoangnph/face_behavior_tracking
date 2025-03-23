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
                 confidence_threshold: float = 0.15,  # Reduced further to improve detection rate
                 device: Optional[str] = None,
                 input_size: Tuple[int, int] = (640, 640),  # Using standard size for better accuracy
                 size_factor: float = 0.75):  # Increased from 0.5 to 0.75 for better accuracy
        """
        Initialize YOLOv8 person detector.
        
        Args:
            model_path: Path to YOLOv8 model file.
            confidence_threshold: Minimum confidence threshold for detections.
                                 Lower values increase detection rate at cost of more false positives.
            device: Device to run inference on ('cpu', 'cuda', 'mps', etc.).
                    If None, will use the best available device.
            input_size: Input size for the model (width, height) to optimize performance.
            size_factor: Scale factor to resize input images before detection (0.5 = half size).
                        Lower values improve performance at cost of accuracy.
        """
        super().__init__(confidence_threshold=confidence_threshold)
        
        # Ensure model directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Load model
        self.model = YOLO(model_path)
        
        # Set device if provided
        if device:
            self.model.to(device)
            
        # Store input size for resizing
        self.input_size = input_size
        self.size_factor = size_factor
    
    def detect(self, image: np.ndarray) -> List[PersonDetection]:
        """
        Detect persons in the input image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV).
            
        Returns:
            List of PersonDetection objects.
        """
        # Store original dimensions for scaling back
        original_height, original_width = image.shape[:2]
        
        # Resize image by factor to improve performance
        if self.size_factor != 1.0:
            working_width = int(original_width * self.size_factor)
            working_height = int(original_height * self.size_factor)
            working_image = cv2.resize(image, (working_width, working_height))
        else:
            working_width, working_height = original_width, original_height
            working_image = image
        
        # Prepare image for model
        if (working_width, working_height) != self.input_size:
            # Keep aspect ratio and resize
            model_input = self._resize_with_aspect_ratio(working_image, self.input_size)
        else:
            model_input = working_image
        
        # Run inference
        results = self.model(model_input, verbose=False)
        
        # Process results
        detections = self._process_results(
            results, 
            original_width, 
            original_height, 
            working_width, 
            working_height
        )
        
        return detections
    
    def _resize_with_aspect_ratio(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image
            target_size: Target size (width, height)
            
        Returns:
            Resized image
        """
        # Ensure target dimensions are integers
        target_width, target_height = int(target_size[0]), int(target_size[1])
        h, w = image.shape[:2]
        
        # Create output image with target size
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # If image exceeds target dimensions, scale down
        if w > target_width or h > target_height:
            # Calculate scale to fit inside target
            scale = min(target_width / w, target_height / h)
            new_width = int(w * scale)
            new_height = int(h * scale)
            
            # Resize using direct resize (no need for canvas)
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Center on canvas
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            
            # Copy resized image to canvas
            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        else:
            # Image is smaller, scale up
            scale = min(target_width / w, target_height / h)
            new_width = int(w * scale)
            new_height = int(h * scale)
            
            # Resize using direct resize
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Center on canvas
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            
            # Copy resized image to canvas
            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            
        return canvas
    
    def _process_results(
        self, 
        results, 
        original_width: int, 
        original_height: int,
        working_width: int = None,
        working_height: int = None
    ) -> List[PersonDetection]:
        """
        Process YOLOv8 results into PersonDetection objects.
        
        Args:
            results: YOLOv8 results object.
            original_width: Original image width.
            original_height: Original image height.
            working_width: Width of image after initial resize.
            working_height: Height of image after initial resize.
            
        Returns:
            List of PersonDetection objects.
        """
        detections = []
        
        # Use working dimensions as default if not provided
        if working_width is None:
            working_width = original_width
        if working_height is None:
            working_height = original_height
        
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
                
                # Get bounding box in normalized coordinates
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                
                # Scale coordinates to working image size if necessary
                input_width, input_height = self.input_size
                if (working_width, working_height) != self.input_size:
                    # Calculate scale and offsets
                    scale = min(input_width / working_width, input_height / working_height)
                    new_width = int(working_width * scale)
                    new_height = int(working_height * scale)
                    
                    x_offset = (input_width - new_width) // 2
                    y_offset = (input_height - new_height) // 2
                    
                    # Adjust coordinates to working image size
                    x1 = (x1 - x_offset) / scale
                    y1 = (y1 - y_offset) / scale
                    x2 = (x2 - x_offset) / scale
                    y2 = (y2 - y_offset) / scale
                
                # Scale back to original image size if necessary
                if working_width != original_width or working_height != original_height:
                    scale_x = original_width / working_width
                    scale_y = original_height / working_height
                    
                    x1 *= scale_x
                    y1 *= scale_y
                    x2 *= scale_x
                    y2 *= scale_y
                
                # Clip to image boundaries
                x1 = max(0, min(original_width, int(x1)))
                y1 = max(0, min(original_height, int(y1)))
                x2 = max(0, min(original_width, int(x2)))
                y2 = max(0, min(original_height, int(y2)))
                
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