"""
Base detection module that provides common structures and interfaces for all detection components.
"""
from typing import List, Union, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class BoundingBox:
    """
    Represents a bounding box with various coordinate formats.
    Can be initialized with either:
    1. x1, y1, x2, y2 (top-left and bottom-right corners)
    2. x1, y1, width, height (top-left corner, width, and height)
    3. xmin, ymin, width, height (same as above, alternate naming)
    """
    x1: Optional[int] = None
    y1: Optional[int] = None
    x2: Optional[int] = None
    y2: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    xmin: Optional[int] = None
    ymin: Optional[int] = None
    
    def __post_init__(self):
        """Compute missing values based on provided coordinates."""
        # Handle alternate naming (xmin, ymin)
        if self.xmin is not None and self.x1 is None:
            self.x1 = self.xmin
        if self.ymin is not None and self.y1 is None:
            self.y1 = self.ymin
            
        # If x1, y1, width, height are provided
        if self.x1 is not None and self.y1 is not None and self.width is not None and self.height is not None:
            self.x2 = self.x1 + self.width
            self.y2 = self.y1 + self.height
            
        # If x1, y1, x2, y2 are provided
        elif self.x1 is not None and self.y1 is not None and self.x2 is not None and self.y2 is not None:
            self.width = self.x2 - self.x1
            self.height = self.y2 - self.y1
            
        else:
            raise ValueError("Either (x1, y1, width, height) or (x1, y1, x2, y2) must be provided")
        
        # Ensure values are integers
        self.x1 = int(self.x1)
        self.y1 = int(self.y1)
        self.x2 = int(self.x2)
        self.y2 = int(self.y2)
        self.width = int(self.width)
        self.height = int(self.height)
        
    def to_xyxy(self) -> Tuple[int, int, int, int]:
        """Return bounding box as (x1, y1, x2, y2) format."""
        return self.x1, self.y1, self.x2, self.y2
        
    def to_xywh(self) -> Tuple[int, int, int, int]:
        """Return bounding box as (x, y, width, height) format."""
        return self.x1, self.y1, self.width, self.height
    
    def to_relative(self, image_width: int, image_height: int) -> Tuple[float, float, float, float]:
        """Convert to relative coordinates (0-1 range) in xywh format."""
        return (
            self.x1 / image_width,
            self.y1 / image_height,
            self.width / image_width,
            self.height / image_height
        )
    
    def area(self) -> int:
        """Return the area of the bounding box."""
        return self.width * self.height
    
    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int) -> 'BoundingBox':
        """Create a BoundingBox from (x1, y1, x2, y2) format."""
        return cls(x1=x1, y1=y1, x2=x2, y2=y2)
        
    @classmethod
    def from_xywh(cls, x: int, y: int, width: int, height: int) -> 'BoundingBox':
        """Create a BoundingBox from (x, y, width, height) format."""
        return cls(x1=x, y1=y, width=width, height=height)
    
    @classmethod
    def from_relative(cls, rel_x: float, rel_y: float, rel_w: float, rel_h: float, 
                      image_width: int, image_height: int) -> 'BoundingBox':
        """Create a BoundingBox from relative coordinates (0-1 range)."""
        return cls(
            x1=int(rel_x * image_width),
            y1=int(rel_y * image_height),
            width=int(rel_w * image_width),
            height=int(rel_h * image_height)
        )


@dataclass
class Detection:
    """
    Represents a single detection with bounding box, confidence score, and label.
    """
    bbox: BoundingBox
    confidence: float
    label: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary format for serialization."""
        return {
            "bbox": {
                "x1": self.bbox.x1,
                "y1": self.bbox.y1,
                "x2": self.bbox.x2,
                "y2": self.bbox.y2,
                "width": self.bbox.width,
                "height": self.bbox.height
            },
            "confidence": float(self.confidence),
            "label": self.label
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Detection':
        """Create a Detection from dictionary format."""
        bbox = BoundingBox(
            x1=data["bbox"]["x1"],
            y1=data["bbox"]["y1"],
            x2=data["bbox"]["x2"],
            y2=data["bbox"]["y2"]
        )
        return cls(
            bbox=bbox,
            confidence=data["confidence"],
            label=data["label"]
        )


class BaseDetector:
    """
    Base class for all detection models.
    """
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the detector.
        
        Args:
            confidence_threshold: Minimum confidence threshold for detections.
        """
        self.confidence_threshold = confidence_threshold
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detect objects in the input image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV).
            
        Returns:
            List of Detection objects.
        """
        raise NotImplementedError("Subclasses must implement detect() method")
    
    def _filter_detections(self, detections: List[Detection]) -> List[Detection]:
        """
        Filter detections based on confidence threshold.
        
        Args:
            detections: List of Detection objects.
            
        Returns:
            Filtered list of Detection objects.
        """
        return [d for d in detections if d.confidence >= self.confidence_threshold] 