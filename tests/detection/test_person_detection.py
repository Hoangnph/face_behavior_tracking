import unittest
import os
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

# Import will be available after implementation
from src.detection.person_detection import YOLOPersonDetector, PersonDetection


class TestYOLOPersonDetector(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create a sample test image
        self.test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        # Draw a simple person-like shape
        cv2.rectangle(self.test_image, (200, 100), (400, 600), (200, 200, 200), -1)  # Body
        cv2.circle(self.test_image, (300, 75), 50, (200, 200, 200), -1)  # Head
        
        # Ensure the models directory exists
        os.makedirs(os.path.join("models", "yolo"), exist_ok=True)
    
    @patch('ultralytics.YOLO')
    def test_initialization(self, mock_yolo_class):
        """Test that the detector initializes correctly"""
        # Set up mock
        mock_instance = MagicMock()
        mock_yolo_class.return_value = mock_instance
        
        # Initialize detector
        detector = YOLOPersonDetector(model_path="models/yolo/yolov8n.pt", confidence_threshold=0.5)
        
        # Check initialization
        mock_yolo_class.assert_called_once_with("models/yolo/yolov8n.pt")
        self.assertEqual(detector.confidence_threshold, 0.5)
        self.assertIs(detector.model, mock_instance)
    
    @patch('ultralytics.YOLO')
    def test_detect_persons(self, mock_yolo_class):
        """Test person detection with mocked YOLO model"""
        # Set up mock YOLO instance
        mock_instance = MagicMock()
        mock_yolo_class.return_value = mock_instance
        
        # Create mock detection result with one person
        mock_box = MagicMock()
        mock_box.cls = np.array([0])  # Class 0 = person
        mock_box.conf = np.array([0.85])
        mock_box.xyxy = np.array([[100, 200, 300, 500]])  # x1, y1, x2, y2
        
        # Create mock result object
        mock_result = MagicMock()
        mock_result.boxes = mock_box
        
        # Set mock return value for YOLO model inference
        mock_instance.return_value = [mock_result]
        
        # Initialize and run detector
        detector = YOLOPersonDetector()
        detections = detector.detect(self.test_image)
        
        # Assertions
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].confidence, 0.85)
        self.assertEqual(detections[0].bbox.x1, 100)
        self.assertEqual(detections[0].bbox.y1, 200)
        self.assertEqual(detections[0].bbox.x2, 300)
        self.assertEqual(detections[0].bbox.y2, 500)
        self.assertEqual(detections[0].bbox.width, 200)
        self.assertEqual(detections[0].bbox.height, 300)
    
    @patch('ultralytics.YOLO')
    def test_filter_non_person_detections(self, mock_yolo_class):
        """Test that non-person detections are filtered out"""
        # Set up mock YOLO instance
        mock_instance = MagicMock()
        mock_yolo_class.return_value = mock_instance
        
        # Create mock with mixed detections (person and non-person)
        mock_box = MagicMock()
        mock_box.cls = np.array([0, 2, 1])  # Only first is person (class 0)
        mock_box.conf = np.array([0.9, 0.8, 0.7])
        mock_box.xyxy = np.array([
            [100, 200, 300, 400],  # Person
            [150, 250, 350, 450],  # Non-person (class 2)
            [200, 300, 400, 500]   # Non-person (class 1)
        ])
        
        # Setup iteration for the boxes
        def get_box(idx):
            box = MagicMock()
            box.cls = np.array([mock_box.cls[idx]])
            box.conf = np.array([mock_box.conf[idx]])
            box.xyxy = np.array([mock_box.xyxy[idx]])
            return box
        
        mock_box.__iter__.return_value = [get_box(0), get_box(1), get_box(2)]
        
        # Create mock result object
        mock_result = MagicMock()
        mock_result.boxes = mock_box
        
        # Set mock return value for YOLO model inference
        mock_instance.return_value = [mock_result]
        
        # Initialize and run detector
        detector = YOLOPersonDetector()
        detections = detector.detect(self.test_image)
        
        # Should only return the person detection
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].bbox.x1, 100)
        self.assertEqual(detections[0].bbox.y1, 200)
    
    @patch('ultralytics.YOLO')
    def test_confidence_threshold(self, mock_yolo_class):
        """Test that detections below confidence threshold are filtered out"""
        # Set up mock YOLO instance
        mock_instance = MagicMock()
        mock_yolo_class.return_value = mock_instance
        
        # Create mock with multiple person detections of varying confidence
        mock_box = MagicMock()
        mock_box.cls = np.array([0, 0, 0])  # All persons
        mock_box.conf = np.array([0.9, 0.6, 0.3])
        mock_box.xyxy = np.array([
            [100, 200, 300, 400],  # High confidence
            [150, 250, 350, 450],  # Medium confidence
            [200, 300, 400, 500]   # Low confidence
        ])
        
        # Setup iteration for the boxes
        def get_box(idx):
            box = MagicMock()
            box.cls = np.array([mock_box.cls[idx]])
            box.conf = np.array([mock_box.conf[idx]])
            box.xyxy = np.array([mock_box.xyxy[idx]])
            return box
        
        mock_box.__iter__.return_value = [get_box(0), get_box(1), get_box(2)]
        
        # Create mock result object
        mock_result = MagicMock()
        mock_result.boxes = mock_box
        
        # Set mock return value for YOLO model inference
        mock_instance.return_value = [mock_result]
        
        # Initialize with 0.7 threshold and run detector
        detector = YOLOPersonDetector(confidence_threshold=0.7)
        detections = detector.detect(self.test_image)
        
        # Should only return the high confidence detection
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].confidence, 0.9)
    
    @patch('ultralytics.YOLO')
    def test_no_detections(self, mock_yolo_class):
        """Test behavior when no detections are found"""
        # Set up mock YOLO instance with empty results
        mock_instance = MagicMock()
        mock_yolo_class.return_value = mock_instance
        
        # Create mock with no detections
        mock_box = MagicMock()
        mock_box.cls = np.array([])
        mock_box.conf = np.array([])
        mock_box.xyxy = np.array([])
        
        # Make iteration return empty list
        mock_box.__iter__.return_value = []
        
        # Create mock result object
        mock_result = MagicMock()
        mock_result.boxes = mock_box
        
        # Set mock return value for YOLO model inference
        mock_instance.return_value = [mock_result]
        
        # Initialize and run detector
        detector = YOLOPersonDetector()
        detections = detector.detect(self.test_image)
        
        # Should return empty list
        self.assertEqual(len(detections), 0)


if __name__ == '__main__':
    unittest.main() 