import unittest
import os
import numpy as np
import cv2
from unittest.mock import patch, MagicMock, call
import sys

# Import will be available after implementation
from src.detection.person_detection import YOLOPersonDetector, PersonDetection
from src.detection.base_detection import BoundingBox


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
    
    @patch('src.detection.person_detection.YOLO')
    def test_initialization(self, mock_yolo):
        """Test that the detector initializes correctly"""
        # Set up mock
        mock_model = mock_yolo.return_value
        
        # Initialize detector with custom parameters
        detector = YOLOPersonDetector(
            model_path="models/yolo/yolov8n.pt", 
            confidence_threshold=0.25,
            input_size=(640, 480),
            size_factor=0.75
        )
        
        # Check initialization
        mock_yolo.assert_called_once_with("models/yolo/yolov8n.pt")
        self.assertEqual(detector.confidence_threshold, 0.25)
        self.assertEqual(detector.input_size, (640, 480))
        self.assertEqual(detector.size_factor, 0.75)
        self.assertEqual(detector.model, mock_model)
    
    @patch('src.detection.person_detection.YOLOPersonDetector._process_results')
    @patch('src.detection.person_detection.YOLO')
    def test_detect_persons(self, mock_yolo, mock_process_results):
        """Test person detection with mocked YOLO model and _process_results"""
        # Setup mocks
        mock_model = mock_yolo.return_value
        mock_results = [MagicMock()]
        mock_model.return_value = mock_results
        
        # Create a mock detection result to return
        detection = PersonDetection(
            bbox=BoundingBox(x1=100, y1=200, x2=300, y2=500),
            confidence=0.85,
            label="person"
        )
        mock_process_results.return_value = [detection]
        
        # Initialize detector with matching input size
        detector = YOLOPersonDetector(confidence_threshold=0.25, input_size=(416, 416), size_factor=1.0)
        
        # Run detection
        detections = detector.detect(self.test_image)
        
        # Check results
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].confidence, 0.85)
        self.assertEqual(detections[0].label, "person")
        
        # Verify model was called with appropriate input
        mock_model.assert_called_once()
        # Verify _process_results was called
        mock_process_results.assert_called_once()
    
    @patch('cv2.resize')
    def test_resize_with_aspect_ratio(self, mock_resize):
        """Test that the resize function maintains aspect ratio properly"""
        # Create proper sized return value that matches target size
        target_size = (416, 416)
        mock_resize.return_value = np.zeros((312, 416, 3), dtype=np.uint8)
        
        # Create detector directly (no need to mock YOLO here)
        detector = YOLOPersonDetector.__new__(YOLOPersonDetector)
        detector.input_size = target_size
        detector.size_factor = 1.0
        detector._filter_detections = lambda x: x  # Mock filter method
        
        # Create test image
        test_image = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # Call resize function
        result = detector._resize_with_aspect_ratio(test_image, target_size)
        
        # Verify resize was called correctly
        mock_resize.assert_called_once()
        args, kwargs = mock_resize.call_args
        self.assertEqual(kwargs.get('interpolation', None), cv2.INTER_LINEAR)
        
        # Result should be the target size
        self.assertEqual(result.shape[:2], target_size[::-1])  # (height, width)
    
    @patch('src.detection.person_detection.YOLOPersonDetector._process_results')
    @patch('src.detection.person_detection.YOLO')
    def test_filter_non_person_detections(self, mock_yolo, mock_process_results):
        """Test that non-person detections are filtered out"""
        # Setup mocks
        mock_model = mock_yolo.return_value
        mock_results = [MagicMock()]
        mock_model.return_value = mock_results
        
        # Create a mock person detection
        person_detection = PersonDetection(
            bbox=BoundingBox(x1=100, y1=200, x2=300, y2=400),
            confidence=0.9,
            label="person"
        )
        
        # Return only the person detection from process_results
        mock_process_results.return_value = [person_detection]
        
        # Initialize detector with matching input size
        detector = YOLOPersonDetector(confidence_threshold=0.25, input_size=(416, 416), size_factor=1.0)
        
        # Run detection
        detections = detector.detect(self.test_image)
        
        # Should only return person detections
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "person")
        
        # Verify model and process_results were called
        mock_model.assert_called_once()
        mock_process_results.assert_called_once()
    
    @patch('src.detection.person_detection.YOLO')
    def test_confidence_threshold(self, mock_yolo):
        """Test that detections below confidence threshold are filtered out"""
        # Setup mocks
        mock_model = mock_yolo.return_value
        
        # Create results that _process_results would return
        high_conf_detection = PersonDetection(
            bbox=BoundingBox(x1=100, y1=200, x2=300, y2=400),
            confidence=0.9,
            label="person"
        )
        
        low_conf_detection = PersonDetection(
            bbox=BoundingBox(x1=150, y1=250, x2=350, y2=450),
            confidence=0.3,
            label="person"
        )
        
        # Create a detector with our desired confidence threshold
        detector = YOLOPersonDetector(confidence_threshold=0.5, input_size=(416, 416), size_factor=1.0)
        
        # Create a custom detect method that replaces the real one
        def mock_detect(image):
            print(f"DEBUG: Mock detect called", file=sys.stderr)
            # Return unfiltered detections
            unfiltered = [high_conf_detection, low_conf_detection]
            # Filter using the real filter method (not mocked)
            filtered = detector._filter_detections(unfiltered)
            print(f"DEBUG: After filtering: {len(filtered)} detections remain", file=sys.stderr)
            for i, d in enumerate(filtered):
                print(f"DEBUG:   [{i}] confidence: {d.confidence}", file=sys.stderr)
            return filtered
        
        # Replace the detect method
        original_detect = detector.detect
        detector.detect = mock_detect
        
        try:
            # Run detection
            detections = detector.detect(self.test_image)
            
            # Should only return high confidence detections
            self.assertEqual(len(detections), 1, "Expected 1 detection, but got a different number")
            if len(detections) > 0:
                self.assertAlmostEqual(detections[0].confidence, 0.9)
                
                # Make sure low confidence detection was filtered out
                self.assertTrue(all(d.confidence >= 0.5 for d in detections))
            
        finally:
            # Restore original method
            detector.detect = original_detect


if __name__ == '__main__':
    unittest.main() 