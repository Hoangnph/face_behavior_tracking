import unittest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

# Import will be available after implementation
from src.detection.detection_scheduler import DetectionScheduler
from src.detection.base_detection import Detection, BoundingBox


class TestDetectionScheduler(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create a sample test image
        self.test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Create mock detectors
        self.mock_face_detector = MagicMock()
        self.mock_person_detector = MagicMock()
    
    def test_initialization(self):
        """Test that the scheduler initializes correctly"""
        scheduler = DetectionScheduler(
            detectors={
                "face": self.mock_face_detector,
                "person": self.mock_person_detector
            },
            default_detector="face"
        )
        
        self.assertEqual(scheduler.default_detector, "face")
        self.assertEqual(len(scheduler.detectors), 2)
        self.assertIn("face", scheduler.detectors)
        self.assertIn("person", scheduler.detectors)
    
    def test_detect_with_default_detector(self):
        """Test detection with default detector"""
        # Set up mock face detector
        face_bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        face_detection = Detection(bbox=face_bbox, confidence=0.9, label="face")
        self.mock_face_detector.detect.return_value = [face_detection]
        
        # Initialize scheduler with default detector
        scheduler = DetectionScheduler(
            detectors={"face": self.mock_face_detector},
            default_detector="face"
        )
        
        # Run detection
        detections = scheduler.detect(self.test_image)
        
        # Assertions
        self.mock_face_detector.detect.assert_called_once_with(self.test_image)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "face")
        self.assertEqual(detections[0].bbox.x1, 100)
    
    def test_detect_with_specified_detector(self):
        """Test detection with explicitly specified detector"""
        # Set up mock person detector
        person_bbox = BoundingBox(x1=200, y1=200, x2=400, y2=600)
        person_detection = Detection(bbox=person_bbox, confidence=0.8, label="person")
        self.mock_person_detector.detect.return_value = [person_detection]
        
        # Initialize scheduler with face as default
        scheduler = DetectionScheduler(
            detectors={
                "face": self.mock_face_detector,
                "person": self.mock_person_detector
            },
            default_detector="face"
        )
        
        # Run detection with person detector
        detections = scheduler.detect(self.test_image, detector_name="person")
        
        # Assertions
        self.mock_person_detector.detect.assert_called_once_with(self.test_image)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "person")
        self.assertEqual(detections[0].bbox.x1, 200)
    
    def test_detect_with_all_detectors(self):
        """Test detection with all detectors"""
        # Set up mock detectors
        face_bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        face_detection = Detection(bbox=face_bbox, confidence=0.9, label="face")
        self.mock_face_detector.detect.return_value = [face_detection]
        
        person_bbox = BoundingBox(x1=200, y1=200, x2=400, y2=600)
        person_detection = Detection(bbox=person_bbox, confidence=0.8, label="person")
        self.mock_person_detector.detect.return_value = [person_detection]
        
        # Initialize scheduler
        scheduler = DetectionScheduler(
            detectors={
                "face": self.mock_face_detector,
                "person": self.mock_person_detector
            },
            default_detector="face"
        )
        
        # Run detection with all detectors
        detections = scheduler.detect_all(self.test_image)
        
        # Assertions
        self.mock_face_detector.detect.assert_called_once_with(self.test_image)
        self.mock_person_detector.detect.assert_called_once_with(self.test_image)
        self.assertEqual(len(detections), 2)
        self.assertIn("face", [d.label for d in detections])
        self.assertIn("person", [d.label for d in detections])
    
    def test_invalid_detector_name(self):
        """Test behavior with invalid detector name"""
        scheduler = DetectionScheduler(
            detectors={"face": self.mock_face_detector},
            default_detector="face"
        )
        
        # Should raise ValueError with invalid detector name
        with self.assertRaises(ValueError):
            scheduler.detect(self.test_image, detector_name="invalid_detector")
    
    def test_detection_frequency(self):
        """Test running detectors at different frequencies"""
        # Set up mock detectors
        face_detection = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
            confidence=0.9, 
            label="face"
        )
        self.mock_face_detector.detect.return_value = [face_detection]
        
        person_detection = Detection(
            bbox=BoundingBox(x1=200, y1=200, x2=400, y2=600),
            confidence=0.8, 
            label="person"
        )
        self.mock_person_detector.detect.return_value = [person_detection]
        
        # Initialize scheduler with frequencies
        scheduler = DetectionScheduler(
            detectors={
                "face": self.mock_face_detector,
                "person": self.mock_person_detector
            },
            default_detector="face",
            frequencies={
                "face": 1,      # Run every frame
                "person": 5     # Run every 5 frames
            }
        )
        
        # Simulate 10 frames
        all_detections = []
        for i in range(10):
            detections = scheduler.detect_with_frequency(self.test_image, frame_number=i)
            all_detections.append(detections)
        
        # Face detector should run on all 10 frames
        self.assertEqual(self.mock_face_detector.detect.call_count, 10)
        
        # Person detector should run on frames 0, 5
        self.assertEqual(self.mock_person_detector.detect.call_count, 2)
        
        # Frames 0, 5 should have both face and person detections
        self.assertEqual(len(all_detections[0]), 2)
        self.assertEqual(len(all_detections[5]), 2)
        
        # Other frames should only have face detections
        for i in [1, 2, 3, 4, 6, 7, 8, 9]:
            self.assertEqual(len(all_detections[i]), 1)
            self.assertEqual(all_detections[i][0].label, "face")


if __name__ == '__main__':
    unittest.main() 