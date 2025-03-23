import unittest
import os
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

# Import will be available after implementation
from src.detection.face_detection import MediaPipeFaceDetector, FaceDetection


class TestMediaPipeFaceDetector(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create a sample test image
        self.test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        # Draw a simple face-like shape for basic testing
        cv2.circle(self.test_image, (150, 150), 100, (200, 200, 200), -1)  # Head
        cv2.circle(self.test_image, (120, 120), 15, (0, 0, 0), -1)  # Left eye
        cv2.circle(self.test_image, (180, 120), 15, (0, 0, 0), -1)  # Right eye
        cv2.rectangle(self.test_image, (130, 180), (170, 200), (0, 0, 0), -1)  # Mouth
        
        # Ensure the models directory exists
        os.makedirs(os.path.join("models", "mediapipe"), exist_ok=True)
    
    def test_initialization(self):
        """Test that the detector initializes correctly"""
        detector = MediaPipeFaceDetector(min_detection_confidence=0.5)
        self.assertEqual(detector.min_detection_confidence, 0.5)
        self.assertIsNotNone(detector.face_detection)
    
    @patch('mediapipe.solutions.face_detection.FaceDetection')
    def test_detect_faces(self, mock_face_detection_class):
        """Test face detection with mocked MediaPipe"""
        # Setup mock
        mock_face_detection = MagicMock()
        mock_face_detection_class.return_value.__enter__.return_value = mock_face_detection
        
        # Create mock detection results
        mock_detection = MagicMock()
        mock_bbox = MagicMock()
        mock_bbox.xmin = 0.1
        mock_bbox.ymin = 0.1
        mock_bbox.width = 0.8
        mock_bbox.height = 0.8
        
        mock_detection.location_data.relative_bounding_box = mock_bbox
        mock_detection.score = [0.95]
        
        mock_result = MagicMock()
        mock_result.detections = [mock_detection]
        mock_face_detection.process.return_value = mock_result
        
        # Initialize and run detector
        detector = MediaPipeFaceDetector()
        faces = detector.detect(self.test_image)
        
        # Assertions
        self.assertEqual(len(faces), 1)
        self.assertAlmostEqual(faces[0].confidence, 0.95)
        self.assertEqual(faces[0].bbox.x1, int(0.1 * 300))
        self.assertEqual(faces[0].bbox.y1, int(0.1 * 300))
        self.assertEqual(faces[0].bbox.width, int(0.8 * 300))
        self.assertEqual(faces[0].bbox.height, int(0.8 * 300))
    
    def test_detect_no_faces(self):
        """Test behavior when no faces are detected"""
        # Create a blank image with no faces
        blank_image = np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Initialize and run detector
        detector = MediaPipeFaceDetector(min_detection_confidence=0.9)
        faces = detector.detect(blank_image)
        
        # Should return empty list
        self.assertEqual(len(faces), 0)
    
    def test_rgb_conversion(self):
        """Test that BGR to RGB conversion is performed correctly"""
        # Create a test image with specific BGR values
        bgr_image = np.zeros((10, 10, 3), dtype=np.uint8)
        bgr_image[0, 0] = [255, 0, 0]  # Blue pixel in BGR
        
        with patch('src.detection.face_detection.MediaPipeFaceDetector._process_detections') as mock_process:
            mock_process.return_value = []
            detector = MediaPipeFaceDetector()
            detector.detect(bgr_image)
            
            # Get the RGB image that was passed to the detector
            args, _ = mock_process.call_args
            rgb_image = args[1]
            
            # The blue pixel should now be red
            self.assertEqual(list(rgb_image[0, 0]), [0, 0, 255])
    
    def test_threshold_filtering(self):
        """Test that detections below threshold are filtered out"""
        with patch('mediapipe.solutions.face_detection.FaceDetection') as mock_face_detection_class:
            # Setup mock
            mock_face_detection = MagicMock()
            mock_face_detection_class.return_value.__enter__.return_value = mock_face_detection
            
            # Create two mock detections with different confidence scores
            mock_detection1 = MagicMock()
            mock_bbox1 = MagicMock()
            mock_bbox1.xmin = 0.1
            mock_bbox1.ymin = 0.1
            mock_bbox1.width = 0.2
            mock_bbox1.height = 0.2
            mock_detection1.location_data.relative_bounding_box = mock_bbox1
            mock_detection1.score = [0.75]
            
            mock_detection2 = MagicMock()
            mock_bbox2 = MagicMock()
            mock_bbox2.xmin = 0.5
            mock_bbox2.ymin = 0.5
            mock_bbox2.width = 0.2
            mock_bbox2.height = 0.2
            mock_detection2.location_data.relative_bounding_box = mock_bbox2
            mock_detection2.score = [0.45]
            
            mock_result = MagicMock()
            mock_result.detections = [mock_detection1, mock_detection2]
            mock_face_detection.process.return_value = mock_result
            
            # Set threshold to 0.5, only the first detection should be returned
            detector = MediaPipeFaceDetector(min_detection_confidence=0.5)
            faces = detector.detect(self.test_image)
            
            self.assertEqual(len(faces), 1)
            self.assertAlmostEqual(faces[0].confidence, 0.75)


if __name__ == '__main__':
    unittest.main() 