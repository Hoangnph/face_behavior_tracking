import unittest
import os
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

# Import will be available after implementation
from src.detection.onnx_detection import ONNXDetector, convert_to_onnx


class TestONNXIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create a sample test image
        self.test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Ensure the models directory exists
        os.makedirs(os.path.join("models", "onnx"), exist_ok=True)
        
        # Mock ONNX runtime session
        self.mock_session = MagicMock()
        self.mock_session.get_inputs.return_value = [MagicMock(name="input", shape=[1, 3, 640, 640])]
        self.mock_session.get_outputs.return_value = [MagicMock(name="output")]
        
        # Mock detection outputs - adjust to match YOLOv8 output format
        mock_output = np.zeros((1, 8400, 85), dtype=np.float32)  # YOLOv8 format
        
        # Add one detection with all necessary values
        # Format: [x, y, w, h, confidence, class probabilities (80 classes for COCO)]
        detection_data = np.zeros(85)
        detection_data[0:4] = [0.2, 0.3, 0.1, 0.2]  # x, y, w, h
        detection_data[4] = 0.9  # confidence
        detection_data[5] = 1.0  # class 0 probability (person)
        
        mock_output[0, 0, :] = detection_data
        self.mock_session.run.return_value = [mock_output]
    
    @patch('onnxruntime.InferenceSession')
    def test_onnx_detector_initialization(self, mock_inference_session):
        """Test ONNX detector initialization"""
        # Setup mock
        mock_inference_session.return_value = self.mock_session
        
        # Initialize detector
        detector = ONNXDetector(
            model_path="models/onnx/yolov8n.onnx",
            input_name="input",
            confidence_threshold=0.5
        )
        
        # Assertions
        mock_inference_session.assert_called_once_with("models/onnx/yolov8n.onnx")
        self.assertEqual(detector.input_name, "input")
        self.assertEqual(detector.confidence_threshold, 0.5)
        self.assertEqual(detector.session, self.mock_session)
        self.assertEqual(detector.input_shape, [1, 3, 640, 640])
    
    @patch('onnxruntime.InferenceSession')
    def test_detect_with_onnx(self, mock_inference_session):
        """Test detection using ONNX model"""
        # Setup mock
        mock_inference_session.return_value = self.mock_session
        
        # Initialize detector
        detector = ONNXDetector(
            model_path="models/onnx/yolov8n.onnx",
            input_name="input",
            class_mapping={0: "person"}
        )
        
        # Run detection
        detections = detector.detect(self.test_image)
        
        # Assertions
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "person")
        self.assertAlmostEqual(detections[0].confidence, 0.9)
        
        # Verify image preprocessing and inference
        self.mock_session.run.assert_called_once()
        # Get the call arguments
        args, kwargs = self.mock_session.run.call_args
        # Check input shape (should be preprocessed)
        input_data = kwargs[args[1]]
        self.assertEqual(input_data["input"].shape, (1, 3, 640, 640))
    
    @patch('onnxruntime.InferenceSession')
    def test_confidence_filtering(self, mock_inference_session):
        """Test confidence threshold filtering"""
        # Setup mock with multiple detections
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input", shape=[1, 3, 640, 640])]
        
        # Create mock output with multiple detections of varying confidence
        mock_output = np.zeros((1, 8400, 85), dtype=np.float32)
        
        # Add detections with different confidences
        high_conf_data = np.zeros(85)
        high_conf_data[0:4] = [0.2, 0.3, 0.1, 0.2]
        high_conf_data[4] = 0.9
        high_conf_data[5] = 1.0  # class 0
        
        med_conf_data = np.zeros(85)
        med_conf_data[0:4] = [0.5, 0.6, 0.1, 0.2]
        med_conf_data[4] = 0.6
        med_conf_data[5] = 1.0  # class 0
        
        low_conf_data = np.zeros(85)
        low_conf_data[0:4] = [0.7, 0.8, 0.1, 0.2]
        low_conf_data[4] = 0.3
        low_conf_data[5] = 1.0  # class 0
        
        mock_output[0, 0, :] = high_conf_data
        mock_output[0, 1, :] = med_conf_data
        mock_output[0, 2, :] = low_conf_data
        
        mock_session.run.return_value = [mock_output]
        mock_inference_session.return_value = mock_session
        
        # Initialize detector with high threshold
        detector = ONNXDetector(
            model_path="models/onnx/yolov8n.onnx",
            input_name="input",
            confidence_threshold=0.7,  # Only highest confidence should pass
            class_mapping={0: "person"}
        )
        
        # Run detection
        detections = detector.detect(self.test_image)
        
        # Should only return the high confidence detection
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].confidence, 0.9)
    
    @patch('onnxruntime.InferenceSession')
    def test_class_filtering(self, mock_inference_session):
        """Test class filtering"""
        # Setup mock with multiple classes
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input", shape=[1, 3, 640, 640])]
        
        # Create mock output with multiple classes
        mock_output = np.zeros((1, 8400, 85), dtype=np.float32)
        
        # Add detections of different classes
        person_data = np.zeros(85)
        person_data[0:4] = [0.2, 0.3, 0.1, 0.2]
        person_data[4] = 0.9
        person_data[5] = 1.0  # class 0 (person)
        
        bicycle_data = np.zeros(85)
        bicycle_data[0:4] = [0.5, 0.6, 0.1, 0.2]
        bicycle_data[4] = 0.8
        bicycle_data[6] = 1.0  # class 1 (bicycle)
        
        mock_output[0, 0, :] = person_data
        mock_output[0, 1, :] = bicycle_data
        
        mock_session.run.return_value = [mock_output]
        mock_inference_session.return_value = mock_session
        
        # Initialize detector with only person class mapping
        detector = ONNXDetector(
            model_path="models/onnx/yolov8n.onnx",
            input_name="input",
            class_mapping={0: "person"}  # Only map person class
        )
        
        # Run detection
        detections = detector.detect(self.test_image)
        
        # Should only return person detections
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "person")
    
    @patch('onnx.save_model')
    @patch('ultralytics.YOLO')
    def test_convert_to_onnx(self, mock_yolo, mock_save_model):
        """Test conversion of model to ONNX format"""
        # Setup mocks
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        
        # Mock export method
        def mock_export(format, **kwargs):
            return "mock_export_path.onnx"
        
        mock_model.export = mock_export
        
        # Call conversion function
        onnx_path = convert_to_onnx(
            model_path="models/yolo/yolov8n.pt",
            output_path="models/onnx/yolov8n.onnx"
        )
        
        # Assertions
        mock_yolo.assert_called_once_with("models/yolo/yolov8n.pt")
        self.assertEqual(onnx_path, "models/onnx/yolov8n.onnx")


if __name__ == '__main__':
    unittest.main() 