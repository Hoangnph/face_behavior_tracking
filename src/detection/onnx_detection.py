"""
ONNX runtime integration for optimized inference.
"""
from typing import List, Dict, Any, Optional, Tuple, Union
import os
import numpy as np
import cv2
import onnxruntime as ort
from ultralytics import YOLO
from .base_detection import BaseDetector, Detection, BoundingBox


def convert_to_onnx(model_path: str, 
                    output_path: Optional[str] = None, 
                    opset: int = 12, 
                    simplify: bool = True,
                    dynamic: bool = True) -> str:
    """
    Convert a YOLOv8 model to ONNX format.
    
    Args:
        model_path: Path to the YOLOv8 model file.
        output_path: Path to save the ONNX model. If None, will use the same path with .onnx extension.
        opset: ONNX opset version.
        simplify: Whether to simplify the ONNX model.
        dynamic: Whether to use dynamic input shapes.
        
    Returns:
        Path to the converted ONNX model.
    """
    # Load YOLOv8 model
    model = YOLO(model_path)
    
    # Set output path if not provided
    if output_path is None:
        output_path = os.path.splitext(model_path)[0] + '.onnx'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Export model to ONNX format
    model.export(format='onnx', opset=opset, simplify=simplify, dynamic=dynamic)
    
    # Return the path to the converted model
    return output_path


class ONNXDetector(BaseDetector):
    """
    Object detector using ONNX runtime for optimized inference.
    """
    def __init__(self, 
                 model_path: str,
                 input_name: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 class_mapping: Optional[Dict[int, str]] = None,
                 input_size: Optional[Tuple[int, int]] = None):
        """
        Initialize ONNX detector.
        
        Args:
            model_path: Path to ONNX model file.
            input_name: Name of the input tensor. If None, will use the first input.
            confidence_threshold: Minimum confidence threshold for detections.
            class_mapping: Mapping from class IDs to class names.
            input_size: Input size (width, height) for the model. If None, will use the model's input shape.
        """
        super().__init__(confidence_threshold=confidence_threshold)
        
        # Load ONNX model
        self.session = ort.InferenceSession(model_path)
        
        # Get input details
        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()
        
        # Set input name
        if input_name is None:
            self.input_name = self.inputs[0].name
        else:
            self.input_name = input_name
            
        # Get input shape
        self.input_shape = self.inputs[0].shape
        
        # Set input size
        try:
            if input_size is None and len(self.input_shape) >= 4:
                # Use model's input shape if available
                # Sometimes these are symbolic dimensions, so handle potential errors
                try:
                    self.input_height = int(self.input_shape[2])
                    self.input_width = int(self.input_shape[3])
                except (ValueError, TypeError):
                    # Fall back to default if conversion fails
                    print(f"Could not parse input dimensions from {self.input_shape}, using defaults")
                    self.input_height = 640
                    self.input_width = 640
            elif input_size is not None:
                # Use provided input size
                self.input_width = int(input_size[0])
                self.input_height = int(input_size[1])
            else:
                # Default to 640x640 if shape is not available
                self.input_width, self.input_height = 640, 640
        except Exception as e:
            print(f"Error setting input dimensions: {str(e)}, using defaults")
            self.input_width, self.input_height = 640, 640
        
        # Set class mapping
        self.class_mapping = class_mapping or {0: "person"}  # Default to person detection
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detect objects in the input image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV).
            
        Returns:
            List of Detection objects.
        """
        # Preprocess image
        input_tensor = self._preprocess_image(image)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Process outputs
        return self._process_outputs(outputs, image)
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess input image for ONNX model.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV).
            
        Returns:
            Preprocessed image tensor ready for inference.
        """
        # Get image dimensions
        if image.ndim == 2:  # Grayscale
            h, w = image.shape
            # Convert to 3 channels
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            h, w = image.shape[:2]
        
        # Ensure width and height are integers
        target_width = int(self.input_width)
        target_height = int(self.input_height)
        
        # Resize image with explicit width and height (not tuple)
        resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values (0-1)
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to channel-first format (NCHW)
        transposed = normalized.transpose(2, 0, 1)
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def _process_outputs(self, outputs: List[np.ndarray], original_image: np.ndarray) -> List[Detection]:
        """
        Process ONNX model outputs into Detection objects.
        
        Args:
            outputs: ONNX model output.
            original_image: Original input image.
            
        Returns:
            List of Detection objects.
        """
        # Get original image size
        orig_height, orig_width = original_image.shape[:2]
        
        # Process outputs based on model type
        detections = []
        
        # Check if output is in YOLOv8 format
        if len(outputs) == 1 and outputs[0].shape[2] > 5:  # YOLOv8 format
            # YOLOv8 format: [batch_idx, x, y, w, h, conf, cls1, cls2, ...]
            output = outputs[0]
            
            # Get number of detections and classes
            num_detections = output.shape[1]
            
            for i in range(num_detections):
                # Get detection data
                detection_data = output[0, i, :]
                
                # Skip if no data (all zeros)
                if np.all(detection_data == 0):
                    continue
                
                # Get confidence and class ID
                confidence = float(detection_data[4])
                if confidence < self.confidence_threshold:
                    continue
                
                # Get class with highest probability
                class_id = int(np.argmax(detection_data[5:]))
                
                # Skip if class is not in mapping
                if class_id not in self.class_mapping:
                    continue
                
                # Get bounding box (x, y, w, h format)
                x, y, w, h = detection_data[:4]
                
                # Convert to pixel coordinates
                x1 = int((x - w/2) * orig_width)
                y1 = int((y - h/2) * orig_height)
                width = int(w * orig_width)
                height = int(h * orig_height)
                
                # Create detection
                bbox = BoundingBox(x1=x1, y1=y1, width=width, height=height)
                detection = Detection(
                    bbox=bbox,
                    confidence=confidence,
                    label=self.class_mapping.get(class_id, f"class_{class_id}")
                )
                
                detections.append(detection)
        
        # Return filtered detections
        return self._filter_detections(detections)
    
    def visualize(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Visualize detections on the image.
        
        Args:
            image: Input image as numpy array.
            detections: List of Detection objects.
            
        Returns:
            Image with visualized detections.
        """
        # Create a copy of the image
        vis_image = image.copy()
        
        # Color mapping for classes
        color_map = {
            "person": (0, 0, 255),  # Red
            "face": (0, 255, 0),    # Green
            "default": (255, 0, 0)  # Blue
        }
        
        for detection in detections:
            # Draw bounding box
            x1, y1, x2, y2 = detection.bbox.to_xyxy()
            color = color_map.get(detection.label, color_map["default"])
            
            # Draw rectangle with label and confidence
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            text = f"{detection.label}: {detection.confidence:.2f}"
            cv2.putText(vis_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_image

    def _resize_with_aspect_ratio(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image as numpy array.
            target_size: Target size (width, height).
            
        Returns:
            Resized image with padding if necessary.
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
            
            # Resize using direct resize
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


if __name__ == "__main__":
    # Test code for debugging ONNX detector
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use a test image if none provided
        image_path = "data/faces/known_customers/customer1.jpg"
    
    # Create an ONNX detector if model exists
    onnx_model_path = "models/onnx/yolov8n.onnx"
    yolo_model_path = "models/yolo/yolov8n.pt"
    
    if not os.path.exists(onnx_model_path):
        print(f"ONNX model not found at {onnx_model_path}")
        print("Converting YOLOv8 model to ONNX format...")
        os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)
        convert_to_onnx(yolo_model_path, onnx_model_path)
        print(f"ONNX model saved to {onnx_model_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image from {image_path}")
        sys.exit(1)
    
    # Create detector
    detector = ONNXDetector(
        model_path=onnx_model_path,
        confidence_threshold=0.2,
        class_mapping={0: "person"}
    )
    
    # Test preprocessing
    print("Testing image preprocessing...")
    try:
        input_tensor = detector._preprocess_image(image)
        print(f"Preprocessed image shape: {input_tensor.shape}")
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
    
    # Test detection
    print("Testing detection...")
    try:
        detections = detector.detect(image)
        print(f"Detected {len(detections)} objects")
        for i, detection in enumerate(detections):
            print(f"  {i+1}. {detection.label}: {detection.confidence:.2f}")
    except Exception as e:
        print(f"Error in detection: {str(e)}")
    
    # Test visualization
    print("Testing visualization...")
    try:
        vis_image = detector.visualize(image, detections)
        output_path = "data/output/onnx_test_output.jpg"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, vis_image)
        print(f"Visualization saved to {output_path}")
    except Exception as e:
        print(f"Error in visualization: {str(e)}") 