"""
Detection module for identifying and locating people and faces in images and video.
"""

from .base_detection import BoundingBox, Detection, BaseDetector
from .face_detection import FaceDetection, MediaPipeFaceDetector
from .person_detection import PersonDetection, YOLOPersonDetector
from .onnx_detection import ONNXDetector, convert_to_onnx
from .detection_scheduler import DetectionScheduler

__all__ = [
    'BoundingBox',
    'Detection',
    'BaseDetector',
    'FaceDetection',
    'MediaPipeFaceDetector',
    'PersonDetection',
    'YOLOPersonDetector',
    'ONNXDetector',
    'convert_to_onnx',
    'DetectionScheduler',
]
