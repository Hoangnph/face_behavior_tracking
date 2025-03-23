#!/usr/bin/env python
"""
Test script to validate that the environment is correctly set up.
"""
import sys
import pytest
import platform
import importlib
from typing import Dict


def test_python_version():
    """Test that the Python version is compatible."""
    required_version = (3, 8)
    assert sys.version_info >= required_version, f"Python {required_version[0]}.{required_version[1]}+ is required"


def is_package_available(package_name: str) -> bool:
    """Check if a package can be imported."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


@pytest.mark.parametrize("package_name", [
    "cv2",          # OpenCV
    "numpy",        # NumPy
    "mediapipe",    # MediaPipe
    "onnxruntime",  # ONNX Runtime
    "torch"         # PyTorch
])
def test_core_dependencies(package_name: str):
    """Test that core dependencies are installed."""
    assert is_package_available(package_name), f"Package {package_name} is not installed"


def test_mediapipe_initialization():
    """Test that MediaPipe models can be initialized."""
    if not is_package_available("mediapipe"):
        pytest.skip("MediaPipe not installed")
    
    import mediapipe as mp
    
    # Test face detection initialization
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    assert face_detection is not None, "Failed to initialize MediaPipe Face Detection"
    
    # Test pose estimation initialization
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    assert pose is not None, "Failed to initialize MediaPipe Pose"


def test_yolo_initialization():
    """Test that YOLOv8 model can be initialized."""
    if not is_package_available("ultralytics"):
        pytest.skip("Ultralytics not installed")
    
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        assert model is not None, "Failed to initialize YOLOv8 model"
    except Exception as e:
        pytest.fail(f"Error initializing YOLOv8 model: {e}")


def test_system_info():
    """Print system information for debugging."""
    system_info = {
        "Platform": platform.platform(),
        "Python Version": platform.python_version(),
        "Processor": platform.processor(),
        "Architecture": platform.architecture(),
    }
    
    # Get package versions
    packages = ["cv2", "numpy", "mediapipe", "onnxruntime", "torch", "ultralytics"]
    for package in packages:
        if is_package_available(package):
            try:
                module = importlib.import_module(package)
                version = getattr(module, "__version__", "unknown")
                system_info[f"{package} Version"] = version
            except Exception:
                system_info[f"{package} Version"] = "unavailable"
    
    # Log system info (will be visible in pytest output)
    for key, value in system_info.items():
        print(f"{key}: {value}")
    
    assert True  # Always passes, just for info


def test_gpu_availability():
    """Test if GPU acceleration is available (non-critical test)."""
    has_gpu = False
    gpu_info = "No GPU detected"
    
    # Check GPU via PyTorch
    if is_package_available("torch"):
        import torch
        if torch.cuda.is_available():
            has_gpu = True
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            gpu_info = f"CUDA GPU: {device_name} (Total: {device_count})"
    
    # Check ONNX Runtime
    if not has_gpu and is_package_available("onnxruntime"):
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        gpu_providers = [p for p in providers if 'GPU' in p or 'CUDA' in p]
        if gpu_providers:
            has_gpu = True
            gpu_info = f"ONNX GPU Providers: {', '.join(gpu_providers)}"
    
    # Print result but don't fail tests if no GPU
    print(f"GPU Acceleration: {'Available' if has_gpu else 'Not available'}")
    print(f"GPU Info: {gpu_info}")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 