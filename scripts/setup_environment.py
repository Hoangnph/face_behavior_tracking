#!/usr/bin/env python
"""
Environment setup and validation script for Human Tracking project.
This script verifies that all required dependencies are installed and working.
"""
import sys
import os
import subprocess
import importlib
from typing import List, Dict, Tuple, Optional, Union, Any


def check_python_version() -> bool:
    """Check that Python version meets requirements (3.9+)."""
    required_version = (3, 9)
    current_version = sys.version_info
    
    if current_version < required_version:
        print(f"ERROR: Python {required_version[0]}.{required_version[1]}+ required, "
              f"but {current_version[0]}.{current_version[1]} detected")
        return False
    
    print(f"✓ Python version: {sys.version}")
    return True


def check_package_available(package_name: str) -> bool:
    """Check if a package is available for import."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def check_core_dependencies() -> Dict[str, bool]:
    """Check that all required dependencies are installed."""
    dependencies = {
        "OpenCV": "cv2",
        "NumPy": "numpy",
        "MediaPipe": "mediapipe",
        "ONNX Runtime": "onnxruntime",
        "PyTorch": "torch",
        "Ultralytics (YOLOv8)": "ultralytics",
    }
    
    results = {}
    
    for name, package in dependencies.items():
        available = check_package_available(package)
        if available:
            # If available, try to get version
            try:
                module = importlib.import_module(package)
                version = getattr(module, "__version__", "unknown version")
                print(f"✓ {name}: {version}")
            except Exception:
                print(f"✓ {name}: installed (version unavailable)")
        else:
            print(f"✗ {name}: not installed")
        
        results[name] = available
    
    return results


def check_gpu_availability() -> Tuple[bool, str]:
    """Check if GPU is available for inference."""
    # Check GPU via PyTorch
    if check_package_available("torch"):
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            print(f"✓ GPU available: {device_name} (Total devices: {device_count})")
            return True, device_name
    
    # Check ONNX Runtime providers
    if check_package_available("onnxruntime"):
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        gpu_providers = [p for p in providers if 'GPU' in p or 'CUDA' in p]
        if gpu_providers:
            print(f"✓ ONNX GPU providers: {', '.join(gpu_providers)}")
            return True, str(gpu_providers)
    
    print("✗ No GPU acceleration available")
    return False, "None"


def check_mediapipe_models() -> bool:
    """Try initializing MediaPipe models to verify they work."""
    if not check_package_available("mediapipe"):
        return False
    
    try:
        import mediapipe as mp
        
        # Check face detection
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        
        # Check pose estimation
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        print("✓ MediaPipe models initialized successfully")
        return True
    except Exception as e:
        print(f"✗ MediaPipe model initialization failed: {e}")
        return False


def check_project_structure() -> bool:
    """Verify project directory structure."""
    required_dirs = ["src", "tests", "data", "logs", "scripts", "docs"]
    missing_dirs = []
    
    # Get the project root directory (parent of this script's directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Check each required directory
    for directory in required_dirs:
        dir_path = os.path.join(project_root, directory)
        if not os.path.isdir(dir_path):
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"✗ Missing directories: {', '.join(missing_dirs)}")
        return False
    
    print("✓ Project structure verified")
    return True


def check_bytetrack() -> bool:
    """Check if ByteTrack is installed (optional component)."""
    # Check if already installed
    if check_package_available("bytetrack"):
        print("✓ ByteTrack is installed")
        return True
    
    try:
        # Try to import through alternative path
        import sys
        sys.path.append("external/ByteTrack")
        import yolox
        print("✓ ByteTrack found via external path")
        return True
    except ImportError:
        print("⚠ ByteTrack is not installed (optional component)")
        return False


def install_bytetrack() -> bool:
    """Install ByteTrack from GitHub (best effort)."""
    try:
        # Check if already installed
        if check_package_available("bytetrack"):
            print("✓ ByteTrack already installed")
            return True
        
        print("Installing ByteTrack from GitHub...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "git+https://github.com/ifzhang/ByteTrack.git"], 
            check=True,
            capture_output=True,
            text=True
        )
        print("✓ ByteTrack installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠ ByteTrack installation failed: {e.stdout} {e.stderr}")
        print("⚠ ByteTrack is optional and can be installed later if needed")
        return True  # Return true to not fail the entire setup


def main() -> int:
    """Run all environment checks and setup."""
    print("=" * 50)
    print("Human Tracking - Environment Setup and Validation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check core dependencies
    dependency_results = check_core_dependencies()
    all_dependencies_installed = all(dependency_results.values())
    
    # Check GPU
    has_gpu, _ = check_gpu_availability()
    
    # Check MediaPipe
    mediapipe_ok = check_mediapipe_models()
    
    # Check project structure
    structure_ok = check_project_structure()
    
    # Check ByteTrack (optional)
    bytetrack_ok = check_bytetrack()
    if not bytetrack_ok:
        bytetrack_ok = install_bytetrack()
    
    # Summary
    print("\n" + "=" * 50)
    print("Environment Setup Summary")
    print("=" * 50)
    
    if all_dependencies_installed and structure_ok and mediapipe_ok:
        print("✓ Core environment is ready for development.")
        if not bytetrack_ok:
            print("⚠ ByteTrack (optional) could not be installed automatically. You may need to install it manually.")
        return 0
    else:
        print("⚠ Some checks failed. Please resolve the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 