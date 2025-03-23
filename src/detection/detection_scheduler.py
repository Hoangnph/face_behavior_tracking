"""
Detection scheduler to manage and schedule multiple detection models.
"""
from typing import Dict, List, Optional, Any, Union
import time
import numpy as np
from .base_detection import BaseDetector, Detection


class DetectionScheduler:
    """
    Scheduler for managing multiple detection models.
    Supports running detectors in sequence, at different frequencies, and with different priorities.
    """
    
    def __init__(self, 
                 detectors: Dict[str, BaseDetector],
                 default_detector: Optional[str] = None,
                 frequencies: Optional[Dict[str, int]] = None):
        """
        Initialize detection scheduler.
        
        Args:
            detectors: Dictionary mapping detector names to detector instances.
            default_detector: Name of the default detector to use if not specified.
            frequencies: Dictionary mapping detector names to frequency of execution.
                        A frequency of 1 means run every frame, 2 means every other frame, etc.
        """
        self.detectors = detectors
        self.default_detector = default_detector or list(detectors.keys())[0] if detectors else None
        self.frequencies = frequencies or {name: 1 for name in detectors.keys()}
        
        # Initialize frame counters for each detector
        self.frame_counters = {name: 0 for name in detectors.keys()}
        
        # Initialize execution time tracking
        self.execution_times = {name: [] for name in detectors.keys()}
    
    def detect(self, image: np.ndarray, detector_name: Optional[str] = None) -> List[Detection]:
        """
        Run detection with a specific detector.
        
        Args:
            image: Input image as numpy array.
            detector_name: Name of the detector to use. If None, uses the default detector.
            
        Returns:
            List of Detection objects.
        """
        # Use default detector if none specified
        if detector_name is None:
            detector_name = self.default_detector
        
        # Validate detector name
        if detector_name not in self.detectors:
            raise ValueError(f"Unknown detector: {detector_name}")
        
        # Get detector and run detection
        detector = self.detectors[detector_name]
        
        # Measure execution time
        start_time = time.time()
        detections = detector.detect(image)
        end_time = time.time()
        
        # Update execution time tracking
        execution_time = end_time - start_time
        self.execution_times[detector_name].append(execution_time)
        # Keep only the last 100 times
        if len(self.execution_times[detector_name]) > 100:
            self.execution_times[detector_name].pop(0)
        
        return detections
    
    def detect_all(self, image: np.ndarray) -> List[Detection]:
        """
        Run detection with all detectors.
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            List of Detection objects from all detectors.
        """
        all_detections = []
        
        for name in self.detectors.keys():
            # Run detection with each detector
            detections = self.detect(image, detector_name=name)
            all_detections.extend(detections)
        
        return all_detections
    
    def detect_with_frequency(self, image: np.ndarray, frame_number: Optional[int] = None) -> List[Detection]:
        """
        Run detection with detectors based on their scheduled frequency.
        
        Args:
            image: Input image as numpy array.
            frame_number: Current frame number (optional). If None, will use internal counters.
            
        Returns:
            List of Detection objects from scheduled detectors.
        """
        all_detections = []
        
        # If frame number is not provided, use the internal counter
        if frame_number is None:
            frame_number = max(self.frame_counters.values()) if self.frame_counters else 0
        
        for name, detector in self.detectors.items():
            # Get detector frequency
            frequency = self.frequencies.get(name, 1)
            
            # Check if this detector should run on this frame
            if frame_number % frequency == 0:
                # Run detection
                detections = self.detect(image, detector_name=name)
                all_detections.extend(detections)
            
            # Update frame counter for this detector
            self.frame_counters[name] = frame_number
        
        return all_detections
    
    def get_average_execution_time(self, detector_name: Optional[str] = None) -> Dict[str, float]:
        """
        Get average execution time for detectors.
        
        Args:
            detector_name: Name of the detector to get time for. If None, returns times for all detectors.
            
        Returns:
            Dictionary mapping detector names to average execution time in seconds.
        """
        if detector_name is not None:
            if detector_name not in self.execution_times:
                raise ValueError(f"Unknown detector: {detector_name}")
            
            # Return average time for specific detector
            times = self.execution_times[detector_name]
            avg_time = sum(times) / len(times) if times else 0
            return {detector_name: avg_time}
        else:
            # Return average times for all detectors
            return {
                name: sum(times) / len(times) if times else 0
                for name, times in self.execution_times.items()
            }
    
    def set_frequency(self, detector_name: str, frequency: int) -> None:
        """
        Set the execution frequency for a detector.
        
        Args:
            detector_name: Name of the detector.
            frequency: Frequency of execution (1 = every frame, 2 = every other frame, etc.).
        """
        if detector_name not in self.detectors:
            raise ValueError(f"Unknown detector: {detector_name}")
        
        if frequency < 1:
            raise ValueError("Frequency must be a positive integer")
        
        self.frequencies[detector_name] = frequency
    
    def set_default_detector(self, detector_name: str) -> None:
        """
        Set the default detector.
        
        Args:
            detector_name: Name of the detector to set as default.
        """
        if detector_name not in self.detectors:
            raise ValueError(f"Unknown detector: {detector_name}")
        
        self.default_detector = detector_name
    
    def add_detector(self, name: str, detector: BaseDetector, frequency: int = 1) -> None:
        """
        Add a new detector to the scheduler.
        
        Args:
            name: Name of the detector.
            detector: Detector instance.
            frequency: Frequency of execution (1 = every frame, 2 = every other frame, etc.).
        """
        if name in self.detectors:
            raise ValueError(f"Detector with name '{name}' already exists")
        
        self.detectors[name] = detector
        self.frequencies[name] = frequency
        self.frame_counters[name] = 0
        self.execution_times[name] = []
    
    def remove_detector(self, name: str) -> None:
        """
        Remove a detector from the scheduler.
        
        Args:
            name: Name of the detector to remove.
        """
        if name not in self.detectors:
            raise ValueError(f"Unknown detector: {name}")
        
        # Remove detector
        del self.detectors[name]
        del self.frequencies[name]
        del self.frame_counters[name]
        del self.execution_times[name]
        
        # Update default detector if needed
        if name == self.default_detector:
            self.default_detector = list(self.detectors.keys())[0] if self.detectors else None 