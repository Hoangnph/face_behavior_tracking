"""Frame processor for preprocessing video frames."""

import cv2
import numpy as np
import time
import logging
from typing import Dict, Tuple, Optional, List, Callable, Any

logger = logging.getLogger(__name__)

class FrameProcessor:
    """Process frames from video sources with configurable operations."""
    
    def __init__(self, 
                 resize_dims: Optional[Tuple[int, int]] = None,
                 convert_rgb: bool = False,
                 normalize: bool = False,
                 custom_processors: Optional[List[Callable]] = None):
        """Initialize frame processor with desired operations.
        
        Args:
            resize_dims: Optional (width, height) tuple for resizing
            convert_rgb: Whether to convert from BGR to RGB colorspace
            normalize: Whether to normalize pixel values (0-1 range)
            custom_processors: List of custom processing functions
        """
        self.resize_dims = resize_dims
        self.convert_rgb = convert_rgb
        self.normalize = normalize
        self.custom_processors = custom_processors or []
        self.stats = {
            'total_frames': 0,
            'processing_time': 0.0,
            'avg_processing_time': 0.0,
            'frame_times': [],
            'max_processing_time': 0.0
        }
        
        logger.info(f"Initialized FrameProcessor with resize={resize_dims}, "
                   f"convert_rgb={convert_rgb}, normalize={normalize}, "
                   f"custom_processors={len(self.custom_processors)}")
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a single frame with the configured operations.
        
        Args:
            frame: Input frame (as numpy array)
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (processed_frame, metadata)
        """
        if frame is None or frame.size == 0:
            logger.warning("Received empty frame for processing")
            return frame, {"error": "Empty frame"}
        
        metadata = {
            "original_shape": frame.shape,
            "operations": [],
            "processing_time": 0,
        }
        
        start_time = time.time()
        
        # Create a copy to avoid modifying the original
        result = frame.copy()
        
        # Apply resize if configured
        if self.resize_dims:
            result = cv2.resize(result, self.resize_dims)
            metadata["operations"].append("resize")
            metadata["resize_dims"] = self.resize_dims
        
        # Convert color space if required
        if self.convert_rgb:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            metadata["operations"].append("bgr2rgb")
        
        # Normalize if required
        if self.normalize:
            result = result.astype(np.float32) / 255.0
            metadata["operations"].append("normalize")
        
        # Apply custom processors
        for i, processor in enumerate(self.custom_processors):
            try:
                result = processor(result)
                metadata["operations"].append(f"custom_{i}")
            except Exception as e:
                logger.error(f"Error in custom processor {i}: {str(e)}")
                metadata["error_processor_{i}"] = str(e)
        
        # Calculate processing time
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # ms
        
        # Update stats
        self.stats['total_frames'] += 1
        self.stats['processing_time'] += processing_time
        self.stats['frame_times'].append(processing_time)
        self.stats['max_processing_time'] = max(self.stats['max_processing_time'], processing_time)
        self.stats['avg_processing_time'] = self.stats['processing_time'] / self.stats['total_frames']
        
        # Keep only the last 100 frame times for rolling average
        if len(self.stats['frame_times']) > 100:
            self.stats['frame_times'].pop(0)
        
        # Add processing time to metadata
        metadata["processing_time"] = processing_time
        metadata["final_shape"] = result.shape
        
        return result, metadata
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics.
        
        Returns:
            Dict[str, Any]: Dictionary of processing statistics
        """
        stats = self.stats.copy()
        
        # Calculate rolling average from the last 100 frames
        if self.stats['frame_times']:
            stats['rolling_avg_time'] = sum(self.stats['frame_times']) / len(self.stats['frame_times'])
        
        # Remove the full list of frame times to keep the output manageable
        stats.pop('frame_times', None)
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            'total_frames': 0,
            'processing_time': 0.0,
            'avg_processing_time': 0.0,
            'frame_times': [],
            'max_processing_time': 0.0
        }
        logger.info("Frame processor statistics reset") 