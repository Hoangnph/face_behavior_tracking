"""Video input pipeline for managing video sources and frame processing."""

import cv2
import time
import logging
import threading
from typing import Dict, Tuple, Optional, Any, List, Union, Callable
import queue
import numpy as np

from src.video_input.video_source import VideoSource
from src.video_input.frame_processor import FrameProcessor

logger = logging.getLogger(__name__)

class VideoPipeline:
    """High-level pipeline for managing video input and processing."""
    
    def __init__(self, 
                 video_source: VideoSource,
                 frame_processor: Optional[FrameProcessor] = None,
                 buffer_size: int = 30,
                 target_fps: Optional[float] = None,
                 enable_threading: bool = True):
        """Initialize the video pipeline.
        
        Args:
            video_source: Video source instance
            frame_processor: Optional frame processor for preprocessing
            buffer_size: Maximum number of frames to buffer
            target_fps: Target framerate (None for source native)
            enable_threading: Whether to use threading for the pipeline
        """
        self.video_source = video_source
        self.frame_processor = frame_processor or FrameProcessor()
        self.buffer_size = buffer_size
        self.target_fps = target_fps
        self.enable_threading = enable_threading
        
        # Frame buffer
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        
        # Threading control
        self.is_running = False
        self.thread = None
        
        # Stats
        self.stats = {
            'source_fps': 0,
            'actual_fps': 0,
            'dropped_frames': 0,
            'processed_frames': 0,
            'buffer_utilization': 0,
            'frame_times': [],
            'start_time': 0,
            'last_frame_time': 0
        }
        
        logger.info(f"Initialized VideoPipeline with buffer_size={buffer_size}, "
                   f"target_fps={target_fps}, threading={enable_threading}")
    
    def start(self) -> bool:
        """Start the video pipeline.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.is_running:
            logger.warning("Pipeline already running")
            return True
        
        # Open the video source
        if not self.video_source.open():
            logger.error("Failed to open video source")
            return False
        
        # Get source properties
        props = self.video_source.get_properties()
        self.stats['source_fps'] = props.get('fps', 30.0)
        
        # Set target FPS if not specified
        if self.target_fps is None:
            self.target_fps = self.stats['source_fps']
        
        # Calculate frame interval
        self.frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0
        
        # Start the pipeline
        self.is_running = True
        self.stats['start_time'] = time.time()
        
        # Start thread if enabled
        if self.enable_threading:
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            logger.info("Started video pipeline in threaded mode")
        else:
            logger.info("Started video pipeline in synchronous mode")
        
        return True
    
    def stop(self) -> None:
        """Stop the video pipeline."""
        self.is_running = False
        
        # Wait for thread to finish if it exists
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning("Pipeline thread did not terminate gracefully")
        
        # Close the video source
        self.video_source.close()
        
        # Clear the buffer
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Stopped video pipeline")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """Read a processed frame from the pipeline.
        
        Returns:
            Tuple[bool, Optional[np.ndarray], Dict[str, Any]]: (success, frame, metadata)
        """
        # If not running in threaded mode, capture and process frames directly
        if not self.enable_threading:
            return self._capture_and_process()
        
        # Otherwise, get a frame from the buffer
        try:
            success, frame, metadata = self.frame_buffer.get(timeout=1.0)
            self.stats['last_frame_time'] = time.time()
            
            # Update stats
            self.stats['processed_frames'] += 1
            self.stats['buffer_utilization'] = self.frame_buffer.qsize() / self.buffer_size
            
            return success, frame, metadata
        except queue.Empty:
            logger.warning("Frame buffer empty, pipeline may be too slow")
            return False, None, {"error": "Buffer empty"}
    
    def _capture_and_process(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """Capture and process a single frame from the video source.
        
        Returns:
            Tuple[bool, Optional[np.ndarray], Dict[str, Any]]: (success, frame, metadata)
        """
        # Capture frame
        success, frame = self.video_source.read()
        
        if not success:
            return False, None, {"error": "Failed to read frame"}
        
        # Process frame
        processed_frame, metadata = self.frame_processor.process(frame)
        
        # Update stats
        current_time = time.time()
        elapsed = current_time - self.stats['start_time']
        if elapsed > 0:
            self.stats['actual_fps'] = self.stats['processed_frames'] / elapsed
        
        # Record frame time
        if self.stats['last_frame_time'] > 0:
            self.stats['frame_times'].append(current_time - self.stats['last_frame_time'])
            if len(self.stats['frame_times']) > 100:
                self.stats['frame_times'].pop(0)
        
        self.stats['last_frame_time'] = current_time
        self.stats['processed_frames'] += 1
        
        return True, processed_frame, metadata
    
    def _capture_loop(self) -> None:
        """Continuous frame capture loop for threaded operation."""
        next_frame_time = time.time()
        
        while self.is_running:
            # Respect target FPS
            current_time = time.time()
            
            # If we're ahead of schedule, sleep
            if current_time < next_frame_time:
                time.sleep(next_frame_time - current_time)
            
            # Schedule next frame
            next_frame_time = time.time() + self.frame_interval
            
            # Capture and process frame
            success, frame, metadata = self._capture_and_process()
            
            # Skip if capture failed
            if not success:
                continue
            
            # If buffer is full, drop oldest frame
            if self.frame_buffer.full():
                try:
                    self.frame_buffer.get_nowait()
                    self.stats['dropped_frames'] += 1
                except queue.Empty:
                    pass
            
            # Add to buffer
            try:
                self.frame_buffer.put_nowait((success, frame, metadata))
            except queue.Full:
                self.stats['dropped_frames'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics.
        
        Returns:
            Dict[str, Any]: Dictionary of pipeline statistics
        """
        stats = self.stats.copy()
        
        # Add processor stats
        processor_stats = self.frame_processor.get_stats()
        stats['processor'] = processor_stats
        
        # Calculate average frame time
        if self.stats['frame_times']:
            stats['avg_frame_time'] = sum(self.stats['frame_times']) / len(self.stats['frame_times'])
            stats['current_fps'] = 1.0 / stats['avg_frame_time'] if stats['avg_frame_time'] > 0 else 0
        
        # Remove large arrays
        stats.pop('frame_times', None)
        
        # Add current buffer utilization
        stats['buffer_utilization'] = self.frame_buffer.qsize() / self.buffer_size if self.enable_threading else 0
        
        return stats
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop() 