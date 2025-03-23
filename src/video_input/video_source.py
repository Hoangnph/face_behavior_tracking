"""Video source classes for different types of video inputs."""

from abc import ABC, abstractmethod
import cv2
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

class VideoSource(ABC):
    """Abstract base class for all video sources."""
    
    def __init__(self, source_id: Any):
        """Initialize the video source.
        
        Args:
            source_id: Identifier for the video source (device index, file path, URL)
        """
        self.source_id = source_id
        self.cap = None
        self.is_opened = False
        self.properties = {}
    
    @abstractmethod
    def open(self) -> bool:
        """Open the video source.
        
        Returns:
            bool: True if opened successfully, False otherwise
        """
        pass
    
    def close(self) -> None:
        """Release the video source resources."""
        if self.cap is not None and self.is_opened:
            self.cap.release()
            self.is_opened = False
            logger.info(f"Closed video source: {self.source_id}")
    
    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """Read a frame from the video source.
        
        Returns:
            Tuple[bool, Optional[cv2.Mat]]: (success, frame)
        """
        if not self.is_opened:
            logger.warning("Attempting to read from closed video source")
            return False, None
            
        ret, frame = self.cap.read()
        return ret, frame
    
    def get_properties(self) -> Dict[str, Any]:
        """Get properties of the video source.
        
        Returns:
            Dict[str, Any]: Dictionary of video properties
        """
        if not self.is_opened:
            return self.properties
            
        # Update properties
        self.properties = {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        return self.properties
    
    def set_property(self, prop_id: int, value: Any) -> bool:
        """Set a property of the video source.
        
        Args:
            prop_id: OpenCV property identifier
            value: Value to set
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_opened:
            logger.warning("Attempting to set property on closed video source")
            return False
            
        return self.cap.set(prop_id, value)
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class WebcamSource(VideoSource):
    """Video source for webcam/camera devices."""
    
    def __init__(self, device_index: int = 0, resolution: Optional[Tuple[int, int]] = None):
        """Initialize webcam source.
        
        Args:
            device_index: Camera device index
            resolution: Optional (width, height) tuple for desired resolution
        """
        super().__init__(device_index)
        self.resolution = resolution
    
    def open(self) -> bool:
        """Open the webcam.
        
        Returns:
            bool: True if opened successfully, False otherwise
        """
        self.cap = cv2.VideoCapture(self.source_id)
        self.is_opened = self.cap.isOpened()
        
        if not self.is_opened:
            logger.error(f"Failed to open webcam with device index: {self.source_id}")
            return False
        
        # Set resolution if specified
        if self.resolution:
            width, height = self.resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Get and log camera properties
        props = self.get_properties()
        logger.info(f"Opened webcam (index: {self.source_id}) with properties: {props}")
        return True


class FileSource(VideoSource):
    """Video source for video files."""
    
    def open(self) -> bool:
        """Open the video file.
        
        Returns:
            bool: True if opened successfully, False otherwise
        """
        self.cap = cv2.VideoCapture(self.source_id)
        self.is_opened = self.cap.isOpened()
        
        if not self.is_opened:
            logger.error(f"Failed to open video file: {self.source_id}")
            return False
        
        # Get and log file properties
        props = self.get_properties()
        logger.info(f"Opened video file: {self.source_id} with properties: {props}")
        return True


class RTSPSource(VideoSource):
    """Video source for RTSP streams."""
    
    def __init__(self, url: str, connection_timeout: int = 10):
        """Initialize RTSP source.
        
        Args:
            url: RTSP stream URL
            connection_timeout: Connection timeout in seconds
        """
        super().__init__(url)
        self.connection_timeout = connection_timeout
    
    def open(self) -> bool:
        """Open the RTSP stream.
        
        Returns:
            bool: True if opened successfully, False otherwise
        """
        # Configure OpenCV to use FFMPEG with a timeout for RTSP
        self.cap = cv2.VideoCapture(self.source_id, cv2.CAP_FFMPEG)
        
        # Set timeout
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.connection_timeout * 1000)
        
        # Try to open the stream
        self.is_opened = self.cap.isOpened()
        
        if not self.is_opened:
            logger.error(f"Failed to open RTSP stream: {self.source_id}")
            return False
        
        # Get and log stream properties
        props = self.get_properties()
        logger.info(f"Opened RTSP stream: {self.source_id} with properties: {props}")
        return True 