"""Tests for video source classes."""

import os
import cv2
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.video_input.video_source import VideoSource, WebcamSource, FileSource, RTSPSource

# Sample video file path (adjust as needed)
SAMPLE_VIDEO = os.path.join("data", "samples", "test_video.mp4")

def create_mock_capture(width=640, height=480, fps=30, frame_count=100, is_opened=True):
    """Create a mock OpenCV capture object for testing."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = is_opened
    
    # Set up property method
    def mock_get(prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return width
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return height
        elif prop_id == cv2.CAP_PROP_FPS:
            return fps
        elif prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return frame_count
        return 0
    
    mock_cap.get.side_effect = mock_get
    
    # Set up read method to return a test frame
    test_frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Add some content to the test frame
    cv2.rectangle(test_frame, (50, 50), (width-50, height-50), (0, 255, 0), 2)
    mock_cap.read.return_value = (True, test_frame)
    
    return mock_cap


class TestWebcamSource:
    """Tests for WebcamSource class."""
    
    @patch('cv2.VideoCapture')
    def test_init_default(self, mock_video_capture):
        """Test initialization with default parameters."""
        mock_video_capture.return_value = create_mock_capture()
        
        # Create source
        source = WebcamSource()
        
        # Check initialization
        assert source.source_id == 0
        assert source.resolution is None
        assert source.is_opened is False
    
    @patch('cv2.VideoCapture')
    def test_open_success(self, mock_video_capture):
        """Test successful opening of webcam."""
        mock_video_capture.return_value = create_mock_capture()
        
        # Create and open source
        source = WebcamSource(device_index=0)
        success = source.open()
        
        # Verify
        assert success is True
        assert source.is_opened is True
        mock_video_capture.assert_called_once_with(0)
    
    @patch('cv2.VideoCapture')
    def test_open_failure(self, mock_video_capture):
        """Test failed opening of webcam."""
        mock_video_capture.return_value = create_mock_capture(is_opened=False)
        
        # Create and open source
        source = WebcamSource(device_index=1)
        success = source.open()
        
        # Verify
        assert success is False
        assert source.is_opened is False
    
    @patch('cv2.VideoCapture')
    def test_read_frame(self, mock_video_capture):
        """Test reading a frame from the webcam."""
        mock_cap = create_mock_capture()
        mock_video_capture.return_value = mock_cap
        
        # Create, open, and read from source
        source = WebcamSource()
        source.open()
        ret, frame = source.read()
        
        # Verify
        assert ret is True
        assert frame.shape == (480, 640, 3)
        assert isinstance(frame, np.ndarray)
    
    @patch('cv2.VideoCapture')
    def test_get_properties(self, mock_video_capture):
        """Test getting properties from the webcam."""
        mock_video_capture.return_value = create_mock_capture(
            width=1280, height=720, fps=60, frame_count=0
        )
        
        # Create, open, and get properties
        source = WebcamSource()
        source.open()
        props = source.get_properties()
        
        # Verify
        assert props['width'] == 1280
        assert props['height'] == 720
        assert props['fps'] == 60
    
    @patch('cv2.VideoCapture')
    def test_close(self, mock_video_capture):
        """Test closing the webcam."""
        mock_cap = create_mock_capture()
        mock_video_capture.return_value = mock_cap
        
        # Create, open, and close source
        source = WebcamSource()
        source.open()
        source.close()
        
        # Verify
        assert source.is_opened is False
        mock_cap.release.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_context_manager(self, mock_video_capture):
        """Test using webcam source as a context manager."""
        mock_cap = create_mock_capture()
        mock_video_capture.return_value = mock_cap
        
        # Use as context manager
        with WebcamSource() as source:
            assert source.is_opened is True
            ret, frame = source.read()
            assert ret is True
        
        # Verify closed after context
        assert source.is_opened is False
        mock_cap.release.assert_called_once()


class TestFileSource:
    """Tests for FileSource class."""
    
    @patch('cv2.VideoCapture')
    def test_open_success(self, mock_video_capture):
        """Test successful opening of video file."""
        mock_video_capture.return_value = create_mock_capture()
        
        # Create and open source
        source = FileSource(SAMPLE_VIDEO)
        success = source.open()
        
        # Verify
        assert success is True
        assert source.is_opened is True
        mock_video_capture.assert_called_once_with(SAMPLE_VIDEO)
    
    @patch('cv2.VideoCapture')
    def test_open_nonexistent_file(self, mock_video_capture):
        """Test opening a nonexistent video file."""
        mock_video_capture.return_value = create_mock_capture(is_opened=False)
        
        # Create and open source
        source = FileSource("nonexistent.mp4")
        success = source.open()
        
        # Verify
        assert success is False
        assert source.is_opened is False


class TestRTSPSource:
    """Tests for RTSPSource class."""
    
    @patch('cv2.VideoCapture')
    def test_init(self, mock_video_capture):
        """Test initialization."""
        # Create source
        url = "rtsp://example.com/stream"
        source = RTSPSource(url, connection_timeout=5)
        
        # Check initialization
        assert source.source_id == url
        assert source.connection_timeout == 5
        assert source.is_opened is False
    
    @patch('cv2.VideoCapture')
    def test_open_rtsp(self, mock_video_capture):
        """Test opening RTSP stream."""
        mock_cap = create_mock_capture()
        mock_video_capture.return_value = mock_cap
        
        # Create and open source
        url = "rtsp://example.com/stream"
        source = RTSPSource(url)
        success = source.open()
        
        # Verify
        assert success is True
        assert source.is_opened is True
        mock_video_capture.assert_called_once_with(url, cv2.CAP_FFMPEG)
        
        # Verify timeout was set
        timeout_ms = 10 * 1000  # Default 10 seconds
        mock_cap.set.assert_called_once_with(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms) 