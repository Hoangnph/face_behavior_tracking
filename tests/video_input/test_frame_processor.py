"""Tests for frame processor class."""

import cv2
import numpy as np
import pytest
from unittest.mock import patch

from src.video_input.frame_processor import FrameProcessor

def create_test_frame(width=640, height=480):
    """Create a test frame for processing."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Add a rectangle to make it more realistic
    cv2.rectangle(frame, (50, 50), (width-50, height-50), (0, 255, 0), 2)
    # Add some text
    cv2.putText(frame, "Test Frame", (width//4, height//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


class TestFrameProcessor:
    """Tests for FrameProcessor class."""
    
    def test_init_default(self):
        """Test initialization with default parameters."""
        processor = FrameProcessor()
        
        assert processor.resize_dims is None
        assert processor.convert_rgb is False
        assert processor.normalize is False
        assert processor.custom_processors == []
    
    def test_init_with_options(self):
        """Test initialization with custom parameters."""
        def custom_func(frame):
            return frame
            
        processor = FrameProcessor(
            resize_dims=(320, 240),
            convert_rgb=True,
            normalize=True,
            custom_processors=[custom_func]
        )
        
        assert processor.resize_dims == (320, 240)
        assert processor.convert_rgb is True
        assert processor.normalize is True
        assert len(processor.custom_processors) == 1
    
    def test_process_resize(self):
        """Test frame resizing."""
        processor = FrameProcessor(resize_dims=(320, 240))
        
        # Process a test frame
        test_frame = create_test_frame(640, 480)
        result, metadata = processor.process(test_frame)
        
        # Verify result
        assert result.shape == (240, 320, 3)
        assert "resize" in metadata["operations"]
        assert metadata["resize_dims"] == (320, 240)
    
    def test_process_convert_rgb(self):
        """Test color space conversion."""
        processor = FrameProcessor(convert_rgb=True)
        
        # Create a test frame with different R and B channels
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        test_frame[:, :, 0] = 100  # B channel in BGR
        test_frame[:, :, 2] = 200  # R channel in BGR
        
        result, metadata = processor.process(test_frame)
        
        # In RGB, the first channel should now be red (was blue)
        assert result[0, 0, 0] == 200  # R channel in RGB
        assert result[0, 0, 2] == 100  # B channel in RGB
        assert "bgr2rgb" in metadata["operations"]
    
    def test_process_normalize(self):
        """Test frame normalization."""
        processor = FrameProcessor(normalize=True)
        
        # Create a test frame with known values
        test_frame = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        result, metadata = processor.process(test_frame)
        
        # Verify normalization
        assert result.dtype == np.float32
        assert np.allclose(result, 1.0)
        assert "normalize" in metadata["operations"]
    
    def test_process_custom_processor(self):
        """Test custom processor function."""
        # Define a custom processor that inverts colors
        def invert_colors(frame):
            return 255 - frame
            
        processor = FrameProcessor(custom_processors=[invert_colors])
        
        # Create a test frame with known values
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        test_frame[10:20, 10:20] = 255  # White square
        
        result, metadata = processor.process(test_frame)
        
        # Verify inversion
        assert np.all(result[0, 0] == 255)  # Black became white
        assert np.all(result[15, 15] == 0)  # White became black
        assert "custom_0" in metadata["operations"]
    
    def test_process_error_in_custom_processor(self):
        """Test error handling in custom processor."""
        # Define a custom processor that raises an exception
        def faulty_processor(frame):
            raise ValueError("Test error")
            
        processor = FrameProcessor(custom_processors=[faulty_processor])
        
        # Process a test frame
        test_frame = create_test_frame()
        result, metadata = processor.process(test_frame)
        
        # Verify that processing continued despite the error
        assert result.shape == test_frame.shape
        assert "error_processor_{i}" in metadata
    
    def test_process_all_operations(self):
        """Test all operations combined."""
        # Define a simple custom processor
        def add_brightness(frame):
            return frame + 10
            
        processor = FrameProcessor(
            resize_dims=(320, 240),
            convert_rgb=True,
            normalize=True,
            custom_processors=[add_brightness]
        )
        
        # Process a test frame
        test_frame = create_test_frame(640, 480)
        result, metadata = processor.process(test_frame)
        
        # Verify result
        assert result.shape == (240, 320, 3)
        assert result.dtype == np.float32
        assert set(metadata["operations"]) == {"resize", "bgr2rgb", "normalize", "custom_0"}
    
    def test_get_stats(self):
        """Test statistics gathering and retrieval."""
        processor = FrameProcessor()
        
        # Process multiple frames
        test_frame = create_test_frame()
        for _ in range(5):
            processor.process(test_frame)
        
        # Get stats
        stats = processor.get_stats()
        
        # Verify stats
        assert stats["total_frames"] == 5
        assert stats["avg_processing_time"] > 0
        assert stats["max_processing_time"] > 0
        assert "frame_times" not in stats  # Should be removed to keep output manageable
    
    def test_reset_stats(self):
        """Test statistics reset."""
        processor = FrameProcessor()
        
        # Process a frame
        test_frame = create_test_frame()
        processor.process(test_frame)
        
        # Reset stats
        processor.reset_stats()
        
        # Check stats are reset
        stats = processor.get_stats()
        assert stats["total_frames"] == 0
        assert stats["processing_time"] == 0.0
        assert stats["avg_processing_time"] == 0.0 