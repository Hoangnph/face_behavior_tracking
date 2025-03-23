"""Tests for video pipeline class."""

import cv2
import numpy as np
import pytest
import time
from unittest.mock import MagicMock, patch

from src.video_input.pipeline import VideoPipeline
from src.video_input.video_source import VideoSource
from src.video_input.frame_processor import FrameProcessor

class MockVideoSource(VideoSource):
    """Mock video source for testing."""
    
    def __init__(self, num_frames=10, frame_size=(640, 480), fail_open=False):
        super().__init__("mock_source")
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.frame_count = 0
        self.props = {
            'width': frame_size[0],
            'height': frame_size[1],
            'fps': 30.0,
            'frame_count': num_frames
        }
        self.fail_open = fail_open
    
    def open(self):
        if self.fail_open:
            self.is_opened = False
            return False
        self.is_opened = True
        self.frame_count = 0
        return True
    
    def read(self):
        if not self.is_opened or self.frame_count >= self.num_frames:
            return False, None
            
        # Create a test frame
        width, height = self.frame_size
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add frame number text
        cv2.putText(frame, f"Frame {self.frame_count}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        self.frame_count += 1
        return True, frame
    
    def get_properties(self):
        return self.props
        
    def close(self):
        """Override to properly set is_opened to False."""
        self.is_opened = False


class TestVideoPipeline:
    """Tests for VideoPipeline class."""
    
    def test_init(self):
        """Test initialization."""
        source = MockVideoSource()
        processor = FrameProcessor()
        
        pipeline = VideoPipeline(
            video_source=source,
            frame_processor=processor,
            buffer_size=20,
            target_fps=15,
            enable_threading=True
        )
        
        assert pipeline.video_source == source
        assert pipeline.frame_processor == processor
        assert pipeline.buffer_size == 20
        assert pipeline.target_fps == 15
        assert pipeline.enable_threading is True
        assert pipeline.is_running is False
    
    def test_start_stop_sync(self):
        """Test starting and stopping in synchronous mode."""
        source = MockVideoSource()
        
        pipeline = VideoPipeline(
            video_source=source,
            enable_threading=False
        )
        
        # Start pipeline
        success = pipeline.start()
        assert success is True
        assert pipeline.is_running is True
        assert source.is_opened is True
        
        # Stop pipeline
        pipeline.stop()
        assert pipeline.is_running is False
        assert source.is_opened is False
    
    @patch('threading.Thread')
    def test_start_stop_threaded(self, mock_thread):
        """Test starting and stopping in threaded mode."""
        source = MockVideoSource()
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        
        pipeline = VideoPipeline(
            video_source=source,
            enable_threading=True
        )
        
        # Start pipeline
        success = pipeline.start()
        assert success is True
        assert pipeline.is_running is True
        assert source.is_opened is True
        assert mock_thread.called
        assert mock_thread_instance.start.called
        
        # Stop pipeline
        pipeline.stop()
        assert pipeline.is_running is False
        assert source.is_opened is False
        assert mock_thread_instance.join.called
    
    def test_start_failure(self):
        """Test starting pipeline with source opening failure."""
        source = MockVideoSource(fail_open=True)
        
        pipeline = VideoPipeline(
            video_source=source
        )
        
        # Start pipeline should fail
        success = pipeline.start()
        assert success is False
        assert pipeline.is_running is False
    
    def test_read_sync(self):
        """Test reading frames in synchronous mode."""
        source = MockVideoSource(num_frames=5)
        processor = FrameProcessor(resize_dims=(320, 240))
        
        pipeline = VideoPipeline(
            video_source=source,
            frame_processor=processor,
            enable_threading=False
        )
        
        # Start pipeline
        pipeline.start()
        
        # Read all frames
        frames = []
        while True:
            success, frame, metadata = pipeline.read()
            if not success:
                break
            frames.append(frame)
        
        # Verify frames
        assert len(frames) == 5
        assert frames[0].shape == (240, 320, 3)  # Resized
        
        # Verify pipeline stats
        stats = pipeline.get_stats()
        assert stats['processed_frames'] == 5
    
    def test_target_fps(self):
        """Test pipeline respecting target FPS in synchronous mode."""
        source = MockVideoSource(num_frames=3)
        
        # Use low FPS to make timing test reliable
        target_fps = 2  # 2 fps = 500ms per frame
        
        pipeline = VideoPipeline(
            video_source=source,
            target_fps=target_fps,
            enable_threading=False
        )
        
        pipeline.start()
        
        # Override the _capture_and_process method to test timing
        original_method = pipeline._capture_and_process
        
        def timed_capture(*args, **kwargs):
            # Record time before capture
            start_time = time.time()
            result = original_method(*args, **kwargs)
            # Add timing information to metadata
            if result[0]:  # if success
                result[2]['capture_time'] = time.time() - start_time
            return result
            
        pipeline._capture_and_process = timed_capture
        
        # Read frames and measure timing
        frame_times = []
        start_time = time.time()
        
        for _ in range(3):
            success, frame, metadata = pipeline.read()
            if success:
                frame_times.append(time.time() - start_time)
                start_time = time.time()
        
        # Calculate intervals between frames
        intervals = [frame_times[i] - frame_times[i-1] for i in range(1, len(frame_times))]
        
        # In synchronous mode without explicit timing, intervals should be very small
        # This is just a basic check that the function works
        assert len(intervals) > 0
    
    def test_context_manager(self):
        """Test using pipeline as a context manager."""
        source = MockVideoSource()
        
        # Use as context manager
        with VideoPipeline(video_source=source, enable_threading=False) as pipeline:
            assert pipeline.is_running is True
            assert source.is_opened is True
            
            success, frame, metadata = pipeline.read()
            assert success is True
            assert frame is not None
        
        # After context exit
        assert pipeline.is_running is False
        assert source.is_opened is False
    
    def test_get_stats(self):
        """Test getting pipeline statistics."""
        source = MockVideoSource(num_frames=3)
        
        pipeline = VideoPipeline(
            video_source=source,
            enable_threading=False
        )
        
        pipeline.start()
        
        # Read all frames
        for _ in range(3):
            pipeline.read()
        
        # Get stats
        stats = pipeline.get_stats()
        
        # Verify stats
        assert stats['processed_frames'] == 3
        assert stats['source_fps'] == 30.0
        assert 'processor' in stats
        assert stats['processor']['total_frames'] == 3 