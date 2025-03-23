# Video Input Pipeline Implementation

## Overview

The video input pipeline module provides a flexible and efficient way to capture, process, and manage video streams from various sources (webcam, video files, RTSP streams). It forms the foundation for the human tracking and behavior analysis system, providing frames to downstream components.

## Architecture

The video input pipeline consists of the following key components:

### 1. Video Sources

The module provides an abstract base class `VideoSource` and three implementations:

- **WebcamSource**: Captures frames from connected webcams and camera devices
- **FileSource**: Reads frames from video files (.mp4, .avi, etc.)
- **RTSPSource**: Connects to and streams from network cameras using RTSP protocol

All sources implement a common interface with methods like `open()`, `close()`, `read()`, and `get_properties()`.

### 2. Frame Processor

The `FrameProcessor` class handles preprocessing operations on captured frames:

- Resizing frames to specified dimensions
- Converting color spaces (BGR to RGB)
- Normalizing pixel values (0-1 range)
- Applying custom processing functions
- Tracking performance statistics

### 3. Video Pipeline

The `VideoPipeline` class brings together the video sources and frame processor:

- Manages the video source and frame processor objects
- Provides threaded operation with frame buffering
- Controls frame rate
- Handles resource management
- Provides pipeline statistics

## Usage Examples

### Basic Webcam Access

```python
from src.video_input import WebcamSource, FrameProcessor, VideoPipeline

# Create components
source = WebcamSource(device_index=0)
processor = FrameProcessor(resize_dims=(640, 480), convert_rgb=True)

# Create and start pipeline
pipeline = VideoPipeline(source, processor)
pipeline.start()

# Process frames
while True:
    success, frame, metadata = pipeline.read()
    if not success:
        break
    
    # Use frame for downstream processing
    # ...

# Clean up resources
pipeline.stop()
```

### Video File Processing

```python
from src.video_input import FileSource, FrameProcessor, VideoPipeline

# Create components
source = FileSource("path/to/video.mp4")
processor = FrameProcessor(resize_dims=(320, 240))

# Use context manager for automatic resource cleanup
with VideoPipeline(source, processor) as pipeline:
    while True:
        success, frame, metadata = pipeline.read()
        if not success:
            break
        
        # Use frame for downstream processing
        # ...
```

### RTSP Stream Access

```python
from src.video_input import RTSPSource, FrameProcessor, VideoPipeline

# Create components
source = RTSPSource("rtsp://camera_ip:port/stream")
processor = FrameProcessor(
    resize_dims=(640, 480),
    convert_rgb=True,
    normalize=True
)

# Create pipeline with custom settings
pipeline = VideoPipeline(
    video_source=source,
    frame_processor=processor,
    buffer_size=60,  # larger buffer for network streams
    target_fps=15,   # lower target FPS
    enable_threading=True
)

# Use pipeline
# ...
```

## Demo Script

The module includes a demonstration script at `scripts/demo_video_pipeline.py` that showcases all functionality:

```bash
# Webcam demo
python scripts/demo_video_pipeline.py --display

# Video file demo
python scripts/demo_video_pipeline.py --video data/videos/sample.mp4 --display

# With additional options
python scripts/demo_video_pipeline.py --display --width 320 --height 240 --convert-rgb
```

Run with `--help` to see all available options.

## Performance Considerations

- Use threaded mode (`enable_threading=True`) for real-time applications
- Adjust buffer size based on available memory and application requirements
- For optimal performance, match processing resolution to downstream requirements
- Monitor pipeline statistics (`pipeline.get_stats()`) to identify bottlenecks

## Testing

Comprehensive unit tests are available in the `tests/video_input` directory:

- `test_video_source.py`: Tests for video source classes
- `test_frame_processor.py`: Tests for frame processor
- `test_pipeline.py`: Tests for video pipeline

Run tests with:

```bash
python -m pytest tests/video_input -v
``` 