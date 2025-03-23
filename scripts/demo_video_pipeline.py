#!/usr/bin/env python
"""Demo of the video input pipeline.

This script demonstrates the video input pipeline with various sources.
"""

import cv2
import argparse
import logging
import sys
import time
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.video_input import WebcamSource, FileSource, RTSPSource, FrameProcessor, VideoPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('video_pipeline_demo')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Video input pipeline demo')
    
    # Source selection - make it a mutually exclusive group but not required
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument('--webcam', type=int, default=0, 
                            help='Webcam device index (default: 0)')
    source_group.add_argument('--video', type=str, 
                            help='Path to video file')
    source_group.add_argument('--rtsp', type=str, 
                            help='RTSP stream URL')
    
    # Processing options
    parser.add_argument('--width', type=int, default=640,
                      help='Processing width')
    parser.add_argument('--height', type=int, default=480,
                      help='Processing height')
    parser.add_argument('--fps', type=float, default=30.0,
                      help='Target FPS')
    parser.add_argument('--no-threading', action='store_true',
                      help='Disable threading')
    parser.add_argument('--buffer-size', type=int, default=30,
                      help='Frame buffer size')
    parser.add_argument('--convert-rgb', action='store_true',
                      help='Convert BGR to RGB')
    parser.add_argument('--normalize', action='store_true',
                      help='Normalize pixel values')
    parser.add_argument('--display', action='store_true',
                      help='Display video')
    parser.add_argument('--info-interval', type=float, default=5.0,
                      help='Status info interval in seconds')
    
    return parser.parse_args()

def main():
    """Run the video pipeline demo."""
    args = parse_args()
    
    # Create frame processor
    processor = FrameProcessor(
        resize_dims=(args.width, args.height),
        convert_rgb=args.convert_rgb,
        normalize=args.normalize
    )
    
    # Create appropriate video source
    if args.video:
        if not os.path.exists(args.video):
            logger.error(f"Video file not found: {args.video}")
            return
        logger.info(f"Using video file source: {args.video}")
        source = FileSource(args.video)
    elif args.rtsp:
        logger.info(f"Using RTSP stream source: {args.rtsp}")
        source = RTSPSource(args.rtsp)
    else:
        # Default to webcam
        logger.info(f"Using webcam source (device {args.webcam})")
        source = WebcamSource(device_index=args.webcam)
    
    # Create pipeline
    pipeline = VideoPipeline(
        video_source=source,
        frame_processor=processor,
        buffer_size=args.buffer_size,
        target_fps=args.fps,
        enable_threading=not args.no_threading
    )
    
    # Start the pipeline
    if not pipeline.start():
        logger.error("Failed to start video pipeline")
        return
    
    try:
        # Process frames
        frame_count = 0
        start_time = time.time()
        last_info_time = start_time
        
        while True:
            # Read frame from pipeline
            success, frame, metadata = pipeline.read()
            
            if not success:
                logger.info("End of video stream")
                break
            
            # Process frame (actually handled by the pipeline)
            frame_count += 1
            
            # Display frame if requested
            if args.display:
                # Convert back to uint8 if normalized
                if args.normalize:
                    frame = (frame * 255).astype('uint8')
                
                # Convert back to BGR if RGB
                if args.convert_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Add metadata text
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                proc_time = metadata.get('processing_time', 0)
                cv2.putText(frame, f"Process: {proc_time:.1f}ms", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Display the frame
                cv2.imshow('Video Pipeline Demo', frame)
                
                # Break on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Log status periodically
            current_time = time.time()
            if current_time - last_info_time >= args.info_interval:
                elapsed = current_time - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                stats = pipeline.get_stats()
                logger.info(f"Processed {frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")
                logger.info(f"Pipeline stats: {stats['processed_frames']} frames, "
                           f"avg time: {stats.get('avg_frame_time', 0) * 1000:.1f}ms, "
                           f"buffer: {stats['buffer_utilization']:.1%}")
                
                last_info_time = current_time
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Stop the pipeline
        pipeline.stop()
        
        # Clean up
        if args.display:
            cv2.destroyAllWindows()
        
        # Show final stats
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        logger.info(f"Total: {frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")

if __name__ == '__main__':
    main() 