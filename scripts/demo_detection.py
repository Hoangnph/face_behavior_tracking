#!/usr/bin/env python
"""
Demo script for face and person detection modules.

Usage:
    python scripts/demo_detection.py --webcam
    python scripts/demo_detection.py --video PATH_TO_VIDEO
    python scripts/demo_detection.py --image PATH_TO_IMAGE
    python scripts/demo_detection.py --sample-faces  # Use sample faces from data/faces
    python scripts/demo_detection.py --sample-video  # Use sample video from data/videos/sample_2.mp4

Options:
    --webcam           Use webcam as input source
    --video PATH       Path to video file
    --image PATH       Path to image file
    --sample-faces     Use sample faces from data/faces directory
    --sample-video     Use sample video from data/videos/sample_2.mp4
    --face             Enable face detection
    --person           Enable person detection
    --onnx             Use ONNX runtime for person detection
    --confidence CONF  Detection confidence threshold (default: 0.5)
    --display          Display detection results
    --output PATH      Save output to file
    --help             Show this help message
"""

import os
import sys
import argparse
import time
import glob
import cv2
import numpy as np
from src.video_input import WebcamSource, FileSource, FrameProcessor, VideoPipeline
from src.detection import (
    MediaPipeFaceDetector, FaceDetection,
    YOLOPersonDetector, PersonDetection,
    ONNXDetector, convert_to_onnx,
    DetectionScheduler
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Detection module demo')
    
    # Input sources
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--webcam', action='store_true', help='Use webcam as input source')
    input_group.add_argument('--video', type=str, help='Path to video file')
    input_group.add_argument('--image', type=str, help='Path to image file')
    input_group.add_argument('--sample-faces', action='store_true', help='Use sample faces from data/faces directory')
    input_group.add_argument('--sample-video', action='store_true', help='Use sample video from data/videos/sample_2.mp4')
    
    # Detection options
    parser.add_argument('--face', action='store_true', help='Enable face detection')
    parser.add_argument('--person', action='store_true', help='Enable person detection')
    parser.add_argument('--onnx', action='store_true', help='Use ONNX runtime for person detection')
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold')
    
    # Output options
    parser.add_argument('--display', action='store_true', help='Display detection results')
    parser.add_argument('--output', type=str, help='Save output to file')
    parser.add_argument('--output-dir', type=str, default='data/output', help='Directory to save output files')
    
    args = parser.parse_args()
    
    # If no detection type specified, enable both
    if not (args.face or args.person):
        args.face = True
        args.person = True
    
    return args


def main():
    """Main function."""
    args = parse_args()
    
    # Set up default paths
    if args.sample_video:
        args.video = "data/videos/sample_2.mp4"
        if not os.path.exists(args.video):
            print(f"Error: Sample video not found at {args.video}")
            return 1
    
    # Initialize detectors
    detectors = {}
    
    if args.face:
        face_detector = MediaPipeFaceDetector(min_detection_confidence=args.confidence)
        detectors['face'] = face_detector
        print("Face detector initialized")
    
    if args.person:
        if args.onnx:
            # Check if ONNX model exists
            onnx_model_path = "models/onnx/yolov8n.onnx"
            yolo_model_path = "models/yolo/yolov8n.pt"
            
            if not os.path.exists(onnx_model_path):
                print(f"ONNX model not found at {onnx_model_path}")
                print("Converting YOLOv8 model to ONNX format...")
                os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)
                convert_to_onnx(yolo_model_path, onnx_model_path)
                print(f"ONNX model saved to {onnx_model_path}")
            
            person_detector = ONNXDetector(
                model_path=onnx_model_path,
                confidence_threshold=args.confidence,
                class_mapping={0: "person"}
            )
            print("ONNX person detector initialized")
        else:
            person_detector = YOLOPersonDetector(confidence_threshold=args.confidence)
            print("YOLOv8 person detector initialized")
        
        detectors['person'] = person_detector
    
    # Initialize detection scheduler if multiple detectors are used
    if len(detectors) > 1:
        scheduler = DetectionScheduler(
            detectors=detectors,
            default_detector="face" if "face" in detectors else "person"
        )
        print("Detection scheduler initialized")
    
    # Process sample faces if requested
    if args.sample_faces:
        process_sample_faces(detectors, args)
        return 0
    
    # Create video source
    if args.webcam:
        source = WebcamSource(device_index=0)
        processor = FrameProcessor(resize_dims=(640, 480))
    elif args.video:
        source = FileSource(args.video)
        processor = FrameProcessor()
    elif args.image:
        # Load image
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Cannot load image from {args.image}")
            return 1
        
        # Process single image
        process_single_image(image, detectors, args)
        return 0
    
    # Create video pipeline
    with VideoPipeline(source, processor) as pipeline:
        # Create output directory if needed
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            
        frame_count = 0
        processed_count = 0
        start_time = time.time()
        
        while True:
            success, frame, metadata = pipeline.read()
            if not success:
                break
            
            frame_count += 1
            
            # Skip frames if processing is slow (every 3rd frame for sample_2 which is larger)
            should_process = args.webcam or frame_count % 3 == 0
            
            if should_process:
                processed_count += 1
                
                # Process frame with detectors
                if len(detectors) > 1 and 'scheduler' in locals():
                    # Use scheduler for multiple detectors
                    all_detections = scheduler.detect_all(frame)
                    processed_frame = frame.copy()
                    
                    # Visualize detections based on their type
                    for detector_name, detector in detectors.items():
                        relevant_detections = [d for d in all_detections if d.label == detector_name or 
                                             (detector_name == 'person' and d.label == 'person') or
                                             (detector_name == 'face' and d.label == 'face')]
                        processed_frame = detector.visualize(processed_frame, relevant_detections)
                else:
                    # Process with individual detectors
                    processed_frame = process_frame(frame, detectors)
                
                # Display result
                if args.display:
                    cv2.imshow('Detection Demo', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Save output if specified
                if args.output:
                    if args.video or args.webcam:
                        # For video sources, initialize video writer on first frame
                        if not hasattr(main, 'video_writer'):
                            fps = source.get_properties().get('fps', 30)
                            h, w = processed_frame.shape[:2]
                            main.video_writer = cv2.VideoWriter(
                                args.output,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps,
                                (w, h)
                            )
                        
                        main.video_writer.write(processed_frame)
        
        # Print performance statistics
        elapsed_time = time.time() - start_time
        fps = processed_count / elapsed_time
        print(f"Processed {processed_count} frames out of {frame_count} in {elapsed_time:.2f} seconds ({fps:.2f} FPS)")
    
    # Clean up
    cv2.destroyAllWindows()
    if hasattr(main, 'video_writer'):
        main.video_writer.release()
    
    print("Testing completed successfully!")
    return 0


def process_frame(frame, detectors):
    """Process a frame with detection models."""
    result = frame.copy()
    
    # Run face detection if available
    if 'face' in detectors:
        face_detector = detectors['face']
        face_detections = face_detector.detect(frame)
        result = face_detector.visualize(result, face_detections)
    
    # Run person detection if available
    if 'person' in detectors:
        person_detector = detectors['person']
        person_detections = person_detector.detect(frame)
        result = person_detector.visualize(result, person_detections)
    
    return result


def process_single_image(image, detectors, args):
    """Process a single image with detection models."""
    # Process image
    processed_image = process_frame(image, detectors)
    
    # Display result
    if args.display:
        cv2.imshow('Detection Result', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Save output if specified
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        cv2.imwrite(args.output, processed_image)
        print(f"Output saved to {args.output}")


def process_sample_faces(detectors, args):
    """Process sample faces from the data/faces directory."""
    # Define paths
    face_dirs = [
        "data/faces/known_customers",
        "data/faces/employees"
    ]
    
    # Create output directory
    output_dir = args.output_dir or "data/output/faces"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all face images
    face_images = []
    for directory in face_dirs:
        if os.path.exists(directory):
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                face_images.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    
    if not face_images:
        print("No face images found in the specified directories")
        return
    
    print(f"Found {len(face_images)} face images to process")
    
    # Process each face image
    success_count = 0
    for i, image_path in enumerate(face_images):
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot load image from {image_path}")
            continue
        
        # Process image
        try:
            processed_image = process_frame(image, detectors)
            
            # Save output
            rel_path = os.path.relpath(image_path, "data/faces")
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, processed_image)
            
            # Display progress
            success_count += 1
            if i % 10 == 0:
                print(f"Processed {i+1}/{len(face_images)} images")
                
                # Display result periodically if requested
                if args.display and i < 20:  # Only show first 20 to avoid overwhelming display
                    cv2.imshow(f'Face Detection - {os.path.basename(image_path)}', processed_image)
                    cv2.waitKey(500)  # Show for 500ms
                    cv2.destroyAllWindows()
                    
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    print(f"Successfully processed {success_count}/{len(face_images)} face images")
    print(f"Results saved to {output_dir}")
    
    # Clean up
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main()) 