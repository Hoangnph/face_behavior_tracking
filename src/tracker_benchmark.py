#!/usr/bin/env python
"""
Benchmark script for comparing different tracking methods.

This script evaluates and compares the performance of the AFC and DeepSORT
trackers on the same video sequence, measuring tracking accuracy, speed,
and other metrics.
"""

import argparse
import cv2
import numpy as np
import os
import time
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

# Local imports
from src.tracking.afc_tracker import AFCTracker
from src.tracking.deep_sort_tracker import DeepSORTTracker, DEEP_FEATURES_AVAILABLE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PersonDetector:
    """Person detector using OpenCV DNN."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the person detector.
        
        Args:
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        
        # Initialize OpenCV DNN person detector
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        prototxt_path = os.path.join(model_dir, 'MobileNetSSD_deploy.prototxt')
        model_path = os.path.join(model_dir, 'MobileNetSSD_deploy.caffemodel')
        
        # Check if model files exist
        if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
            logger.warning(f"Model files not found. Using a placeholder detector.")
            self.model = None
        else:
            self.model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            logger.info("Initialized OpenCV DNN detector")
    
    def detect(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Detect persons in the frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            List of detections in format [x, y, w, h, confidence]
        """
        if self.model is None:
            # Placeholder detection - just return a few random boxes
            detections = []
            h, w = frame.shape[:2]
            
            # Generate some random detections
            num_detections = np.random.randint(1, 5)
            for _ in range(num_detections):
                x = np.random.randint(0, w - 50)
                y = np.random.randint(0, h - 100)
                w_box = np.random.randint(50, 150)
                h_box = np.random.randint(100, 200)
                conf = np.random.uniform(0.6, 0.9)
                detections.append(np.array([x, y, w_box, h_box, conf]))
            
            return detections
        
        # Preprocess frame for MobileNet SSD
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5
        )
        
        # Set the input and perform forward pass
        self.model.setInput(blob)
        detections = self.model.forward()
        
        # Filter detections
        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])
            
            # Class 15 is person in MobileNet SSD
            if class_id == 15 and confidence > self.confidence_threshold:
                # Convert coordinates to absolute values
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                x1, y1, x2, y2 = box.astype(int)
                
                # Convert to [x, y, w, h, confidence]
                x, y = x1, y1
                w, h = x2 - x1, y2 - y1
                results.append(np.array([x, y, w, h, confidence]))
        
        return results

def load_ground_truth(ground_truth_path: str) -> Dict[int, List[List[int]]]:
    """
    Load ground truth annotations from a file.
    
    Args:
        ground_truth_path: Path to ground truth file
        
    Returns:
        Dictionary mapping frame indices to lists of bounding boxes [x, y, w, h, id]
    """
    ground_truth = {}
    
    if not os.path.exists(ground_truth_path):
        logger.warning(f"Ground truth file {ground_truth_path} not found.")
        return ground_truth
    
    try:
        with open(ground_truth_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    frame_idx = int(parts[0])
                    obj_id = int(parts[1])
                    x, y, w, h = map(int, parts[2:6])
                    
                    if frame_idx not in ground_truth:
                        ground_truth[frame_idx] = []
                    
                    ground_truth[frame_idx].append([x, y, w, h, obj_id])
    except Exception as e:
        logger.error(f"Error loading ground truth: {e}")
    
    logger.info(f"Loaded ground truth for {len(ground_truth)} frames")
    return ground_truth

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.
    
    Args:
        box1: First bounding box [x, y, w, h]
        box2: Second bounding box [x, y, w, h]
        
    Returns:
        IoU score (0-1)
    """
    # Convert to xyxy format
    x1_1, y1_1 = box1[0], box1[1]
    x2_1, y2_1 = box1[0] + box1[2], box1[1] + box1[3]
    
    x1_2, y1_2 = box2[0], box2[1]
    x2_2, y2_2 = box2[0] + box2[2], box2[1] + box2[3]
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        # No intersection
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / max(union_area, 1e-6)
    
    return iou

def run_benchmark(
    video_path: str,
    ground_truth_path: Optional[str] = None,
    output_dir: str = 'data/benchmark',
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    show_display: bool = False,
    max_frames: Optional[int] = None
) -> Dict:
    """
    Run benchmark to compare different tracking methods.
    
    Args:
        video_path: Path to input video
        ground_truth_path: Path to ground truth annotations (optional)
        output_dir: Directory to save benchmark results
        confidence_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for matching predictions with ground truth
        show_display: Whether to show visualization
        max_frames: Maximum number of frames to process (None for all)
        
    Returns:
        Dictionary containing benchmark metrics
    """
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video {video_path}")
        return {}
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    
    logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize output video writers
    afc_output_path = os.path.join(output_dir, 'benchmark_afc.mp4')
    deepsort_output_path = os.path.join(output_dir, 'benchmark_deepsort.mp4')
    
    afc_writer = cv2.VideoWriter(afc_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    deepsort_writer = cv2.VideoWriter(deepsort_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # Initialize trackers
    afc_tracker = AFCTracker(
        correlation_threshold=0.5,
        n_init=3,
        max_age=30
    )
    
    deepsort_tracker = DeepSORTTracker(
        max_iou_distance=0.7,
        max_feature_distance=0.5,
        n_init=3,
        max_age=30
    )
    
    logger.info("Initialized trackers for benchmark")
    
    # Initialize person detector
    detector = PersonDetector(confidence_threshold=confidence_threshold)
    
    # Load ground truth if available
    ground_truth = {}
    if ground_truth_path:
        ground_truth = load_ground_truth(ground_truth_path)
    
    # Performance metrics
    metrics = {
        'afc': {
            'processing_times': [],
            'track_counts': [],
            'accuracy': [] if ground_truth else None,
            'track_switches': 0,
            'false_positives': 0,
            'false_negatives': 0
        },
        'deepsort': {
            'processing_times': [],
            'track_counts': [],
            'accuracy': [] if ground_truth else None,
            'track_switches': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
    }
    
    # Track consistency tracking
    prev_tracks = {
        'afc': {},
        'deepsort': {}
    }
    
    # Process frames
    frame_count = 0
    while frame_count < total_frames:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get detections
        detections = detector.detect(frame)
        
        # Process with AFC tracker
        afc_start_time = time.time()
        afc_results = afc_tracker.update(frame, detections)
        afc_elapsed = time.time() - afc_start_time
        metrics['afc']['processing_times'].append(afc_elapsed)
        metrics['afc']['track_counts'].append(len(afc_results))
        
        # Process with DeepSORT tracker
        deepsort_start_time = time.time()
        deepsort_results = deepsort_tracker.update(frame, detections)
        deepsort_elapsed = time.time() - deepsort_start_time
        metrics['deepsort']['processing_times'].append(deepsort_elapsed)
        metrics['deepsort']['track_counts'].append(len(deepsort_results))
        
        # Check track consistency
        for tracker_name, results in [('afc', afc_results), ('deepsort', deepsort_results)]:
            # Convert results to dictionary mapping track IDs to bounding boxes
            current_tracks = {track_id: (x, y, w, h) for track_id, x, y, w, h in results}
            
            # Check for track switches (same box, different ID)
            if prev_tracks[tracker_name]:
                for curr_id, curr_box in current_tracks.items():
                    for prev_id, prev_box in prev_tracks[tracker_name].items():
                        # If the same object (high IoU) has different IDs, count as a track switch
                        iou = calculate_iou(curr_box, prev_box)
                        if iou > 0.7 and curr_id != prev_id:
                            metrics[tracker_name]['track_switches'] += 1
            
            # Update previous tracks
            prev_tracks[tracker_name] = current_tracks
        
        # Check accuracy against ground truth if available
        if ground_truth and frame_count in ground_truth:
            gt_boxes = ground_truth[frame_count]
            
            for tracker_name, results in [('afc', afc_results), ('deepsort', deepsort_results)]:
                true_positives = 0
                matched_gt = set()
                
                # Match predictions with ground truth
                for _, x, y, w, h in results:
                    pred_box = np.array([x, y, w, h])
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for i, (gt_x, gt_y, gt_w, gt_h, _) in enumerate(gt_boxes):
                        if i in matched_gt:
                            continue
                        
                        gt_box = np.array([gt_x, gt_y, gt_w, gt_h])
                        iou = calculate_iou(pred_box, gt_box)
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = i
                    
                    if best_iou > iou_threshold:
                        true_positives += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        metrics[tracker_name]['false_positives'] += 1
                
                # Unmatched ground truth boxes are false negatives
                metrics[tracker_name]['false_negatives'] += len(gt_boxes) - len(matched_gt)
                
                # Calculate accuracy for this frame
                precision = true_positives / max(len(results), 1)
                recall = true_positives / max(len(gt_boxes), 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-6)
                metrics[tracker_name]['accuracy'].append(f1)
        
        # Draw results on frame
        afc_frame = frame.copy()
        deepsort_frame = frame.copy()
        
        # Draw AFC results
        for track_id, x, y, w, h in afc_results:
            cv2.rectangle(afc_frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            label = f"ID: {track_id}"
            cv2.putText(afc_frame, label, (int(x), int(y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw DeepSORT results
        for track_id, x, y, w, h in deepsort_results:
            cv2.rectangle(deepsort_frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
            label = f"ID: {track_id}"
            cv2.putText(deepsort_frame, label, (int(x), int(y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add frame information
        for frame_img, tracker_name in [(afc_frame, 'AFC'), (deepsort_frame, 'DeepSORT')]:
            cv2.putText(frame_img, f"Frame: {frame_count}/{total_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if tracker_name == 'AFC':
                avg_time = np.mean(metrics['afc']['processing_times']) if metrics['afc']['processing_times'] else 0
                cv2.putText(frame_img, f"{tracker_name} - {1/avg_time:.1f} FPS", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                avg_time = np.mean(metrics['deepsort']['processing_times']) if metrics['deepsort']['processing_times'] else 0
                cv2.putText(frame_img, f"{tracker_name} - {1/avg_time:.1f} FPS", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frames to output videos
        afc_writer.write(afc_frame)
        deepsort_writer.write(deepsort_frame)
        
        # Show visualization
        if show_display:
            cv2.imshow('AFC Tracker', afc_frame)
            cv2.imshow('DeepSORT Tracker', deepsort_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Increment frame counter
        frame_count += 1
        
        # Log progress
        if frame_count % 50 == 0:
            logger.info(f"Processed {frame_count}/{total_frames} frames")
    
    # Release resources
    cap.release()
    afc_writer.release()
    deepsort_writer.release()
    if show_display:
        cv2.destroyAllWindows()
    
    # Calculate final metrics
    for tracker in ['afc', 'deepsort']:
        metrics[tracker]['avg_processing_time'] = np.mean(metrics[tracker]['processing_times']) if metrics[tracker]['processing_times'] else 0
        metrics[tracker]['avg_fps'] = 1 / metrics[tracker]['avg_processing_time'] if metrics[tracker]['avg_processing_time'] > 0 else 0
        metrics[tracker]['avg_track_count'] = np.mean(metrics[tracker]['track_counts']) if metrics[tracker]['track_counts'] else 0
        
        if metrics[tracker]['accuracy'] is not None:
            metrics[tracker]['avg_accuracy'] = np.mean(metrics[tracker]['accuracy']) if metrics[tracker]['accuracy'] else 0
    
    # Save visualization of results
    plot_path = os.path.join(output_dir, 'benchmark_results.png')
    plot_benchmark_results(metrics, plot_path)
    
    # Log results
    logger.info("=== Benchmark Results ===")
    logger.info(f"Video: {video_path}")
    logger.info(f"Frames processed: {frame_count}")
    
    logger.info("\nAFC Tracker:")
    logger.info(f"Average processing time: {metrics['afc']['avg_processing_time']*1000:.2f} ms")
    logger.info(f"Average FPS: {metrics['afc']['avg_fps']:.2f}")
    logger.info(f"Average track count: {metrics['afc']['avg_track_count']:.2f}")
    logger.info(f"Track switches: {metrics['afc']['track_switches']}")
    
    if metrics['afc']['accuracy'] is not None:
        logger.info(f"Average accuracy (F1): {metrics['afc']['avg_accuracy']:.4f}")
        logger.info(f"False positives: {metrics['afc']['false_positives']}")
        logger.info(f"False negatives: {metrics['afc']['false_negatives']}")
    
    logger.info("\nDeepSORT Tracker:")
    logger.info(f"Average processing time: {metrics['deepsort']['avg_processing_time']*1000:.2f} ms")
    logger.info(f"Average FPS: {metrics['deepsort']['avg_fps']:.2f}")
    logger.info(f"Average track count: {metrics['deepsort']['avg_track_count']:.2f}")
    logger.info(f"Track switches: {metrics['deepsort']['track_switches']}")
    
    if metrics['deepsort']['accuracy'] is not None:
        logger.info(f"Average accuracy (F1): {metrics['deepsort']['avg_accuracy']:.4f}")
        logger.info(f"False positives: {metrics['deepsort']['false_positives']}")
        logger.info(f"False negatives: {metrics['deepsort']['false_negatives']}")
    
    logger.info(f"\nResults saved to {output_dir}")
    
    return metrics

def plot_benchmark_results(metrics: Dict, output_path: str):
    """
    Plot benchmark results.
    
    Args:
        metrics: Dictionary containing benchmark metrics
        output_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Processing time comparison
    plt.subplot(2, 2, 1)
    plt.bar(['AFC', 'DeepSORT'], 
           [metrics['afc']['avg_processing_time']*1000, 
            metrics['deepsort']['avg_processing_time']*1000])
    plt.ylabel('Processing Time (ms)')
    plt.title('Average Processing Time')
    
    # Plot 2: FPS comparison
    plt.subplot(2, 2, 2)
    plt.bar(['AFC', 'DeepSORT'], 
           [metrics['afc']['avg_fps'], 
            metrics['deepsort']['avg_fps']])
    plt.ylabel('Frames Per Second')
    plt.title('Average FPS')
    
    # Plot 3: Track count over time
    plt.subplot(2, 2, 3)
    frames = range(len(metrics['afc']['track_counts']))
    plt.plot(frames, metrics['afc']['track_counts'], label='AFC')
    plt.plot(frames, metrics['deepsort']['track_counts'], label='DeepSORT')
    plt.xlabel('Frame')
    plt.ylabel('Number of Tracks')
    plt.title('Track Count Over Time')
    plt.legend()
    
    # Plot 4: Accuracy over time or track switches
    plt.subplot(2, 2, 4)
    if metrics['afc']['accuracy'] is not None:
        frames = range(len(metrics['afc']['accuracy']))
        plt.plot(frames, metrics['afc']['accuracy'], label='AFC')
        plt.plot(frames, metrics['deepsort']['accuracy'], label='DeepSORT')
        plt.xlabel('Frame')
        plt.ylabel('F1 Score')
        plt.title('Tracking Accuracy Over Time')
        plt.legend()
    else:
        plt.bar(['AFC', 'DeepSORT'], 
               [metrics['afc']['track_switches'], 
                metrics['deepsort']['track_switches']])
        plt.ylabel('Count')
        plt.title('Track ID Switches')
    
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Benchmark visualization saved to {output_path}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Benchmark different tracking methods")
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--ground_truth", type=str, default=None,
        help="Path to ground truth annotations file (optional)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/benchmark",
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--iou", type=float, default=0.5,
        help="IoU threshold for matching predictions with ground truth"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show visualization during benchmark"
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Maximum number of frames to process (default: all frames)"
    )
    
    args = parser.parse_args()
    
    run_benchmark(
        video_path=args.video,
        ground_truth_path=args.ground_truth,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        show_display=args.show,
        max_frames=args.max_frames
    )

if __name__ == "__main__":
    main() 