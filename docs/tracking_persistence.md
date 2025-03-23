# Persistent Object Tracking System

This document describes the implementation of persistent object tracking systems using AFC (Average Frame Correlation) and DeepSORT methods in the Face Behavior Tracking project.

## Introduction

Persistent object tracking is a critical functionality for maintaining the identity of objects (people) across multiple frames in a video. This feature allows the system to:

1. Maintain the identity (ID) of each person across frames, even when temporarily occluded
2. Track individual behavior over time
3. Minimize "ID switching" - when IDs get swapped between objects
4. Improve the accuracy of face recognition

## Tracking Methods

This project has developed and integrated three tracking methods:

### 1. AFC Tracker (Average Frame Correlation)

Tracking based on appearance correlation between adjacent frames:

- Advantages: Fast, lightweight, doesn't require GPU, suitable for systems with limited resources
- Disadvantages: May struggle when objects change appearance rapidly or are occluded for long periods

How it works:
1. Extract appearance templates from each detection
2. Calculate correlation matrix between templates of current tracks and new detections
3. Match tracks and detections based on correlation level
4. Maintain track IDs across frames

### 2. DeepSORT Tracker

Enhanced method based on SORT (Simple Online and Realtime Tracking) with added deep learning features:

- Advantages: High accuracy, handles occlusion well, robust to appearance changes
- Disadvantages: Requires more resources due to deep feature extractor (MobileNetV2)

How it works:
1. Use Kalman filter to predict new position of objects
2. Extract deep features from each detection using CNN (MobileNetV2)
3. Combine IoU and feature distance to match tracks and detections
4. Maintain and update tracks across frames

### 3. Ensemble Tracker

Combines the advantages of both methods above:

- Advantages: Highest accuracy, minimizes ID switching, handles complex situations well
- Disadvantages: Requires the most resources, slower speed

How it works:
1. Run both trackers (AFC and DeepSORT) in parallel
2. Prioritize results from DeepSORT (higher accuracy)
3. Supplement with results from AFC if DeepSORT doesn't detect an object
4. Combine results into a final track list

## Source Code Structure

```
src/
├── tracking/
│   ├── afc_tracker.py     # AFC Tracker implementation
│   ├── deep_sort_tracker.py  # DeepSORT Tracker implementation
│   └── ensemble_tracker.py   # Ensemble Tracker implementation (optional)
├── tracking_demo_persistent.py  # Demo for running persistent tracking
└── tracker_benchmark.py  # Benchmarking tool for tracking methods
```

## Usage

### Running the Demo

```bash
python src/tracking_demo_persistent.py --video [path/to/video] --known-faces [path/to/known_faces_dir] --output [path/to/output.mp4] --tracker [afc/deepsort/ensemble]
```

Parameters:
- `--video`: Path to video file (default: data/videos/sample_2.mp4)
- `--known-faces`: Directory containing known face images (default: data/known_faces)
- `--output`: Path to save result video (default: data/output/tracking_persistent.mp4)
- `--tracker`: Tracking method (afc, deepsort, ensemble) (default: afc)
- `--confidence`: Confidence threshold for detections (default: 0.5)
- `--no-display`: Don't display video during processing
- `--save-frames`: Save sample frames

### Running the Benchmark

```bash
python src/tracker_benchmark.py --video [path/to/video] --ground-truth [path/to/gt.txt] --output-dir [path/to/output_dir]
```

Parameters:
- `--video`: Path to video file
- `--ground-truth`: Path to ground truth file (optional)
- `--detections`: Path to pre-computed detections file (optional)
- `--output-dir`: Directory to save benchmark results (default: benchmark_results)
- `--iou-threshold`: IoU threshold for evaluation (default: 0.5)
- `--max-frames`: Maximum number of frames to process (-1 for all) (default: -1)

## Data File Formats

### Ground Truth File Format

```
frame_id,track_id,x1,y1,x2,y2
```

Example:
```
0,1,100,100,200,200
0,2,300,150,400,250
1,1,105,105,205,205
```

### Detections File Format

```
frame_id,x1,y1,x2,y2,confidence
```

Example:
```
0,100,100,200,200,0.95
0,300,150,400,250,0.85
1,105,105,205,205,0.92
```

## Performance Comparison

| Method | Precision | Recall | FPS | ID Switches | Track Robustness |
|-------------|-----------|--------|-----|-------------|--------------|
| AFC         | 0.92      | 0.88   | 172 | 12          | Medium       |
| DeepSORT    | 0.94      | 0.90   | 127 | 5           | Good         |
| Ensemble    | 0.95      | 0.92   | 78  | 3           | Very good    |

*Note: These figures are examples and may vary depending on system configuration and input data.*

## Applications

This persistent tracking system can be applied in various scenarios:

1. **Customer behavior analysis in stores**: Tracking customer movement within the store, time spent at product areas.
2. **Security systems**: Tracking suspicious individuals across multiple cameras.
3. **Sports analysis**: Tracking athletes and analyzing movements.
4. **Human-machine interaction**: Tracking users to create seamless interactive experiences.

## Future Work

1. Integration with behavior analysis system
2. Improving Ensemble Tracker performance
3. Adding multi-camera tracking capability
4. Optimizing memory usage for long videos
5. Developing tracking results visualization interface 