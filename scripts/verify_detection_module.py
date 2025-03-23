#!/usr/bin/env python
"""
Verification script for detection module.

This script runs comprehensive tests on the detection module to verify it meets all requirements
from the project specification. It tests against sample faces in data/faces and the sample_2.mp4
video in data/videos.

Usage:
    python scripts/verify_detection_module.py

The script will automatically:
1. Run all unit tests for the detection module
2. Test face detection on sample faces from data/faces
3. Test person detection on sample_2.mp4 from data/videos
4. Benchmark performance for all detection methods
5. Generate a verification report

A manual verification step is required at the end following the @big-project.mdc rule.
"""

import os
import sys
import time
import glob
import unittest
import numpy as np
import cv2
from datetime import datetime
from tqdm import tqdm

# Import detection components
from src.detection import (
    MediaPipeFaceDetector, FaceDetection,
    YOLOPersonDetector, PersonDetection,
    ONNXDetector, convert_to_onnx,
    DetectionScheduler
)
from src.video_input import FileSource, FrameProcessor, VideoPipeline


class DetectionVerifier:
    """Verification class for detection module."""
    
    def __init__(self):
        """Initialize verifier with paths and settings."""
        # Paths
        self.face_dirs = ["data/faces/known_customers", "data/faces/employees"]
        self.sample_video = "data/videos/sample_2.mp4"
        self.output_dir = "data/output/verification"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Report file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_path = os.path.join(self.output_dir, f"verification_report_{timestamp}.txt")
        self.report_file = None
        
        # Test summary
        self.tests_passed = 0
        self.tests_failed = 0
        self.unit_tests_passed = 0
        self.unit_tests_failed = 0
        self.face_tests_passed = 0
        self.face_tests_failed = 0
        self.person_tests_passed = 0
        self.person_tests_failed = 0
        self.performance_tests = {}
        
        # Initialize detectors
        self.face_detector = None
        self.person_detector = None
        self.onnx_detector = None
        self.scheduler = None
    
    def log(self, message, level="INFO"):
        """Log message to console and report file."""
        line = f"[{level}] {message}"
        print(line)
        if self.report_file:
            self.report_file.write(line + "\n")
    
    def start_verification(self):
        """Start the verification process."""
        self.report_file = open(self.report_path, "w")
        self.log("="*80)
        self.log("DETECTION MODULE VERIFICATION REPORT")
        self.log(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("="*80)
        self.log("")
        
        try:
            # Initialize detectors
            self._initialize_detectors()
            
            # Run tests
            self._run_unit_tests()
            self._verify_face_detection()
            self._verify_person_detection()
            self._benchmark_performance()
            
            # Summarize results
            self._summarize_results()
        except Exception as e:
            self.log(f"Verification failed with error: {str(e)}", "ERROR")
        finally:
            # Close report
            if self.report_file:
                self.report_file.close()
                print(f"\nVerification report saved to: {self.report_path}")
    
    def _initialize_detectors(self):
        """Initialize detection components."""
        self.log("Initializing detectors...")
        
        # Face detector
        self.face_detector = MediaPipeFaceDetector(min_detection_confidence=0.5)
        self.log("Face detector initialized")
        
        # Person detector
        self.person_detector = YOLOPersonDetector(confidence_threshold=0.5)
        self.log("Person detector initialized")
        
        # ONNX detector (if available)
        try:
            # Check if ONNX model exists
            onnx_model_path = "models/onnx/yolov8n.onnx"
            yolo_model_path = "models/yolo/yolov8n.pt"
            
            if not os.path.exists(onnx_model_path):
                self.log(f"ONNX model not found at {onnx_model_path}")
                self.log("Converting YOLOv8 model to ONNX format...")
                os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)
                convert_to_onnx(yolo_model_path, onnx_model_path)
                self.log(f"ONNX model saved to {onnx_model_path}")
            
            self.onnx_detector = ONNXDetector(
                model_path=onnx_model_path,
                confidence_threshold=0.5,
                class_mapping={0: "person"}
            )
            self.log("ONNX detector initialized")
        except ImportError:
            self.log("ONNX runtime not available, skipping ONNX tests", "WARNING")
        
        # Scheduler
        self.scheduler = DetectionScheduler(
            detectors={
                "face": self.face_detector,
                "person": self.person_detector
            },
            default_detector="face",
            frequencies={"face": 1, "person": 2}
        )
        self.log("Detection scheduler initialized")
    
    def _run_unit_tests(self):
        """Run unit tests for the detection module."""
        self.log("\n" + "="*40)
        self.log("RUNNING UNIT TESTS")
        self.log("="*40)
        
        # Create a test suite
        test_loader = unittest.TestLoader()
        test_suite = test_loader.discover('tests/detection', pattern='test_*.py')
        
        # Run tests and capture results
        test_runner = unittest.TextTestRunner(verbosity=2)
        test_result = test_runner.run(test_suite)
        
        # Update summary
        self.unit_tests_passed = test_result.testsRun - len(test_result.errors) - len(test_result.failures)
        self.unit_tests_failed = len(test_result.errors) + len(test_result.failures)
        self.tests_passed += self.unit_tests_passed
        self.tests_failed += self.unit_tests_failed
        
        # Log results
        self.log(f"\nUnit tests completed: {test_result.testsRun} tests")
        self.log(f"Passed: {self.unit_tests_passed}, Failed: {self.unit_tests_failed}")
        
        if self.unit_tests_failed > 0:
            self.log("\nFailing tests:", "WARNING")
            for failure in test_result.failures:
                self.log(f"- {failure[0]}", "WARNING")
            for error in test_result.errors:
                self.log(f"- {error[0]}", "WARNING")
    
    def _verify_face_detection(self):
        """Verify face detection on sample faces."""
        self.log("\n" + "="*40)
        self.log("VERIFYING FACE DETECTION")
        self.log("="*40)
        
        # Get all face images
        face_images = []
        for directory in self.face_dirs:
            if os.path.exists(directory):
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    face_images.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
        
        if not face_images:
            self.log("No face images found in the specified directories", "WARNING")
            return
        
        self.log(f"Found {len(face_images)} face images to process")
        
        # Sample a subset of images for verification
        import random
        random.seed(42)  # For reproducibility
        sample_size = min(50, len(face_images))
        sampled_images = random.sample(face_images, sample_size)
        
        # Process each face image
        success_count = 0
        face_count = 0
        
        self.log(f"Testing on {sample_size} sample images...")
        for image_path in tqdm(sampled_images, desc="Processing faces"):
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                self.log(f"Error: Cannot load image from {image_path}", "ERROR")
                self.face_tests_failed += 1
                continue
            
            # Detect faces
            faces = self.face_detector.detect(image)
            
            # Verify detection
            if len(faces) > 0:
                success_count += 1
                face_count += len(faces)
                
                # Save a few sample outputs
                if success_count <= 5:
                    output_path = os.path.join(
                        self.output_dir, 
                        f"face_detection_sample_{success_count}.jpg"
                    )
                    vis_image = self.face_detector.visualize(image, faces)
                    cv2.imwrite(output_path, vis_image)
            else:
                self.face_tests_failed += 1
        
        # Update summary
        self.face_tests_passed = success_count
        self.tests_passed += success_count
        self.tests_failed += (sample_size - success_count)
        
        # Log results
        detection_rate = success_count / sample_size * 100
        self.log(f"\nFace detection results:")
        self.log(f"- Images with faces detected: {success_count}/{sample_size} ({detection_rate:.1f}%)")
        self.log(f"- Total faces detected: {face_count}")
        
        # Validate results
        if detection_rate >= 90:
            self.log("Face detection test PASSED: Detection rate >= 90%", "SUCCESS")
        else:
            self.log(f"Face detection test FAILED: Detection rate {detection_rate:.1f}% < 90%", "FAILURE")
    
    def _verify_person_detection(self):
        """Verify person detection on sample video."""
        self.log("\n" + "="*40)
        self.log("VERIFYING PERSON DETECTION")
        self.log("="*40)
        
        if not os.path.exists(self.sample_video):
            self.log(f"Sample video not found at {self.sample_video}", "ERROR")
            return
        
        self.log(f"Testing person detection on {self.sample_video}")
        
        # Create video source and pipeline
        source = FileSource(self.sample_video)
        processor = FrameProcessor(resize_dims=(640, 480))
        
        # Test statistics
        frame_count = 0
        frames_with_persons = 0
        total_persons = 0
        sample_frames = []
        
        # Test YOLOv8 detector
        self.log("Testing YOLOv8 person detector...")
        with VideoPipeline(source, processor) as pipeline:
            # Process first 300 frames (10 seconds at 30 FPS)
            max_frames = 300
            
            for _ in tqdm(range(max_frames), desc="Processing video"):
                success, frame, metadata = pipeline.read()
                if not success:
                    break
                
                frame_count += 1
                
                # Process every 10th frame
                if frame_count % 10 == 0:
                    # Detect persons
                    persons = self.person_detector.detect(frame)
                    
                    if len(persons) > 0:
                        frames_with_persons += 1
                        total_persons += len(persons)
                        
                        # Save sample frames
                        if len(sample_frames) < 5:
                            vis_frame = self.person_detector.visualize(frame, persons)
                            sample_frames.append(("yolo", frame_count, vis_frame))
        
        # Calculate detection rate
        detection_rate = frames_with_persons / (frame_count // 10) * 100 if frame_count > 0 else 0
        avg_persons = total_persons / frames_with_persons if frames_with_persons > 0 else 0
        
        # Log results
        self.log(f"\nYOLOv8 person detection results:")
        self.log(f"- Processed {frame_count} frames, analyzed {frame_count // 10}")
        self.log(f"- Frames with persons: {frames_with_persons}")
        self.log(f"- Detection rate: {detection_rate:.1f}%")
        self.log(f"- Average persons per frame: {avg_persons:.1f}")
        
        # Save sample frames
        for detector_type, frame_num, frame in sample_frames:
            output_path = os.path.join(
                self.output_dir, 
                f"person_detection_{detector_type}_frame_{frame_num}.jpg"
            )
            cv2.imwrite(output_path, frame)
        
        # Update summary
        if detection_rate >= 70:
            self.log("Person detection test PASSED: Detection rate >= 70%", "SUCCESS")
            self.person_tests_passed += 1
            self.tests_passed += 1
        else:
            self.log(f"Person detection test FAILED: Detection rate {detection_rate:.1f}% < 70%", "FAILURE")
            self.person_tests_failed += 1
            self.tests_failed += 1
    
    def _benchmark_performance(self):
        """Benchmark detection performance."""
        self.log("\n" + "="*40)
        self.log("BENCHMARKING PERFORMANCE")
        self.log("="*40)
        
        if not os.path.exists(self.sample_video):
            self.log(f"Sample video not found at {self.sample_video}", "ERROR")
            return
        
        self.log(f"Running performance benchmarks on {self.sample_video}")
        
        # Test configurations
        test_configs = [
            ("Face Detection", lambda img: self.face_detector.detect(img)),
            ("Person Detection (YOLOv8)", lambda img: self.person_detector.detect(img)),
        ]
        
        # Add ONNX if available
        if self.onnx_detector:
            test_configs.append(
                ("Person Detection (ONNX)", lambda img: self.onnx_detector.detect(img))
            )
        
        # Add scheduler
        test_configs.append(
            ("Detection Scheduler", lambda img: self.scheduler.detect_all(img))
        )
        
        # Create video source
        source = FileSource(self.sample_video)
        processor = FrameProcessor(resize_dims=(640, 480))
        
        # Run benchmarks
        for test_name, detection_func in test_configs:
            self.log(f"\nBenchmarking: {test_name}")
            
            # Reset source
            source.close()
            source.open()
            
            # Create pipeline
            with VideoPipeline(source, processor) as pipeline:
                # Performance variables
                frame_count = 0
                processed_frames = 0
                total_time = 0
                
                # Warm up
                for _ in range(10):
                    success, frame, _ = pipeline.read()
                    if not success:
                        break
                    detection_func(frame)
                
                # Benchmark
                max_frames = 100
                for _ in tqdm(range(max_frames), desc=f"Testing {test_name}"):
                    success, frame, _ = pipeline.read()
                    if not success:
                        break
                    
                    frame_count += 1
                    
                    # Process every 3rd frame
                    if frame_count % 3 == 0:
                        processed_frames += 1
                        
                        # Time the detection
                        start_time = time.time()
                        detection_func(frame)
                        end_time = time.time()
                        
                        total_time += (end_time - start_time)
                
                # Calculate FPS
                if processed_frames > 0 and total_time > 0:
                    inference_fps = processed_frames / total_time
                    effective_fps = frame_count / (total_time * 3)  # Accounting for skipped frames
                    
                    self.log(f"- Processed {processed_frames} frames in {total_time:.2f} seconds")
                    self.log(f"- Inference FPS: {inference_fps:.2f}")
                    self.log(f"- Effective FPS with frame skipping: {effective_fps:.2f}")
                    
                    # Store in results
                    self.performance_tests[test_name] = {
                        "processed_frames": processed_frames,
                        "total_time": total_time,
                        "inference_fps": inference_fps,
                        "effective_fps": effective_fps
                    }
                else:
                    self.log("Not enough frames processed for benchmark", "WARNING")
    
    def _summarize_results(self):
        """Summarize verification results."""
        self.log("\n" + "="*80)
        self.log("VERIFICATION SUMMARY")
        self.log("="*80)
        
        # Overall results
        total_tests = self.tests_passed + self.tests_failed
        pass_rate = self.tests_passed / total_tests * 100 if total_tests > 0 else 0
        
        self.log(f"\nOverall Test Results:")
        self.log(f"- Total tests: {total_tests}")
        self.log(f"- Passed: {self.tests_passed} ({pass_rate:.1f}%)")
        self.log(f"- Failed: {self.tests_failed}")
        
        # Unit tests
        self.log(f"\nUnit Tests:")
        self.log(f"- Passed: {self.unit_tests_passed}")
        self.log(f"- Failed: {self.unit_tests_failed}")
        
        # Integration tests
        self.log(f"\nIntegration Tests:")
        self.log(f"- Face detection tests passed: {self.face_tests_passed}")
        self.log(f"- Person detection tests passed: {self.person_tests_passed}")
        
        # Performance
        self.log(f"\nPerformance Benchmarks:")
        for test_name, results in self.performance_tests.items():
            self.log(f"- {test_name}: {results['inference_fps']:.2f} FPS")
        
        # Overall status
        status = "PASSED" if pass_rate >= 90 else "FAILED"
        self.log(f"\nOVERALL VERIFICATION STATUS: {status}")
        
        # Manual verification prompt following @big-project.mdc rule
        self.log("\n" + "="*80)
        self.log("MANUAL VERIFICATION REQUIRED")
        self.log("="*80)
        self.log("\nAccording to @big-project.mdc rule, please manually review and confirm:")
        self.log("1. All tests have been successfully executed")
        self.log("2. The detection module correctly identifies faces and persons")
        self.log("3. Performance meets the project requirements")
        self.log("\nTo confirm verification, examine the output images in:")
        self.log(f"- {self.output_dir}")
        self.log("\nAnd run a manual test using:")
        self.log("python scripts/demo_detection.py --sample-video --display")


if __name__ == "__main__":
    verifier = DetectionVerifier()
    verifier.start_verification() 