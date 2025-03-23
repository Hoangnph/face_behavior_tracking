from src.detection.base_detection import BaseDetector, Detection, BoundingBox

# Create a simple detector inheriting from BaseDetector
class TestDetector(BaseDetector):
    def detect(self, image):
        # Just return some detections
        return self._filter_detections([
            Detection(BoundingBox(x1=10, y1=10, x2=20, y2=20), 0.9, 'test'),
            Detection(BoundingBox(x1=30, y1=30, x2=40, y2=40), 0.3, 'test')
        ])

# Create detector with threshold
detector = TestDetector(confidence_threshold=0.5)
detections = detector.detect(None)

print(f'Number of detections after filtering: {len(detections)}')
for d in detections:
    print(f'  Detection with confidence {d.confidence}') 