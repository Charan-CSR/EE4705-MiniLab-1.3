"""Task 4: YOLO detection and HSV color grounding.
# TODO: consume only the Task 2 onboard camera frames.

Contributor: Marie Chansuphan A0287345U
AI assistance: drafted with ChatGPT; reviewed and tested by the contributor.
Input contract: uint8 RGB image from the Task 2 front camera.
This module performs perception only; it does not control the robot.
"""
from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    class_name: str
    color: str
    confidence: float
    bbox: tuple[int, int, int, int]
    color_fractions: dict[str, float]


def classify_color(rgb_crop):
    """Initial thresholds validated on one frame, not across all poses.

    Fractions include background pixels inside the box. Color confidence is
    separate from YOLO class confidence. Unknown is allowed instead of guessing.
    OpenCV's uint8 hue range is 0..179.
    """
    h, s, v = cv2.split(cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2HSV))
    saturated = (s >= 80) & (v >= 50)
    masks = {
        'red': ((h <= 10) | (h >= 170)) & saturated,
        'green': (h >= 35) & (h <= 85) & saturated,
        'blue': (h >= 90) & (h <= 135) & saturated,
        'white': (s < 60) & (v >= 160),
    }
    fractions = {name: float(mask.mean()) for name, mask in masks.items()}
    color = max(fractions, key=fractions.get)
    if fractions[color] < 0.20:
        color = 'unknown'
    return color, fractions


class ObjectDetector:
    """Load once, reuse in a single perception worker (not the physics loop)."""

    def __init__(self, model_path='yolov8s.pt', confidence=0.25):
        self.model = YOLO(model_path)
        self.confidence = confidence
        # GPU if present, so CPU load cannot stall YOLO
        import torch
        self.device = 0 if torch.cuda.is_available() else 'cpu'

    def detect(self, rgb_frame):
        if (not isinstance(rgb_frame, np.ndarray) or rgb_frame.dtype != np.uint8
                or rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3
                or rgb_frame.size == 0):
            raise ValueError('Expected a nonempty H x W x 3 uint8 RGB frame')
        # Ultralytics expects BGR for numpy image inputs; MuJoCo supplies RGB.
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        result = self.model.predict(
            bgr, device=self.device, imgsz=960, conf=self.confidence,
            classes=[41, 45], verbose=False,
        )[0]
        height, width = rgb_frame.shape[:2]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            color, fractions = classify_color(rgb_frame[y1:y2, x1:x2])
            detections.append(Detection(
                class_name=result.names[int(box.cls.item())],
                color=color,
                confidence=float(box.conf.item()),
                bbox=(x1, y1, x2, y2),
                color_fractions=fractions,
            ))
        return detections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--model', default='yolov8s.pt')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    bgr = cv2.imread(str(args.image))
    if bgr is None:
        parser.error(f'Cannot read image: {args.image}')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    detections = ObjectDetector(args.model).detect(rgb)
    for d in detections:
        print(f'[DETECT] class={d.class_name} color={d.color} '
              f'conf={d.confidence:.2f} bbox={list(d.bbox)}')
    if not detections:
        print('[PERCEPTION] no cup or bowl detections')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps([asdict(d) for d in detections], indent=2) + '\n',
            encoding='utf-8',
        )
        print(f'[PERCEPTION] saved={args.output}')


if __name__ == '__main__':
    main()
"""Task 4 YOLO detection and colour grounding."""


