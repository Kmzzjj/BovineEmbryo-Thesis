"""Run the YOLO detector on a single image or video frame."""

from pathlib import Path

import cv2
from ultralytics import YOLO

from .paths import PRETRAINED


class YOLOv8Detector:
    def __init__(self, model_path=str(PRETRAINED), conf=0.3):
        self.model = YOLO(str(model_path))
        self.conf = conf

    @staticmethod
    def _detections(result):
        boxes = result.boxes
        return [
            {
                "bbox": box,
                "confidence": score,
                "class_id": class_id,
                "class_name": result.names[class_id],
            }
            for box, score, class_id in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.conf.cpu().numpy(),
                boxes.cls.cpu().numpy().astype(int),
            )
        ]

    def detect_image(self, image_path, save_path=None, visualize=False):
        result = self.model(image_path, conf=self.conf)[0]
        annotated = result.plot()

        if save_path:
            output = Path(save_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), annotated):
                raise OSError(f"Could not write {output}")

        if visualize:
            import matplotlib.pyplot as plt

            plt.imshow(annotated[:, :, ::-1])
            plt.title("Detection Result")
            plt.axis("off")
            plt.show()

        return self._detections(result), annotated

    def detect_frame(self, frame):
        result = self.model.predict(source=frame, conf=self.conf)[0]
        return self._detections(result), result.plot()
