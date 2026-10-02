import os
import cv2
from ultralytics import YOLO

class YOLOv8Detector:
    def __init__(self, model_path='models/yolov8n.pt', conf=0.3):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect_image(self, image_path, save_path=None, visualize=False):
        results = self.model(image_path, conf=self.conf)
        result = results[0]
        img_with_boxes = result.plot()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img_with_boxes)

        if visualize:
            import matplotlib.pyplot as plt
            plt.imshow(img_with_boxes[:, :, ::-1])
            plt.title("Detection Result")
            plt.axis('off')
            plt.show()

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names

        detections = []
        for i in range(len(boxes)):
            detections.append({
                "bbox": boxes[i],
                "confidence": scores[i],
                "class_id": classes[i],
                "class_name": names[classes[i]]
            })

        return detections, img_with_boxes

    def detect_frame(self, frame):
        results = self.model.predict(source=frame, conf=self.conf)
        result = results[0]
        img_with_boxes = result.plot()

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names

        detections = []
        for i in range(len(boxes)):
            detections.append({
                "bbox": boxes[i],
                "confidence": scores[i],
                "class_id": classes[i],
                "class_name": names[classes[i]]
            })

        return detections, img_with_boxes



