import cv2
from ultralytics import YOLO
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class PersonBox:
    """
    Represents a detected person bounding box.

    Attributes:
        x1 (int): Left coordinate of the bounding box.
        y1 (int): Top coordinate of the bounding box.
        x2 (int): Right coordinate of the bounding box.
        y2 (int): Bottom coordinate of the bounding box.
        confidence (float): Detection confidence score.
    """
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def coordinates(self) -> list[int]:
        """
        Returns bounding box coordinates as a list

        Returns:
            list[int]: Bounding box coordinates as a list
        """
        return [self.x1, self.y1, self.x2, self.y2]

def load_model() -> YOLO:
    """
    Loads YOLO detection model.

    Returns:
        YOLO: YOLO detection model
    """
    model = YOLO("yolo11n.pt")
    return model


def process_frame(frame: np.ndarray, model: YOLO) -> list[PersonBox]:
    """
    Runs inference on a single frame and returns detected people boxes.

    Args:
        frame (np.ndarray): Frame to run inference on
        model (YOLO): YOLO detection model

    Returns:
        people_boxes (list[PersonBox]): People boxes
    """

    people_boxes = []
    results = model.predict(frame, imgsz=(1088, 1920), iou=0.8, conf=0.35, classes=[0], verbose=False)
    result = results[0]
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        people_boxes.append(PersonBox(x1, y1, x2, y2, conf))
    return people_boxes


def draw_boxes(frame: np.ndarray, boxes: list[PersonBox]) -> np.ndarray:
    """
    Draws bounding boxes and labels on the frame

    Args:
        frame (np.ndarray): Frame to draw on
        boxes (list[PersonBox]): People boxes

    Returns:
        frame (np.ndarray): Frame with bounding boxes and labels
    """

    for box in boxes:
        x1, y1, x2, y2 = box.coordinates
        text = f"person, {box.confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 200, 0), -1)

        cv2.putText(frame, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return frame


def main() -> None:
    """Main entry point: read video, run detection, save output."""
    input_path = "crowd.mp4"
    output_path = "result.mp4"

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {input_path}")

    model = load_model()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        cap.get(cv2.CAP_PROP_FPS),
        (int(cap.get(3)), int(cap.get(4)))
    )

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(frame_count), desc="Processing frames"):
        ret, frame = cap.read()

        if not ret:
            break

        people_boxes = process_frame(frame, model)
        frame = draw_boxes(frame, people_boxes)
        out.write(frame)

    cap.release()
    out.release()


if __name__ == "__main__":
    main()
