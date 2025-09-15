import cv2
import numpy as np
import supervision as sv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="s56BKSrEhxiyRHyxAGD8"
)

client_config = InferenceConfiguration(confidence_threshold=0.3, iou_threshold=0.3)

label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()


def detect_hash_marks(img_path: str, output_img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)

    with CLIENT.use_configuration(client_config):
        hash_marks_result = CLIENT.infer(img_path, model_id="hash-yards-intersection/4")

    label_annotator.color = sv.ColorPalette.from_matplotlib('viridis', 5)
    bounding_box_annotator.color = sv.ColorPalette.from_matplotlib('viridis', 5)

    hash_mark_labels = [item["class"] for item in hash_marks_result["predictions"]]
    hash_mark_detections = sv.Detections.from_inference(hash_marks_result)

    xyxy = hash_mark_detections.xyxy  # shape: (N, 4)
    hash_mark_centers = np.column_stack([
        (xyxy[:, 0] + xyxy[:, 2]) / 2,  # center x
        (xyxy[:, 1] + xyxy[:, 3]) / 2  # center y
    ])

    hash_marks_image = bounding_box_annotator.annotate(
        scene=img, detections=hash_mark_detections)
    hash_marks_image = label_annotator.annotate(
        scene=hash_marks_image, detections=hash_mark_detections, labels=hash_mark_labels)

    cv2.imwrite(output_img_path, hash_marks_image)

    return hash_mark_centers


if __name__ == "__main__":
    img_path = "C:/Users\Justin.Getzke\AppData\Roaming\com.uoc.football-pre-play-analysis\extracted/2108 BOISE STATE OFF vs COLORADO STATE\snaps\play_005_snap2.jpg"
    output_img_path = "hash_marks.jpg"
    detections = detect_hash_marks(img_path)
    print(f"Output written to {output_img_path}")
