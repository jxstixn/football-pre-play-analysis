import cv2
import numpy as np
import supervision as sv
from roboflow import Roboflow
from supervision import Detections

rf = Roboflow(api_key="s56BKSrEhxiyRHyxAGD8")
label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()


def detect_hash_marks(img_path: str, output_img_path: str) -> Detections:
    img = cv2.imread(img_path)

    hash_marks_project = rf.workspace().project("hash-yards-intersection")
    hash_marks_model = hash_marks_project.version(4).model

    hash_marks_result = hash_marks_model.predict(img_path, confidence=30, overlap=30).json()

    label_annotator.color = sv.ColorPalette.from_matplotlib('viridis', 5)
    bounding_box_annotator.color = sv.ColorPalette.from_matplotlib('viridis', 5)

    hash_marks_labels = [item["class"] for item in hash_marks_result["predictions"]]
    hash_marks_detections = sv.Detections.from_inference(hash_marks_result)

    hash_marks_image = img.copy()

    hash_marks_image = bounding_box_annotator.annotate(
        scene=img, detections=hash_marks_detections)
    hash_marks_image = label_annotator.annotate(
        scene=hash_marks_image, detections=hash_marks_detections, labels=hash_marks_labels)

    cv2.imwrite(output_img_path, hash_marks_image)

    return hash_marks_detections


def calculate_bounding_hash_marks(hash_marks_detections: Detections) -> tuple[np.ndarray, np.floating]:
    # Step 1: Get bounding boxes and calculate centers
    xyxy = hash_marks_detections.xyxy  # shape: (N, 4)
    hash_centers = np.column_stack([
        (xyxy[:, 0] + xyxy[:, 2]) / 2,  # center x
        (xyxy[:, 1] + xyxy[:, 3]) / 2  # center y
    ])

    # Step 2: Group by similar y (same row tolerance)
    y_tolerance = 30
    grouped_rows = {}

    for x, y in hash_centers:
        matched = False
        for key in grouped_rows:
            if abs(y - key) < y_tolerance:
                grouped_rows[key].append((x, y))
                matched = True
                break
        if not matched:
            grouped_rows[y] = [(x, y)]

    # Step 3: Pick the row with most hash marks
    best_row_y = max(grouped_rows, key=lambda k: len(grouped_rows[k]))
    row_points = grouped_rows[best_row_y]

    # Step 4: Sort by x (still useful for sequence), but useful (x, y) in distance
    row_points_sorted = sorted(row_points, key=lambda pt: pt[0])

    # Compute Euclidean distances between consecutive points
    pixel_distances = [
        np.linalg.norm(np.array(row_points_sorted[i + 1]) - np.array(row_points_sorted[i]))
        for i in range(len(row_points_sorted) - 1)
    ]
    avg_pixel_distance = np.mean(pixel_distances)
    pixel_per_yard = avg_pixel_distance / 5

    print(f"Average pixel distance between hash marks: {avg_pixel_distance:.2f} px")
    print(f"Estimated pixel per yard: {pixel_per_yard:.2f} px/yard")

    return hash_centers, avg_pixel_distance


if __name__ == "__main__":
    img_path = "C:/Users\Justin.Getzke\AppData\Roaming\com.uoc.football-pre-play-analysis\extracted/2108 BOISE STATE OFF vs COLORADO STATE\snaps\play_005_snap2.jpg"
    output_img_path = "hash_marks.jpg"
    detections = detect_hash_marks(img_path, output_img_path)
    print(f"Output written to {output_img_path}")

    hash_centers = calculate_bounding_hash_marks(detections)
