import cv2
import numpy as np
import supervision as sv
from roboflow import Roboflow
from supervision import Detections

rf = Roboflow(api_key="s56BKSrEhxiyRHyxAGD8")
label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()


def detect_players(img_path: str, output_img_path: str) -> Detections:
    img = cv2.imread(img_path)

    player_detection_project = rf.workspace().project("football-presnap-tracker")
    player_detection_model = player_detection_project.version(3).model

    player_detection_result = player_detection_model.predict(
        img_path,
        confidence=30,
        overlap=50

    ).json()

    player_detection_labels = [item["class"] for item in player_detection_result["predictions"]]
    player_detections = sv.Detections.from_inference(player_detection_result)

    # Your original detections and labels
    detections = player_detections
    labels = player_detection_labels
    confidences = detections.confidence
    class_names = detections.data['class_name']

    # Combine everything for sorting and indexing
    combined = list(zip(
        range(len(labels)),  # original index
        labels,
        confidences
    ))

    # Sort by confidence descending
    combined.sort(key=lambda x: x[2], reverse=True)

    # Counters and limits
    limits = {
        "defense": 11,
        "qb": 1,
        "oline": 10,  # Will combine this with "skill" and "qb"
        "skill": 10,
    }

    selected_indices = []
    oline_skill_qb_count = 0
    class_counts = {"defense": 0, "qb": 0, "oline": 0, "skill": 0}

    # Select according to constraints
    for idx, label, conf in combined:
        if label == "ref":  # Always skip refs
            selected_indices.append(idx)

        if label == "defense":
            selected_indices.append(idx)
            class_counts["defense"] += 1
        elif label in {"oline", "skill", "qb"}:
            selected_indices.append(idx)
            class_counts[label] += 1
            oline_skill_qb_count += 1

    # Now filter detections and labels
    filtered_detections = detections[np.array(selected_indices)]
    filtered_labels = [labels[i] for i in selected_indices]

    label_annotator.color = sv.ColorPalette.DEFAULT
    bounding_box_annotator.color = sv.ColorPalette.DEFAULT

    player_detection_image = img.copy()

    player_detection_image = bounding_box_annotator.annotate(
        scene=player_detection_image, detections=filtered_detections)
    player_detection_image = label_annotator.annotate(
        scene=player_detection_image, detections=filtered_detections, labels=filtered_labels)

    cv2.imwrite(output_img_path, player_detection_image)

    return filtered_detections


if __name__ == "__main__":
    img_path = "C:/Users\Justin.Getzke\AppData\Roaming\com.uoc.football-pre-play-analysis\extracted/2108 BOISE STATE OFF vs COLORADO STATE\snaps\play_005_snap2.jpg"
    output_img_path = "players.jpg"
    detections = detect_players(img_path, output_img_path)
    print(f"Output written to {output_img_path}")
    print(detections)