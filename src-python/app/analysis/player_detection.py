import cv2
import numpy as np
import supervision as sv
from supervision import Detections
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="s56BKSrEhxiyRHyxAGD8"
)

client_config = InferenceConfiguration(confidence_threshold=0.4, iou_threshold=0.5)

label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()


def calculate_los(image, yard_lines, player_detections):
    H, _ = image.shape[:2]

    # Collect foot positions by class
    oline_positions = []
    defense_positions = []
    for det in player_detections:
        bbox = det[0]
        cls = det[5]['class_name']
        x1, y1, x2, y2 = bbox
        foot_x = int((x1 + x2) / 2)
        foot_y = int(y2)
        if cls == 'oline':
            oline_positions.append((foot_x, foot_y))
        elif cls == 'defense':
            defense_positions.append((foot_x, foot_y))

    if not oline_positions or not defense_positions:
        return None

    # Find closest oline-defense pair
    best_pair = None
    best_dist_sq = float('inf')
    for ox, oy in oline_positions:
        for dx, dy in defense_positions:
            d2 = (ox - dx) ** 2 + (oy - dy) ** 2
            if d2 < best_dist_sq:
                best_dist_sq = d2
                best_pair = ((ox, oy), (dx, dy))

    (ox, oy), (dx, dy) = best_pair
    mid_x = (ox + dx) / 2.0
    mid_y = (oy + dy) / 2.0

    # Choose yard line from yard_lines whose x at y=mid_y is closest to mid_x
    if yard_lines and len(yard_lines) > 0:
        best_abs = float('inf')
        best_inv_slope = 0.0
        for (x_top_i, x_bot_i) in yard_lines:
            inv_slope_i = (x_bot_i - x_top_i) / float(H)
            x_at_mid = x_top_i + inv_slope_i * mid_y
            diff = abs(mid_x - x_at_mid)
            if diff < best_abs:
                best_abs = diff
                best_inv_slope = inv_slope_i

        # Shift yard line to pass through midpoint
        x_top_new = mid_x - best_inv_slope * mid_y
        x_bot_new = mid_x + best_inv_slope * (H - mid_y)

        p1 = (int(round(x_top_new)), 0)
        p2 = (int(round(x_bot_new)), H)
    else:
        # Fallback: vertical through midpoint
        p1 = (int(round(mid_x)), 0)
        p2 = (int(round(mid_x)), H)

    return {
        'midpoint': (mid_x, mid_y),
        'pair': ((ox, oy), (dx, dy)),
        'p1': p1,
        'p2': p2,
    }


def detect_players(img_path: str, output_img_path: str, yard_lines) -> Detections:
    player_detection_image = cv2.imread(img_path)

    with CLIENT.use_configuration(client_config):
        player_detection_result = CLIENT.infer(img_path, model_id="football-presnap-tracker/6")

    player_detection_labels = [item["class"] for item in player_detection_result["predictions"]]
    player_detections = sv.Detections.from_inference(player_detection_result)

    # Your original detections and labels
    detections = player_detections
    labels = player_detection_labels
    confidences = detections.confidence

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
            continue

        if label == "defense":
            if class_counts["defense"] < limits["defense"]:
                selected_indices.append(idx)
                class_counts["defense"] += 1
        elif label in {"oline", "skill", "qb"}:
            if oline_skill_qb_count < 11 and class_counts[label] < limits[label]:
                selected_indices.append(idx)
                class_counts[label] += 1
                oline_skill_qb_count += 1

    # Now filter detections and labels
    filtered_detections = detections[np.array(selected_indices)]
    filtered_labels = [labels[i] for i in selected_indices]

    label_annotator.color = sv.ColorPalette.DEFAULT
    bounding_box_annotator.color = sv.ColorPalette.DEFAULT

    player_detection_image = bounding_box_annotator.annotate(
        scene=player_detection_image, detections=filtered_detections)
    player_detection_image = label_annotator.annotate(
        scene=player_detection_image, detections=filtered_detections, labels=filtered_labels)

    los = calculate_los(player_detection_image, yard_lines, filtered_detections)
    cv2.line(player_detection_image, los['p1'], los['p2'], (0, 255, 255), 3)
    cv2.imwrite(output_img_path, player_detection_image)

    return filtered_detections