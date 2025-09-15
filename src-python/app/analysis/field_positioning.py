import os
import re
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from app.utils import get_resource_path

logger = logging.getLogger("FieldPositioning")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class FootballYardDetector:
    def __init__(self, field_image_path, templates_dir):
        """
        Initialize the detector with field image and templates directory

        Args:
            field_image_path: Path to the football field image
            templates_dir: Directory containing template images of numbers
        """
        self.field_image = cv2.imread(field_image_path)
        self.field_gray = cv2.cvtColor(self.field_image, cv2.COLOR_BGR2GRAY)
        self.templates_dir = templates_dir
        self.templates = self._load_templates()
        self.detections = []

    def _load_templates(self):
        """Load all template images from the templates directory"""
        templates = {}
        template_files = Path(self.templates_dir).glob("*.png") if Path(self.templates_dir).exists() else []
        template_files = list(template_files) + list(Path(self.templates_dir).glob("*.jpg"))

        for template_path in template_files:
            number = template_path.stem
            if not number:
                continue
            template = cv2.imread(str(template_path), 0)  # Load as grayscale
            if template is not None:
                templates[number] = template
                logger.debug(f"Loaded template for number: {number} from {template_path}")
            else:
                logger.warning(f"Failed to load template image: {template_path}")

        return templates

    def detect_yard_numbers(
            self,
            threshold=0.9,
            scale_range=(0.5, 1.5),
            scale_steps=10,
            roi_boxes=None  # list of (x1, y1, x2, y2)
    ):
        """
        Detect yard line numbers using template matching with multiple scales,
        optionally limited to given ROIs.

        roi_boxes: list of (x1, y1, x2, y2) in pixel coords on the full image.
                   If None, uses the full image as a single ROI.
        """
        self.detections = []

        # Default ROI = full image
        if not roi_boxes:
            H, W = self.field_gray.shape[:2]
            roi_boxes = [(0, 0, W, H)]

        for number, template in self.templates.items():
            logger.debug(f"Detecting number: {number}")
            scales = np.linspace(scale_range[0], scale_range[1], scale_steps)

            for scale in scales:
                new_width = int(template.shape[1] * scale)
                new_height = int(template.shape[0] * scale)
                if new_width < 10 or new_height < 10:
                    continue

                scaled_template = cv2.resize(template, (new_width, new_height))

                for (x1, y1, x2, y2) in roi_boxes:
                    # Crop ROI
                    roi = self.field_gray[y1:y2, x1:x2]
                    if roi.shape[0] < new_height or roi.shape[1] < new_width:
                        continue  # template bigger than ROI

                    # Match within ROI
                    result = cv2.matchTemplate(roi, scaled_template, cv2.TM_CCOEFF_NORMED)
                    locations = np.where(result >= threshold)

                    for pt in zip(*locations[::-1]):
                        confidence = result[pt[1], pt[0]]

                        detection = {
                            'number': number,
                            'x': pt[0] + x1,  # offset back to full image
                            'y': pt[1] + y1,
                            'width': new_width,
                            'height': new_height,
                            'confidence': confidence,
                            'scale': scale,
                            'roi': (x1, y1, x2, y2)
                        }
                        self.detections.append(detection)

        self.detections = self._non_max_suppression(self.detections)
        logger.info(f"Total detections after NMS: {len(self.detections)}")
        return self.detections

    def _non_max_suppression(self, detections, overlap_threshold=0.3):
        """Remove overlapping detections using Non-Maximum Suppression"""
        if len(detections) == 0:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        keep = []

        for i, detection in enumerate(detections):
            overlap = False

            for kept_detection in keep:
                # Calculate overlap
                x1 = max(detection['x'], kept_detection['x'])
                y1 = max(detection['y'], kept_detection['y'])
                x2 = min(detection['x'] + detection['width'], kept_detection['x'] + kept_detection['width'])
                y2 = min(detection['y'] + detection['height'], kept_detection['y'] + kept_detection['height'])

                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    area1 = detection['width'] * detection['height']
                    area2 = kept_detection['width'] * kept_detection['height']
                    union = area1 + area2 - intersection

                    if intersection / union > overlap_threshold:
                        overlap = True
                        break

            if not overlap:
                keep.append(detection)

        return keep

    def visualize_detections(self, save_path=None, figsize=(15, 10), roi_boxes=None):
        """Visualize the detected yard numbers on the field"""
        result_image = self.field_image.copy()
        # Draw ROIs (optional)
        if roi_boxes:
            for (x1, y1, x2, y2) in roi_boxes:
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw bounding boxes and labels
        for detection in self.detections:
            x, y = detection['x'], detection['y']
            w, h = detection['width'], detection['height']
            number = detection['number']
            confidence = detection['confidence']

            # Draw rectangle
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add label
            label = f"{number} ({confidence:.2f})"
            cv2.putText(result_image, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display using matplotlib
        plt.figure(figsize=figsize)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Detected Yard Numbers ({len(self.detections)} found)')
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        plt.show()

        return result_image

    def get_detections_dataframe(self):
        """Convert detections to a pandas DataFrame for analysis"""
        try:
            import pandas as pd
            df = pd.DataFrame(self.detections)
            if not df.empty:
                df = df.sort_values(['y', 'x'])  # Sort by position
            return df
        except ImportError:
            print("pandas not available. Returning list of detections.")
            return self.detections

    def save_detection_coordinates(self, output_file='yard_detections.txt'):
        """Save detection coordinates to a text file"""
        with open(output_file, 'w') as f:
            f.write("Number\tX\tY\tWidth\tHeight\tConfidence\tScale\n")
            for det in sorted(self.detections, key=lambda x: (x['y'], x['x'])):
                f.write(
                    f"{det['number']}\t{det['x']}\t{det['y']}\t{det['width']}\t{det['height']}\t{det['confidence']:.3f}\t{det['scale']:.2f}\n")
        print(f"Detections saved to {output_file}")

    def make_vertical_rois(self, y_ranges, x_range=None):
        """
        y_ranges: list of (y1, y2)
        x_range: (x1, x2) or None for full width
        """
        H, W = self.field_gray.shape[:2]
        if x_range is None:
            x_range = (0, W)
        x1, x2 = x_range
        rois = []
        for (yy1, yy2) in y_ranges:
            yy1c = max(0, min(H, yy1))
            yy2c = max(0, min(H, yy2))
            if yy2c > yy1c:
                rois.append((x1, yy1c, x2, yy2c))
        return rois


def calculate_los_position(classified_lines, x_los):
    """
    Compute LOS yards using detected labels and 10px = 1 feet scale.

    Rules:
    - Detected label "2" means 20 yards, "3" means 30, etc.
    - Field is mirrored at 50; numbers increase toward midfield (50) from both ends.
    - Determine left-to-right direction from label ordering. If larger labels are to the right,
      numbers increase with x; otherwise they decrease with x.
    - Use the nearest labeled line as the base and offset by delta_x/10 yards.
    """
    if not classified_lines:
        logger.warning("No classified yard lines provided.")
        return None

    # Deduplicate by x position, prefer the first occurrence
    by_x = {}
    for x, n in classified_lines:
        if x not in by_x:
            try:
                val = int(n) * 10
            except Exception:
                continue
            by_x[x] = max(10, min(50, val))

    if not by_x:
        logger.warning("No valid numeric labels in classified lines.")
        return None

    labels = sorted(by_x.items(), key=lambda t: t[0])  # (x, yard_value)

    # Infer direction: +1 if numbers increase with x, -1 otherwise
    direction = 1
    for i in range(len(labels) - 1):
        x1, y1 = labels[i]
        x2, y2 = labels[i + 1]
        if x2 == x1:
            continue
        if y2 != y1:
            direction = 1 if (y2 - y1) * (x2 - x1) > 0 else -1
            break

    # Nearest labeled line to LOS
    nearest_x, nearest_yards = min(labels, key=lambda t: abs(t[0] - x_los))
    # Convert pixel distance to yards (10 px = 1 feet)
    delta_px = x_los - nearest_x
    delta_feet = int(round(delta_px / 10.0))
    delta_yards = delta_feet // 3  # 3 feet = 1 yard

    raw_yards = nearest_yards + direction * delta_yards

    # Mirror around 50 to get distance to the nearest goal line
    # Map to [0,100), then fold to [0,50]
    t = raw_yards % 100
    los_yard = t if t <= 50 else 100 - t

    # Clamp to 1..50 as an int (avoid returning 0 unless exactly goal line desired)
    los_yard = int(max(1, min(50, los_yard)))

    logger.info(f"Calculated LOS position: {los_yard} yards (direction={'left' if direction == 1 else 'right'})")
    return los_yard, direction


def classify_yard_lines_and_los(
        img_path: str,
        output_img_path: str,
        x_los: int,
        threshold: float = 0.9,
        numbers_dir: str = None
):
    """
    Classify yard lines based on their x-coordinates relative to the line of scrimmage (x_los)
    and assign yard numbers accordingly.

    Args:
        img_path: Path to the input image
        output_img_path: Path to save the output image with classified yard lines
        x_los: x-coordinate of the line of scrimmage
        threshold: Template matching threshold (0.0 to 1.0)
        numbers_dir: Directory containing template images of yard numbers. If None, uses default resource path.
    """
    img = cv2.imread(img_path)

    H, W = img.shape[:2]

    if not numbers_dir:
        numbers_dir = get_resource_path("resources/field_numbers")
        if not os.path.exists(numbers_dir):
            logger.error(f"Field numbers directory not found at {numbers_dir}")
            raise FileNotFoundError(f"Field numbers directory not found at {numbers_dir}")
        else:
            logger.info(f"Field numbers directory found at {numbers_dir}")

    detector = FootballYardDetector(img_path, numbers_dir)
    rois = detector.make_vertical_rois([(100, 400), (1200, 1500)])
    yard_numbers = detector.detect_yard_numbers(threshold=threshold, scale_range=(0.5, 1.5), scale_steps=10, roi_boxes=rois)

    if not yard_numbers:
        logger.warning("No yard numbers detected.")
        return []

    logger.info(f"Detections: {yard_numbers}")

    # Parse detections: template schema LOCATION_NUMBER[_INV]
    # Use LEFT bbox x for _INV, RIGHT bbox x for normal
    parsed_detections = []
    for det in yard_numbers:
        name = det['number'] or ""
        is_inv = name.endswith('_INV')
        digits = re.findall(r"(\d+)", name)
        if not digits:
            continue
        num_val = int(digits[-1])
        if is_inv:
            x_ref = det['x']  # left side
        else:
            x_ref = det['x'] + det['width']  # right side
        parsed_detections.append({
            'x_ref': x_ref,
            'num': num_val,
            'inv': is_inv,
            'conf': det.get('confidence', 0.0)
        })

    if not parsed_detections:
        logger.warning("No valid yard number detections after parsing.")
        cv2.imwrite(output_img_path, img)
        return []

    # yard lines are spaced in a 150px interval so create a list of x positions
    yard_lines = [x for x in range(0, W, 150)]

    # 1) Associate detections to nearest yard line using a small x-threshold
    threshold_px = 25  # small x-variance threshold to accept association

    # Build list of candidate assignments (per orientation) with distances
    candidates = []
    for idx_det, det in enumerate(parsed_detections):
        # nearest yard line
        nearest_idx = min(range(len(yard_lines)),
                          key=lambda j: abs(yard_lines[j] - det['x_ref'])) if yard_lines else None
        if nearest_idx is None:
            continue
        dist = abs(yard_lines[nearest_idx] - det['x_ref'])
        if dist <= threshold_px:
            candidates.append({
                'det_index': idx_det,
                'line_index': nearest_idx,
                'dist': dist,
                'inv': det['inv'],
                'num': det['num']
            })

    # Sort candidates by proximity ascending
    candidates.sort(key=lambda c: c['dist'])

    # Prepare assignments per line: at most one normal and one inv each
    assignments = [{'normal': None, 'inv': None} for _ in yard_lines]
    det_used = [False] * len(parsed_detections)

    for c in candidates:
        if det_used[c['det_index']]:
            continue
        slot = 'inv' if c['inv'] else 'normal'
        if assignments[c['line_index']][slot] is None:
            assignments[c['line_index']][slot] = c
            det_used[c['det_index']] = True

    # 2) Enforce one unlabeled yard line between labeled ones
    has_any = [(a['normal'] is not None) or (a['inv'] is not None) for a in assignments]
    # distance score per line for tie-breaking: min distance of available labels
    line_score = []
    for a in assignments:
        dists = []
        if a['normal'] is not None:
            dists.append(a['normal']['dist'])
        if a['inv'] is not None:
            dists.append(a['inv']['dist'])
        line_score.append(min(dists) if dists else float('inf'))

    selected = [False] * len(assignments)
    i = 0
    while i < len(assignments):
        if not has_any[i]:
            i += 1
            continue
        # If next line also labeled, keep the closer one
        if i + 1 < len(assignments) and has_any[i + 1]:
            keep_i = i if line_score[i] <= line_score[i + 1] else i + 1
            selected[keep_i] = True
            # Ensure one unlabeled line between labels
            i = keep_i + 2
        else:
            selected[i] = True
            i += 2

    # 3) Render only selected lines and build result list
    classified_lines = []
    for idx_line, sel in enumerate(selected):
        if not sel:
            continue
        x = yard_lines[idx_line]
        a = assignments[idx_line]
        if a['normal'] is not None:
            num = a['normal']['num']
            classified_lines.append((x, num))
            cv2.putText(img, f"{num}", (x - 20, H - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if a['inv'] is not None:
            num = a['inv']['num']
            classified_lines.append((x, num))
            cv2.putText(img, f"{num}", (x - 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    los_yard, direction = calculate_los_position(classified_lines, x_los)

    if los_yard is not None:
        cv2.putText(img, f"LOS: {los_yard} yards", (W // 2 - 100, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0),
                    3)

    cv2.imwrite(output_img_path, img)
    logger.info(f"Classified yard lines saved to {output_img_path}")
    return classified_lines, los_yard, direction