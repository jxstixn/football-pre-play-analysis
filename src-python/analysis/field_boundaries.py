import cv2
import math
import numpy as np


def match_with_yard_lines(yard_lines, hash_centers, tolerance):
    """Match hash marks with detected yard lines"""
    matched_pairs = []
    single_matches = []
    all_distances = []

    for x1, y1, x2, y2 in yard_lines:
        matched = []

        # Handle vertical lines
        if abs(x2 - x1) < 1e-3:
            line_x = x1
            for xh, yh in hash_centers:
                if abs(xh - line_x) < tolerance:
                    matched.append((xh, yh))
        else:
            # Non-vertical case: find intersections
            inv_slope = (x2 - x1) / (y2 - y1)
            for xh, yh in hash_centers:
                expected_x = x1 + (yh - y1) * inv_slope
                if abs(xh - expected_x) < tolerance:
                    matched.append((xh, yh))

        if len(matched) >= 2:
            # Use top-most and bottom-most hash marks
            matched_sorted = sorted(matched, key=lambda pt: pt[1])
            top_pt = matched_sorted[0]
            bottom_pt = matched_sorted[-1]

            dist = np.linalg.norm(np.array(top_pt) - np.array(bottom_pt))
            all_distances.append(dist)
            matched_pairs.append((top_pt, bottom_pt))

        elif len(matched) == 1:
            # Store the actual hash mark position (not yard line center)
            single_matches.append((matched[0], x1, y1, x2, y2))

    return matched_pairs, single_matches, all_distances


def conservative_hash_clustering(hash_centers, height, max_horizontal_distance):
    """Very conservative hash mark pairing - only pair obvious matches"""
    estimated_pairs = []

    # Separate into upper and lower regions with a buffer zone
    upper_hash = [(x, y) for x, y in hash_centers if y < height * 0.45]
    lower_hash = [(x, y) for x, y in hash_centers if y > height * 0.55]

    # Only proceed if we have hash marks in both regions
    if not upper_hash or not lower_hash:
        return estimated_pairs

    # For each upper hash mark, find the closest lower hash mark
    for x_upper, y_upper in upper_hash:
        best_match = None
        best_distance = float('inf')

        for x_lower, y_lower in lower_hash:
            # Check horizontal distance
            horizontal_dist = abs(x_upper - x_lower)
            if horizontal_dist <= max_horizontal_distance:
                # Calculate total distance (but prioritize horizontal alignment)
                total_dist = horizontal_dist + abs(y_upper - y_lower) * 0.1
                if total_dist < best_distance:
                    best_distance = total_dist
                    best_match = (x_lower, y_lower)

        if best_match:
            estimated_pairs.append(((x_upper, y_upper), best_match))

    # Remove duplicate pairings (if a lower hash mark matches multiple upper ones)
    unique_pairs = []
    used_lower = set()

    # Sort by match quality (horizontal distance)
    estimated_pairs.sort(key=lambda pair: abs(pair[0][0] - pair[1][0]))

    for upper_pt, lower_pt in estimated_pairs:
        if lower_pt not in used_lower:
            unique_pairs.append((upper_pt, lower_pt))
            used_lower.add(lower_pt)

    return unique_pairs


def estimate_missing_hash_marks(single_matches, mean_dist, height, used_hash_marks, max_horizontal_distance):
    """Conservative estimation of missing hash marks"""
    estimated_pairs = []

    for hash_point, x1, y1, x2, y2 in single_matches:
        xh, yh = hash_point

        # Skip if this hash mark is already used
        if hash_point in used_hash_marks:
            continue

        is_top = yh < height // 2
        direction = 1 if is_top else -1

        # Calculate the x-position on the yard line at the estimated y
        estimated_y = int(yh + direction * mean_dist)

        # Make sure estimated point is within image bounds
        if estimated_y < 0 or estimated_y >= height:
            continue

        if abs(x2 - x1) < 1e-3:
            # Vertical line - use the hash mark's x position
            estimated_x = xh
        else:
            # Non-vertical line - calculate intersection
            inv_slope = (x2 - x1) / (y2 - y1)
            estimated_x = x1 + (estimated_y - y1) * inv_slope

        estimated_point = (int(estimated_x), estimated_y)

        # Check if estimated point is reasonable (not too far horizontally)
        if abs(estimated_x - xh) > max_horizontal_distance:
            continue

        # Mark both points as used
        used_hash_marks.add(hash_point)
        used_hash_marks.add(estimated_point)

        if is_top:
            estimated_pairs.append((hash_point, estimated_point))
        else:
            estimated_pairs.append((estimated_point, hash_point))

    return estimated_pairs


def estimate_hash_mark_intersections(yard_lines, hash_centers, image, tolerance=15, max_horizontal_distance=50):
    """
    Conservative hash mark intersection estimation that prioritizes accuracy over completeness
    """
    height, width = image.shape[:2]

    all_distances = []
    matched_pairs = []
    estimated_pairs = []
    used_hash_marks = set()  # Track which hash marks are already used

    # Method 1: Match with detected yard lines (PRIMARY METHOD)
    if yard_lines:
        matched_pairs, single_matches, all_distances = match_with_yard_lines(
            yard_lines, hash_centers, tolerance
        )

        # Mark used hash marks
        for top_pt, bottom_pt in matched_pairs:
            used_hash_marks.add(top_pt)
            used_hash_marks.add(bottom_pt)

        # Handle single matches ONLY if we have good reference distances
        if all_distances and single_matches and len(all_distances) >= 2:
            mean_dist = np.mean(all_distances)
            # Only estimate if the standard deviation is reasonable (consistent distances)
            std_dist = np.std(all_distances)
            if std_dist < mean_dist * 0.3:  # Less than 30% variation
                estimated_pairs = estimate_missing_hash_marks(
                    single_matches, mean_dist, height, used_hash_marks, max_horizontal_distance
                )

    # Method 2: CONSERVATIVE fallback - only for very clear cases
    if not matched_pairs and not estimated_pairs:
        estimated_pairs = conservative_hash_clustering(
            hash_centers, height, max_horizontal_distance
        )

    return matched_pairs, estimated_pairs, all_distances


def process_hash_marks(masters, hash_centers, image, tolerance=15, max_horizontal_distance=40):
    """Main function to process hash marks with conservative estimation"""

    H, W = image.shape[:2]

    # Convert masters to yard line segments
    master_segments = [
        (int(round(x_top)), 0, int(round(x_bot)), H)
        for x_top, x_bot in masters
    ] if masters else []

    # Run conservative estimation
    matched_pairs, estimated_pairs, distances = estimate_hash_mark_intersections(
        yard_lines=master_segments,
        hash_centers=hash_centers,
        image=image,
        tolerance=tolerance,
        max_horizontal_distance=max_horizontal_distance
    )

    # Calculate scale if we have distance information
    if distances:
        mean_dist = np.mean(distances)
        px_per_ft = mean_dist / 40  # 40 feet between hash mark rows
        print(f"Vertical scale: {px_per_ft:.2f} px/foot")
        print(f"Quality metrics: {len(matched_pairs)} matched, {len(estimated_pairs)} estimated")
        return mean_dist, matched_pairs, estimated_pairs
    elif matched_pairs or estimated_pairs:
        print("Hash mark intersections estimated without yard line references")
        print(f"Results: {len(matched_pairs)} matched, {len(estimated_pairs)} estimated")
        return None, matched_pairs, estimated_pairs
    else:
        print("No hash mark intersections could be determined")
        return None, [], []


def find_bounding_hash_lines(masters, hash_centers, image, player_detections,
                             tolerance=15, max_horizontal_distance=40):
    """
    Find bounding hash mark lines (left and right) that contain all players

    Args:
        masters: Detected yard lines from HoughLines
        hash_centers: Hash mark intersection points from YOLO
        image: Original image
        player_detections: Player detection results with .xyxy attribute
        tolerance: Tolerance for matching hash marks with yard lines
        max_horizontal_distance: Max horizontal distance for hash mark pairing

    Returns:
        tuple: (left_bounding_line, right_bounding_line, all_hash_lines, debug_info)
    """

    # Step 1: Process hash marks to get connected lines
    mean_dist, matched_pairs, estimated_pairs = process_hash_marks(
        masters, hash_centers, image, tolerance, max_horizontal_distance
    )

    # Combine all hash mark lines
    all_hash_lines = matched_pairs + estimated_pairs

    if not all_hash_lines:
        print("No hash mark lines found - cannot determine bounding lines")
        return None, None, [], {"error": "No hash lines detected"}

    # Step 2: Calculate player bounding box (x-coordinates only)
    player_centroids = np.array([
        [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        for box in player_detections.xyxy
    ])

    if len(player_centroids) == 0:
        print("No players detected - cannot determine bounding region")
        return None, None, all_hash_lines, {"error": "No players detected"}

    min_player_x = player_centroids[:, 0].min()
    max_player_x = player_centroids[:, 0].max()

    # Step 4: Find bounding hash lines
    left_bounding_line = None
    right_bounding_line = None

    # Sort all hash lines by their x-coordinates
    all_hash_lines = sorted(all_hash_lines, key=lambda line: (line[0][0] + line[1][0]) / 2)

    for top_pt, bottom_pt in all_hash_lines:
        if top_pt[0] <= min_player_x and bottom_pt[0] <= min_player_x:
            left_bounding_line = (top_pt, bottom_pt)
        else:
            break

    for top_pt, bottom_pt in all_hash_lines:
        if top_pt[0] >= max_player_x and bottom_pt[0] >= max_player_x:
            right_bounding_line = (top_pt, bottom_pt)
        else:
            break

    # Step 5: Fallback if no suitable bounding lines found
    if left_bounding_line is None:
        left_bounding_line = all_hash_lines[0]  # Leftmost available line
        print("Warning: Using leftmost available hash line as left bound")

    if right_bounding_line is None:
        right_bounding_line = all_hash_lines[-1]  # Rightmost available line
        print("Warning: Using rightmost available hash line as right bound")

    # Step 6: Debug information
    debug_info = {
        "player_x_range": (min_player_x, max_player_x),
        "left_line_x": (left_bounding_line[0][0] + left_bounding_line[1][0]) / 2,
        "right_line_x": (right_bounding_line[0][0] + right_bounding_line[1][0]) / 2,
        "total_hash_lines": len(all_hash_lines),
        "matched_pairs": len(matched_pairs),
        "estimated_pairs": len(estimated_pairs)
    }

    return left_bounding_line, right_bounding_line, all_hash_lines, debug_info


def visualize_bounding_hash_lines(image, left_line, right_line, all_lines):
    """
    Visualize the bounding hash lines and player positions
    """
    vis_img = image.copy()

    # Draw all hash lines in gray
    for top_pt, bottom_pt in all_lines:
        cv2.line(vis_img, tuple(map(int, top_pt)), tuple(map(int, bottom_pt)), (128, 128, 128), 1)

    # Draw bounding lines
    if left_line:
        cv2.line(vis_img, tuple(map(int, left_line[0])), tuple(map(int, left_line[1])), (0, 255, 0), 3)
        cv2.putText(vis_img, "LEFT", tuple(map(int, left_line[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if right_line:
        cv2.line(vis_img, tuple(map(int, right_line[0])), tuple(map(int, right_line[1])), (0, 0, 255), 3)
        cv2.putText(vis_img, "RIGHT", tuple(map(int, right_line[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return vis_img


def get_bounding_region_coordinates(left_line, right_line, image_shape):
    """
    Get the coordinates for the bounding region defined by the hash lines

    Returns:
        tuple: (x_min, y_min, x_max, y_max) of the bounding region
    """
    if not left_line or not right_line:
        return None

    height, width = image_shape[:2]

    # Get x-coordinates of the bounding lines
    left_x = min(left_line[0][0], left_line[1][0])
    right_x = max(right_line[0][0], right_line[1][0])

    # Use full height of the image
    return int(left_x), 0, int(right_x), height


def calculate_field_scale_and_bounds(left_line, right_line, image_shape,
                                     hash_line_real_width_ft=40,
                                     field_extension_ft=60):
    """
    Calculate the full field bounding box by extending beyond hash marks
    Each line is extended individually based on its own scale and direction

    Args:
        left_line: Left bounding hash line ((x1,y1), (x2,y2))
        right_line: Right bounding hash line ((x1,y1), (x2,y2))
        image_shape: Shape of the image (height, width)
        hash_line_real_width_ft: Real width of hash line in feet (default 40)
        field_extension_ft: How much to extend beyond hash marks in feet (default 60)

    Returns:
        dict: Contains extended bounding box and scale information
    """

    if not left_line or not right_line:
        return None

    # Calculate the vertical length of each hash line in pixels
    left_top, left_bottom = left_line
    right_top, right_bottom = right_line

    left_length_px = math.sqrt((left_bottom[0] - left_top[0]) ** 2 + (left_bottom[1] - left_top[1]) ** 2)
    right_length_px = math.sqrt((right_bottom[0] - right_top[0]) ** 2 + (right_bottom[1] - right_top[1]) ** 2)

    # Calculate pixels per foot for each line separately
    left_pixels_per_foot = left_length_px / hash_line_real_width_ft
    right_pixels_per_foot = right_length_px / hash_line_real_width_ft

    # Calculate extension distance in pixels for each line
    left_extension_px = field_extension_ft * left_pixels_per_foot
    right_extension_px = field_extension_ft * right_pixels_per_foot

    # Calculate unit vectors for each hash line (pointing from top to bottom)
    left_vector = np.array([left_bottom[0] - left_top[0], left_bottom[1] - left_top[1]])
    right_vector = np.array([right_bottom[0] - right_top[0], right_bottom[1] - right_top[1]])

    # Normalize vectors
    left_unit = left_vector / np.linalg.norm(left_vector)
    right_unit = right_vector / np.linalg.norm(right_vector)

    # Extend lines beyond hash marks using individual scales
    # Extend upward (negative direction)
    left_extended_top = np.array(left_top) - left_unit * (left_extension_px * 0.75)
    right_extended_top = np.array(right_top) - right_unit * (right_extension_px * 0.75)

    # Extend downward (positive direction)
    left_extended_bottom = np.array(left_bottom) + left_unit * (left_extension_px * 1.3)
    right_extended_bottom = np.array(right_bottom) + right_unit * (right_extension_px * 1.3)

    # Calculate the four corners of the extended field
    extended_field_corners = {
        'left_top': tuple(left_extended_top.astype(int)),
        'left_bottom': tuple(left_extended_bottom.astype(int)),
        'right_top': tuple(right_extended_top.astype(int)),
        'right_bottom': tuple(right_extended_bottom.astype(int))
    }

    # Create a more precise quadrilateral for the field
    field_quad = np.array([
        left_extended_top,
        right_extended_top,
        right_extended_bottom,
        left_extended_bottom
    ], dtype=np.float32)

    return {
        'field_corners': extended_field_corners,
        'field_quadrilateral': field_quad,
        'scale_info': {
            'left_pixels_per_foot': left_pixels_per_foot,
            'right_pixels_per_foot': right_pixels_per_foot,
            'left_hash_length_px': left_length_px,
            'right_hash_length_px': right_length_px,
            'left_extension_px': left_extension_px,
            'right_extension_px': right_extension_px,
            'hash_line_real_width_ft': hash_line_real_width_ft,
            'field_extension_ft': field_extension_ft
        },
        'original_hash_lines': {
            'left': left_line,
            'right': right_line
        }
    }


def visualize_extended_field(vis_img, field_data):
    """
    Visualize the extended field boundaries
    """
    if not field_data:
        return vis_img

    # Draw original hash lines
    left_line = field_data['original_hash_lines']['left']
    right_line = field_data['original_hash_lines']['right']

    cv2.line(vis_img, tuple(map(int, left_line[0])), tuple(map(int, left_line[1])), (0, 255, 0), 3)
    cv2.line(vis_img, tuple(map(int, right_line[0])), tuple(map(int, right_line[1])), (0, 255, 0), 3)

    # Draw extended field quadrilateral
    field_quad = field_data['field_quadrilateral'].astype(int)
    cv2.polylines(vis_img, [field_quad], True, (0, 0, 255), 3)

    # Draw extended corner points
    corners = field_data['field_corners']
    for corner_name, (x, y) in corners.items():
        if 0 <= x < vis_img.shape[1] and 0 <= y < vis_img.shape[0]:
            cv2.circle(vis_img, (x, y), 8, (255, 0, 0), -1)
            cv2.putText(vis_img, corner_name, (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return vis_img


def process_full_field_bounds(masters, hash_centers, img_path, player_detections,
                              hash_line_real_width_ft=40, field_extension_ft=60):
    """
    Complete workflow to process field with extended boundaries
    """
    img = cv2.imread(img_path)

    # Step 1: Find bounding hash lines
    left_line, right_line, all_lines, debug_info = find_bounding_hash_lines(
        masters, hash_centers, img, player_detections
    )

    if not left_line or not right_line:
        print("Failed to find bounding hash lines")
        return None

    # Step 2: Calculate extended field boundaries
    field_data = calculate_field_scale_and_bounds(
        left_line, right_line, img.shape,
        hash_line_real_width_ft, field_extension_ft
    )

    if not field_data:
        print("Failed to calculate field boundaries")
        return None

    # Step 3: Print information
    scale_info = field_data['scale_info']
    print(f"Left field scale: {scale_info['left_pixels_per_foot']:.2f} pixels per foot")
    print(f"Right field scale: {scale_info['right_pixels_per_foot']:.2f} pixels per foot")
    print(f"Left hash line length: {scale_info['left_hash_length_px']:.1f} pixels ({hash_line_real_width_ft} feet)")
    print(f"Right hash line length: {scale_info['right_hash_length_px']:.1f} pixels ({hash_line_real_width_ft} feet)")
    print(f"Left extension distance: {scale_info['left_extension_px']:.1f} pixels ({field_extension_ft} feet)")
    print(f"Right extension distance: {scale_info['right_extension_px']:.1f} pixels ({field_extension_ft} feet)")

    # Step 4: Create visualization
    vis_img = visualize_extended_field(img, field_data)

    return field_data, vis_img, all_lines, debug_info


if __name__ == "__main__":
    from yard_line_detection import detect_yard_lines
    from hash_mark_detection import calculate_bounding_hash_marks, detect_hash_marks
    from player_detection import detect_players

    img_path = "C:/Users\Justin.Getzke\AppData\Roaming\com.uoc.football-pre-play-analysis\extracted/2108 BOISE STATE OFF vs COLORADO STATE\snaps\play_005_snap2.jpg"
    output_img_path = "field_bounds.jpg"

    masters = detect_yard_lines(img_path, "yard_lines_debug.jpg")
    hash_mark_detections = detect_hash_marks(img_path, "hash_marks_debug.jpg")
    hash_centers = calculate_bounding_hash_marks(hash_mark_detections)
    player_detections = detect_players(img_path, "players_debug.jpg")

    _, vis_img, _, debug_info = process_full_field_bounds(masters, hash_centers, img_path, player_detections)

    if vis_img is not None:
        cv2.imwrite(output_img_path, vis_img)
        print(f"Output written to {output_img_path}")
        print("Debug info:", debug_info)
