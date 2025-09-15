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
    H, _ = image.shape[:2]

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
            mean_dist = float(np.mean(all_distances))
            # Only estimate if the standard deviation is reasonable (consistent distances)
            std_dist = np.std(all_distances)
            if std_dist < mean_dist * 0.3:  # Less than 30% variation
                estimated_pairs = estimate_missing_hash_marks(
                    single_matches, mean_dist, H, used_hash_marks, max_horizontal_distance
                )

    # Method 2: CONSERVATIVE fallback - only for very clear cases
    if not matched_pairs and not estimated_pairs:
        estimated_pairs = conservative_hash_clustering(
            hash_centers, H, max_horizontal_distance
        )

    return matched_pairs, estimated_pairs, all_distances


def process_hash_marks(yard_lines, hash_centers, image, tolerance=15, max_horizontal_distance=40):
    """Main function to process hash marks with conservative estimation"""

    H, _ = image.shape[:2]

    # Convert yard_lines to yard line segments
    master_segments = [
        (int(round(x_top)), 0, int(round(x_bot)), H)
        for x_top, x_bot in yard_lines
    ] if yard_lines else []

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
        return mean_dist, matched_pairs, estimated_pairs
    elif matched_pairs or estimated_pairs:
        print("Hash mark intersections estimated without yard line references")
        print(f"Results: {len(matched_pairs)} matched, {len(estimated_pairs)} estimated")
        return None, matched_pairs, estimated_pairs
    else:
        print("No hash mark intersections could be determined")
        return None, [], []


def extend_hash_lines(
        all_hash_lines: list,
        hash_line_width_ft=40,
        field_extension_ft=60,
        top_extension_factor=0.75,
        bottom_extension_factor=1.35
):
    extended_hash_lines = []

    for hash_line in all_hash_lines:
        top, bottom = hash_line
        length_px = math.sqrt((bottom[0] - top[0]) ** 2 + (bottom[1] - top[1]) ** 2)

        px_per_ft = length_px / hash_line_width_ft
        extension_px = field_extension_ft * px_per_ft
        line_vector = np.array([bottom[0] - top[0], bottom[1] - top[1]])
        unit = line_vector / np.linalg.norm(line_vector)

        extended_top = np.array(top) - unit * (extension_px * top_extension_factor)
        extended_bottom = np.array(bottom) + unit * (extension_px * bottom_extension_factor)
        extended_hash_lines.append((extended_top, extended_bottom))

    # Sort extended_hash_lines by their x coordinate
    extended_hash_lines = sorted(
        extended_hash_lines,
        key=lambda line: (line[0][0] + line[1][0]) / 2
    )

    return extended_hash_lines


def _interpolate_x_at_y(pt1, pt2, y):
    # Linear interpolation to find x at given y between pt1 and pt2
    (x1, y1), (x2, y2) = pt1, pt2
    if y2 == y1:
        return (x1 + x2) / 2  # horizontal line, just average x
    t = (y - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)


def find_bounding_hash_lines(yard_lines, hash_centers, image, player_detections,
                             tolerance=15, max_horizontal_distance=40) -> dict:
    """
    Find bounding hash mark lines (left and right) that contain all players

    Args:
        yard_lines: Detected yard lines from HoughLines
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
        yard_lines, hash_centers, image, tolerance, max_horizontal_distance
    )

    # Combine all hash mark lines
    all_hash_lines = matched_pairs + estimated_pairs

    if not all_hash_lines:
        print("No hash mark lines found - cannot determine bounding lines")
        return {}

    extended_hash_lines = extend_hash_lines(all_hash_lines)

    if not extended_hash_lines:
        print("No extended hash lines found - cannot determine bounding lines")
        return {}

    # Step 2: Calculate player bounding box (x-coordinates only)
    player_centroids = np.array([
        [(box[0] + box[2]) / 2, box[3]]
        for box in player_detections.xyxy
    ])

    if len(player_centroids) == 0:
        print("No players detected - cannot determine bounding region")
        return {}

    min_player_x = player_centroids[:, 0].min()
    max_player_x = player_centroids[:, 0].max()

    # Try to find the tightest pair of lines that contains all players
    best_pair = None
    best_avg_width = float('inf')

    num_lines = len(extended_hash_lines)
    if num_lines >= 2:
        for i in range(num_lines - 1):
            line_i = extended_hash_lines[i]
            for j in range(i + 1, num_lines):
                line_j = extended_hash_lines[j]

                valid_for_all = True
                widths = []
                for px, py in player_centroids:
                    xi = _interpolate_x_at_y(line_i[0], line_i[1], py)
                    xj = _interpolate_x_at_y(line_j[0], line_j[1], py)
                    xl, xr = (xi, xj) if xi <= xj else (xj, xi)
                    if px < xl or px > xr:
                        valid_for_all = False
                        break
                    widths.append(xr - xl)

                if valid_for_all and widths:
                    avg_width = float(np.mean(widths))
                    if avg_width < best_avg_width:
                        best_avg_width = avg_width
                        best_pair = (i, j)

    if best_pair is not None:
        # Order left/right by x at the average player y
        y_ref = float(np.mean(player_centroids[:, 1]))
        li = extended_hash_lines[best_pair[0]]
        lj = extended_hash_lines[best_pair[1]]
        xi = _interpolate_x_at_y(li[0], li[1], y_ref)
        xj = _interpolate_x_at_y(lj[0], lj[1], y_ref)
        if xi <= xj:
            left_bounding_line, right_bounding_line = li, lj
        else:
            left_bounding_line, right_bounding_line = lj, li
    else:
        # Fallback: choose nearest lines around the min/max player x at a representative y
        y_ref = float(np.mean(player_centroids[:, 1]))
        best_left = None
        best_left_x = -float('inf')
        best_right = None
        best_right_x = float('inf')

        for top_pt, bottom_pt in extended_hash_lines:
            lx = _interpolate_x_at_y(top_pt, bottom_pt, y_ref)
            if lx <= min_player_x and lx > best_left_x:
                best_left_x = lx
                best_left = (top_pt, bottom_pt)
            if lx >= max_player_x and lx < best_right_x:
                best_right_x = lx
                best_right = (top_pt, bottom_pt)

        left_bounding_line = best_left
        right_bounding_line = best_right

    if left_bounding_line is None:
        left_bounding_line = extended_hash_lines[0]  # Leftmost available line
        print("Warning: Using leftmost available hash line as left bound")

    if right_bounding_line is None:
        right_bounding_line = extended_hash_lines[-1]  # Rightmost available line
        print("Warning: Using rightmost available hash line as right bound")

    if estimated_pairs:
        print(
            f"WARNING: Bounding lines determined with estimated hash marks {len(estimated_pairs)} - accuracy may be reduced")

    return {
        "left_bounding_line": left_bounding_line,
        "right_bounding_line": right_bounding_line,
        "all_hash_lines": all_hash_lines,
        "extended_hash_lines": extended_hash_lines,
        "matched_pairs": matched_pairs,
        "estimated_pairs": estimated_pairs,
        "player_centroids": player_centroids,
    }


def check_if_players_outside(left_line, right_line, player_detections):
    left_outside = 0
    right_outside = 0

    xyxy = player_detections.xyxy
    class_names = player_detections.data.get('class_name', None)
    relevant_classes = ["oline", "defense", "qb", "skill"]

    for i in range(len(xyxy)):
        player_class = class_names[i] if class_names is not None else None
        if player_class not in relevant_classes:
            continue
        # Get center of bounding box
        x1, y1, x2, y2 = xyxy[i]
        px = (x1 + x2) / 2
        py = (y1 + y2) / 2

        # Interpolate x on left line at player's y
        left_line_x = _interpolate_x_at_y(left_line[0], left_line[1], py)
        right_line_x = _interpolate_x_at_y(right_line[0], right_line[1], py)
        if px < left_line_x:
            left_outside += 1
        elif px > right_line_x:
            right_outside += 1

    return left_outside, right_outside


def point_along_line(p1, p2, distance, target):
    """
    Given two points p1, p2 in image coordinates that correspond to a real-world
    distance, compute the point after `target` yards from p1 along the same line.

    Args:
        p1 (tuple): Start point (x1, y1) in pixels
        p2 (tuple): Second point (x2, y2) in pixels
        distance (float): Real-world distance between p1 and p2
        target (float): Real-world distance from p1 where we want the new point

    Returns:
        tuple: (x, y) coordinates in the original image system
    """
    (x1, y1), (x2, y2) = p1, p2

    # Vector from p1 to p2
    dx, dy = x2 - x1, y2 - y1

    # Length of vector in pixel space
    length_pixels = math.hypot(dx, dy)

    if length_pixels == 0:
        raise ValueError("p1 and p2 cannot be the same point.")

    # Pixel per yard
    scale = length_pixels / distance

    # How many pixels to move for the target distance
    move = target * scale

    # Normalize direction
    ux, uy = dx / length_pixels, dy / length_pixels

    # Compute new point
    new_x = x1 + ux * move
    new_y = y1 + uy * move
    return (new_x, new_y)


def process_full_field_bounds(img_path: str, output_img_path: str, yard_lines, hash_mark_centers, player_detections,
                              disable_estimation=False):
    """
    Complete workflow to process field with extended boundaries
    """
    img = cv2.imread(img_path)

    hash_line_result = find_bounding_hash_lines(
        yard_lines, hash_mark_centers, img, player_detections
    )

    extended_hash_lines = hash_line_result.get("extended_hash_lines", [])

    if hash_line_result["left_bounding_line"] is None or hash_line_result["right_bounding_line"] is None:
        print("Failed to find bounding hash lines")
        return None

    left_outside, right_outside = check_if_players_outside(hash_line_result["left_bounding_line"],
                                                           hash_line_result["right_bounding_line"], player_detections)

    left_line = hash_line_result["left_bounding_line"]
    right_line = hash_line_result["right_bounding_line"]
    print(f"Initial players outside - Left: {left_outside}, Right: {right_outside}")
    print(f"Initial bounding lines at x: {left_line[0][0]:.1f}, {right_line[0][0]:.1f}")
    print(f"Total hash lines available: {len(extended_hash_lines)}")
    y_ref = img.shape[0] * 0.5

    left_x_ref = (left_line[0][0] + (y_ref - left_line[0][1]) * (left_line[1][0] - left_line[0][0]) / (
            left_line[1][1] - left_line[0][1])
                  if left_line[1][1] != left_line[0][1] else (left_line[0][0] + left_line[1][0]) / 2.0)
    right_x_ref = (right_line[0][0] + (y_ref - right_line[0][1]) * (right_line[1][0] - right_line[0][0]) / (
            right_line[1][1] - right_line[0][1])
                   if right_line[1][1] != right_line[0][1] else (right_line[0][0] + right_line[1][0]) / 2.0)

    x_min, x_max = (left_x_ref, right_x_ref) if left_x_ref <= right_x_ref else (right_x_ref, left_x_ref)

    num_lines_between = 0
    for line in extended_hash_lines:
        lx = (line[0][0] + (y_ref - line[0][1]) * (line[1][0] - line[0][0]) / (line[1][1] - line[0][1])
              if line[1][1] != line[0][1] else (line[0][0] + line[1][0]) / 2.0)
        if x_min < lx < x_max:
            num_lines_between += 1

    yard_line_amount = num_lines_between + 1
    yard_distance = 5 * yard_line_amount

    iteration = 0
    while (left_outside or right_outside) and iteration < 3 and not disable_estimation:
        if right_outside:
            print(f"WARNING: There are {right_outside} players outside right bounding line")
            new_right_line_top = point_along_line(left_line[0], right_line[0], yard_distance, yard_distance + 5)
            new_right_line_bottom = point_along_line(left_line[1], right_line[1], yard_distance, yard_distance + 5)
            right_line = (new_right_line_top, new_right_line_bottom)

            yard_line_amount += 1
            yard_distance += 5

        if left_outside:
            print(f"WARNING: There are {left_outside} players outside left bounding line")
            # Calculate the point for the next x (where we shift our left_line[0] to)
            new_left_line_top = point_along_line(right_line[0], left_line[0], -yard_distance, -yard_distance - 5)
            new_left_line_bottom = point_along_line(right_line[1], left_line[1], -yard_distance, -yard_distance - 5)
            left_line = (new_left_line_top, new_left_line_bottom)
            yard_line_amount += 1
            yard_distance += 5

        left_outside, right_outside = check_if_players_outside(left_line, right_line, player_detections)
        iteration += 1

        print(f"Iteration {iteration}: yard line amount: {yard_line_amount}, distance: {yard_distance} feet")
        print(f"Players outside - Left: {left_outside}, Right: {right_outside}")

        if iteration == 3 and (left_outside or right_outside):
            print(
                f"WARNING: There are still players outside the boundaries \n\n Left: {left_outside}, Right: {right_outside}")

    print(f"Final yard line amount: {yard_line_amount}, distance: {yard_distance} feet, iterations: {iteration}")

    field_quad = np.array([
        left_line[0],
        right_line[0],
        right_line[1],
        left_line[1]
    ], dtype=np.float32)

    cv2.polylines(img, [field_quad.astype(int)], True, (0, 0, 255), 3)
    cv2.imwrite(output_img_path, img)

    return field_quad, yard_line_amount, yard_distance
