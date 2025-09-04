import cv2
import numpy as np


def calculate_field_dimensions_and_scale(field_data, avg_yard_distance_px, target_field_height_ft=160):
    """
    Calculate real-world field dimensions and create properly scaled perspective transform

    Args:
        field_data: Data from calculate_field_scale_and_bounds
        avg_yard_distance_px: Average pixel distance between yard lines
        target_field_height_ft: Real height of football field (160 feet)

    Returns:
        dict: Contains field dimensions and scaling information
    """

    if not field_data or not avg_yard_distance_px:
        return None

    # Calculate width
    width = (field_data["field_corners"]["right_bottom"][0] - field_data["field_corners"]["left_bottom"][0] +
             field_data["field_corners"]["right_top"][0] - field_data["field_corners"]["left_top"][0]) / 2
    field_width_yards = round(width / avg_yard_distance_px) * 5

    # Convert to feet (1 yard = 3 feet)
    field_width_ft = field_width_yards * 3
    field_height_ft = target_field_height_ft

    # Calculate scale factors
    scale_info = field_data['scale_info']
    avg_pixels_per_foot = (scale_info['left_pixels_per_foot'] + scale_info['right_pixels_per_foot']) / 2

    # Calculate target dimensions for scaled output
    target_width_px = int(field_width_ft * 10)  # 10 pixels per foot for good resolution
    target_height_px = int(field_height_ft * 10)  # 10 pixels per foot for good resolution

    return {
        'field_dimensions': {
            'width_yards': field_width_yards,
            'width_feet': field_width_ft,
            'height_feet': field_height_ft,
        },
        'scaling': {
            'avg_yard_distance_px': avg_yard_distance_px,
            'avg_pixels_per_foot': avg_pixels_per_foot,
            'target_width_px': target_width_px,
            'target_height_px': target_height_px,
            'real_scale_px_per_foot': 10  # Our chosen scale for output
        }
    }


def create_scaled_perspective_transform(field_data, field_dimensions_data):
    """
    Create perspective transformation matrix with proper real-world scaling

    Args:
        field_data: Data from calculate_field_scale_and_bounds
        field_dimensions_data: Data from calculate_field_dimensions_and_scale

    Returns:
        Transformation matrix and target dimensions
    """

    if not field_data or not field_dimensions_data:
        return None, None

    # Source points (field corners in image)
    field_quad = field_data['field_quadrilateral']

    # Target dimensions
    target_width = field_dimensions_data['scaling']['target_width_px']
    target_height = field_dimensions_data['scaling']['target_height_px']

    # Destination points (rectangular field with proper scaling)
    dst_points = np.array([
        [0, 0],  # top-left
        [target_width, 0],  # top-right
        [target_width, target_height],  # bottom-right
        [0, target_height]  # bottom-left
    ], dtype=np.float32)

    # Calculate perspective transformation
    transform_matrix = cv2.getPerspectiveTransform(field_quad, dst_points)

    return transform_matrix, (target_width, target_height)


if __name__ == "__main__":
    from yard_line_detection import detect_yard_lines
    from hash_mark_detection import calculate_bounding_hash_marks, detect_hash_marks
    from player_detection import detect_players
    from field_boundaries import process_full_field_bounds

    img_path = "C:/Users\Justin.Getzke\AppData\Roaming\com.uoc.football-pre-play-analysis\extracted/2108 BOISE STATE OFF vs COLORADO STATE\snaps\play_005_snap2.jpg"

    masters = detect_yard_lines(img_path, "yard_lines_debug.jpg")
    hash_mark_detections = detect_hash_marks(img_path, "hash_marks_debug.jpg")
    hash_centers, avg_yard_distance_px = calculate_bounding_hash_marks(hash_mark_detections)
    player_detections = detect_players(img_path, "C:/Users\Justin.Getzke\AppData\Roaming\com.uoc.football-pre-play-analysis\extracted/2108 BOISE STATE OFF vs COLORADO STATE/analysis\play_005_snap2_player_detection.jpg")

    field_data, vis_img, all_lines, debug_info = process_full_field_bounds(masters, hash_centers, img_path,
                                                                           player_detections)

    field_dimensions_data = calculate_field_dimensions_and_scale(
        field_data, avg_yard_distance_px
    )

    transform_matrix, target_dims = create_scaled_perspective_transform(
        field_data, field_dimensions_data
    )

    image = cv2.imread(img_path)
    scaled_top_down_view = cv2.warpPerspective(image, transform_matrix, target_dims)
    dims = field_dimensions_data['field_dimensions']
    scaling = field_dimensions_data['scaling']

    print("=== FIELD DIMENSIONS ===")
    print(f"Estimated field width: {dims['width_yards']} yards ({dims['width_feet']} feet)")
    print(f"Field height: {dims['height_feet']} feet")
    print()
    print("=== SCALING INFORMATION ===")
    print(f"Average yard distance: {scaling['avg_yard_distance_px']:.2f} pixels")
    print(f"Average pixels per foot: {scaling['avg_pixels_per_foot']:.2f}")
    print(f"Output scale: {scaling['real_scale_px_per_foot']} pixels per foot")
    print(f"Output dimensions: {scaling['target_width_px']} x {scaling['target_height_px']} pixels")
    print(f"Real-world coverage: {dims['width_feet']} x {dims['height_feet']} feet")

    cv2.imwrite("C:/Users\Justin.Getzke\AppData\Roaming\com.uoc.football-pre-play-analysis\extracted/2108 BOISE STATE OFF vs COLORADO STATE/analysis\play_005_snap2_top_down_perspective.jpg", scaled_top_down_view)
