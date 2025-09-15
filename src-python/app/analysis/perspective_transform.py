import cv2
import numpy as np


def generate_transform_matrix(field_bounds: np.ndarray, field_width_yards: int) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Generate a perspective transformation matrix to convert field bounds to a scaled rectangular view.
    """
    # Target dimensions
    target_width = field_width_yards * 3 * 10
    target_height = 1600

    # Destination points (rectangular field with proper scaling)
    dst_points = np.array([
        [0, 0],  # top-left
        [target_width, 0],  # top-right
        [target_width, target_height],  # bottom-right
        [0, target_height]  # bottom-left
    ], dtype=np.float32)

    # Calculate perspective transformation
    transform_matrix = cv2.getPerspectiveTransform(field_bounds, dst_points)

    return transform_matrix, (target_width, target_height)


def perspective_transform(image, field_bounds, field_width_yards):
    """
    Complete workflow with proper real-world scaling
    """
    # Step 3: Create scaled perspective transform
    transform_matrix, target_dims = generate_transform_matrix(
        field_bounds, field_width_yards
    )

    if transform_matrix is None:
        print("Failed to create perspective transform")
        return None

    # Step 4: Apply transformation
    scaled_top_down_view = cv2.warpPerspective(image, transform_matrix, target_dims)

    return scaled_top_down_view, transform_matrix, target_dims


def transform_player_positions(player_positions, transform_matrix):
    """
    Transform player positions from original image coordinates to transformed field coordinates

    Args:
        player_positions: List of player dictionaries with 'foot_position' key containing (x, y) tuples
        transform_matrix: The perspective transformation matrix from cv2.getPerspectiveTransform

    Returns:
        List of player dictionaries with added 'transformed_position' key
    """
    # Extract coordinates for batch transformation
    original_coords = []
    for player in player_positions:
        foot_pos = player['foot_position']
        original_coords.append([foot_pos[0], foot_pos[1]])

    # Convert to numpy array and reshape for cv2.perspectiveTransform
    original_coords = np.array(original_coords, dtype=np.float32)
    original_coords = original_coords.reshape(-1, 1, 2)

    # Apply perspective transformation
    transformed_coords = cv2.perspectiveTransform(original_coords, transform_matrix)

    # Reshape back and update player positions
    transformed_coords = transformed_coords.reshape(-1, 2)

    # Create new list with transformed positions
    transformed_players = []
    for i, player in enumerate(player_positions):
        transformed_player = player.copy()
        transformed_x, transformed_y = transformed_coords[i]
        transformed_player['transformed_position'] = (int(transformed_x), int(transformed_y))
        transformed_players.append(transformed_player)

    return transformed_players


def visualize_transformed_players(transformed_field_image, transformed_players, radius=8, x_shift=0):
    """
    Draw player positions on the transformed field view

    Args:
        transformed_field_image: The warped top-down field image
        transformed_players: List of players with 'transformed_position' and 'class' keys

    Returns:
        Image with players drawn on it
        :param radius: Radius of circle
        :param x_shift: X-shift of player position
    """
    vis_img = transformed_field_image.copy()

    # Color mapping for different player classes
    color_map = {
        'oline': (64, 64, 255),  # Red for offense line
        'qb': (160, 161, 255),  # Orange for quarterback
        'skill': (51, 182, 255),  # Yellow for skill players
        'defense': (251, 81, 163),  # Blue for defense
        'ref': (255, 255, 255),
    }

    for player in transformed_players:
        if 'transformed_position' in player:
            pos = player['transformed_position']
            player_class = player.get('class', 'unknown')
            color = color_map.get(player_class, (255, 255, 255))  # Default white

            # Draw player position
            cv2.circle(vis_img, (pos[0] + x_shift, pos[1]), radius=radius, color=color, thickness=-1)

            cv2.circle(vis_img, (pos[0] + x_shift, pos[1]), radius=radius, color=(0, 0, 0), thickness=2)

    return vis_img


def get_foot_position(bbox):
    """Extract foot position from bounding box (bottom center)"""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def visualize_transformed_player_positions(transformed_image, transform_matrix, player_detections):
    """
    Create a complete visualization combining your existing workflow with player transformation

    Args:
        complete_result: Result from your process_complete_field_scaling function
        player_detections: Your player detections

    Returns:
        Dictionary with both original and transformed visualizations
    """

    # Extract foot positions from detections (using your existing logic)
    player_foot_positions = []
    for detection in player_detections:
        player = {
            "class": detection[5]['class_name'],
            "foot_position": get_foot_position(detection[0]),
        }
        player_foot_positions.append(player)

    # Transform player positions
    transformed_players = transform_player_positions(player_foot_positions, transform_matrix)

    # Create visualizations
    transformed_field = transformed_image.copy()
    transformed_vis = visualize_transformed_players(transformed_field, transformed_players)

    return transformed_vis, transformed_players


def get_los(transformed_players):
    oline_players = [p for p in transformed_players if p.get('class') == 'oline' and 'transformed_position' in p]
    defense_players = [p for p in transformed_players if p.get('class') == 'defense' and 'transformed_position' in p]

    if len(oline_players) == 0 or len(defense_players) == 0:
        print("Not enough oline or defense players to compute closest pair.")
    else:
        # Find closest oline-defense pair (Euclidean distance)
        best_pair = None
        best_dist_sq = float('inf')

        for o_player in oline_players:
            ox, oy = o_player['transformed_position']
            for d_player in defense_players:
                dx, dy = d_player['transformed_position']
                dist_sq = (ox - dx) ** 2 + (oy - dy) ** 2
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_pair = (o_player, d_player)

        (ox, oy) = best_pair[0]['transformed_position']
        (dx, dy) = best_pair[1]['transformed_position']
        x_los = int(round((ox + dx) / 2))

        return x_los


def transform_positions(positions: list[tuple[np.ndarray, np.ndarray]], transform_matrix):
    # Example Input
    # (array([354.5625,  30.625 ]), array([-211.9125,  690.475 ]))
    # (array([485.625,  31.75 ]), array([7.5000e-02, 7.0405e+02]))
    # (array([616.0625,  36.0625]), array([215.5875, 714.5875]))
    # (array([748.4375,  40.9375]), array([435.1125, 723.6125]))
    # (array([885.4375,  42.5625]), array([655.1125, 737.6875]))
    # (array([1024.3125,   44.1875]), array([881.1375, 751.7625]))

    # Return a similar list of tuples but with all coordinates transformed
    transformed = []
    for pos_pair in positions:
        transformed_pair = []
        for pos in pos_pair:
            original_coords = np.array([[pos]], dtype=np.float32)
            transformed_coords = cv2.perspectiveTransform(original_coords, transform_matrix)
            transformed_pair.append(transformed_coords[0][0])
        transformed.append(tuple(transformed_pair))
    return transformed


def transform_image(img_path: str, output_img_path_top_down: str, output_img_path_players: str, field_quad,
                    yard_distance, player_detections):
    img = cv2.imread(img_path)

    top_down_image, transform_matrix, target_dims = perspective_transform(img, field_quad, yard_distance)

    cv2.imwrite(output_img_path_top_down, top_down_image)

    transformed_vis, transformed_players = visualize_transformed_player_positions(top_down_image, transform_matrix,
                                                                                  player_detections)
    x_los = get_los(transformed_players)

    height, width = transformed_vis.shape[:2]

    cv2.line(transformed_vis, (x_los, 0), (x_los, height - 1), (0, 255, 255), 3)

    cv2.imwrite(output_img_path_players, transformed_vis)

    return transformed_players, x_los
