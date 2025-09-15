import os
import logging

import cv2

from app.utils import get_resource_path

logger = logging.getLogger("ArtificialPitch")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def draw_positions_on_artificial_pitch(output_img_path: str, player_positions, x_los, los_yards, direction='left',
                                       artificial_pitch_path=None):
    """
    Draw player positions on an artificial pitch image.

    Args:
        output_img_path: Path to save the annotated image.
        player_positions: List of dicts with keys 'class' and 'transformed_position' (x, y).
        x_los: X-coordinate of the line of scrimmage in transformed coordinates.
        los_yards: Y-coordinate of the line of scrimmage in yards.
        direction: Position of the line of scrimmage ('left' or 'right').

    Returns:
        Annotated artificial pitch image as a numpy array.
    """
    if not artificial_pitch_path:
        artificial_pitch_path = get_resource_path("resources/artificial_pitch.png")
        if not os.path.exists(artificial_pitch_path):
            logger.error(f"Artificial pitch image not found at {artificial_pitch_path}")
            raise FileNotFoundError(f"Artificial pitch image not found at {artificial_pitch_path}")
        else:
            logger.info(f"Artificial pitch image found at {artificial_pitch_path}")

    logger.debug(f"Output image path: {output_img_path}")
    logger.debug(f"Number of player positions: {len(player_positions) if player_positions else 0}")
    logger.debug(f"Line of scrimmage X (transformed): {x_los}")
    logger.debug(f"Line of scrimmage Y (yards): {los_yards}")
    logger.debug(f"Direction: {direction}")
    # Draw los line
    pitch_image = cv2.imread(artificial_pitch_path)
    if pitch_image is None:
        logger.error("Failed to load artificial pitch image.")
        raise ValueError("Failed to load artificial pitch image.")
    pitch_height, pitch_width, _ = pitch_image.shape

    # Map yards to pixels on the artificial pitch (1 yard ≈ 30px) with a small left margin
    pixels_per_yard = 30
    margin = 300

    if direction == 'left':
        los_x_px = int(margin + los_yards * pixels_per_yard)
    elif direction == 'right':
        los_x_px = int(pitch_width - (margin + los_yards * pixels_per_yard))
    else:
        logger.error("Invalid direction value. Use 'left' or 'right'.")
        raise ValueError("Invalid direction value. Use 'left' or 'right'.")
    logger.debug(f"Line of scrimmage X position in pixels: {los_x_px}")
    # Draw LOS as a vertical line
    cv2.line(pitch_image, (los_x_px, 0), (los_x_px, pitch_height), (0, 255, 255), 3)

    # Draw player positions relative to LOS (only adjust X relative to LOS; keep Y as-is)
    for player in player_positions or []:
        try:
            player_class = player.get('class', 'player')
            transformed_position = player.get('transformed_position')
            if transformed_position is None or len(transformed_position) != 2:
                logger.warning("Skipping player with invalid transformed_position: %s", player)
                continue

            transformed_x, transformed_y = transformed_position

            # Relative X distance to LOS in the SAME pixel space as transformed positions
            # No scaling here since inputs are already scaled appropriately
            relative_x_px = int(round(float(transformed_x) - float(x_los)))

            # Offset from LOS pixel position
            player_x_px = int(los_x_px + relative_x_px)

            # Keep Y as provided (no conversion requested); clamp to image bounds
            player_y_px = int(transformed_y)

            # Clamp to image bounds to avoid drawing outside
            player_x_px = max(0, min(pitch_width - 1, player_x_px))
            player_y_px = max(0, min(pitch_height - 1, player_y_px))

            # Color mapping for different player classes
            color_map = {
                'oline': (64, 64, 255),  # Red for offense line
                'qb': (160, 161, 255),  # Orange for quarterback
                'skill': (51, 182, 255),  # Yellow for skill players
                'defense': (251, 81, 163),  # Blue for defense
                'ref': (255, 255, 255),
            }

            # Draw the player as a circle
            cv2.circle(pitch_image, (player_x_px, player_y_px), 10, color_map.get(player_class, (200, 200, 200)), -1)
            cv2.circle(pitch_image, (player_x_px, player_y_px), 10, (0, 0, 0), 2)

            # Annotate with class
            label = f"{player_class}"
            cv2.putText(
                pitch_image,
                label,
                (player_x_px + 8, max(12, player_y_px - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        except Exception as e:
            logger.exception("Failed drawing player %s due to error: %s", player, e)

    cv2.imwrite(output_img_path, pitch_image)

    return pitch_image
