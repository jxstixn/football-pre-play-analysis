import os
import asyncio
import logging
import traceback
from typing import Any, Awaitable, Callable, Dict, List, Tuple

import numpy as np
from supervision import Detections

from app.models import AnalyzeSnapInput
from app.analysis.yard_line_detection import detect_yard_lines
from app.analysis.hash_mark_detection import detect_hash_marks
from app.analysis.player_detection import detect_players
from app.analysis.field_boundaries import process_full_field_bounds
from app.analysis.perspective_transform import transform_image
from app.analysis.field_positioning import classify_yard_lines_and_los
from app.analysis.formation_classifier import classify_formation
from app.analysis.artificial_pitch import draw_positions_on_artificial_pitch

logger = logging.getLogger("SnapAnalysis")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


async def process_snap_analysis_job(job_id: str, params: AnalyzeSnapInput, emit: Callable[[dict], Awaitable[None]]):
    """
    Runs a small pipeline of synchronous stages, emitting a 'stage' event
    *before* each stage starts. Synchronous work is offloaded with asyncio.to_thread.
    Always emits a final 'result' event with status and artifacts.
    """
    file_path = getattr(params, "file_path")
    output_dir = getattr(params, "output_dir")
    os.makedirs(output_dir, exist_ok=True)

    input_filename = os.path.basename(file_path)
    input_name, _ = os.path.splitext(input_filename)
    logger.info(f"Starting snap analysis job {job_id} for file {file_path}")
    # Derive paths you care about
    yard_lines_img = os.path.join(output_dir, f"{input_name}_DEBUG_yard_lines.jpg")
    hash_marks_img = os.path.join(output_dir, f"{input_name}_DEBUG_hash_marks.jpg")
    player_detections_img = os.path.join(output_dir, f"{input_name}_player_detection.jpg")
    field_boundaries_img = os.path.join(output_dir, f"{input_name}_DEBUG_field_boundaries.jpg")
    top_down_perspective = os.path.join(output_dir, f"{input_name}_DEBUG_top_down.jpg")
    annotated_top_down = os.path.join(output_dir, f"{input_name}_annotated_top_down.jpg")
    classified_yard_lines = os.path.join(output_dir, f"{input_name}_DEBUG_classified_yard_lines.jpg")
    formation_result = os.path.join(output_dir, f"{input_name}_formation.json")
    artificial_pitch = os.path.join(output_dir, f"{input_name}_top_down_artificial.jpg")

    artifacts: Dict[str, Any] = {}
    errors: list[dict] = []
    aborted = False  # set True on any non-recoverable error

    stages: List[Tuple[str, Callable[..., Any], tuple, dict]] = [
        ("detect_yard_lines", detect_yard_lines, (file_path, yard_lines_img), {}),
        ("detect_hash_marks", detect_hash_marks, (file_path, hash_marks_img), {}),
        ("detect_players", detect_players, (file_path, player_detections_img), {}),
        ("process_field_boundaries", process_full_field_bounds, (file_path, field_boundaries_img), {}),
        ("transform_image", transform_image, (file_path, top_down_perspective, annotated_top_down), {}),
        ("classify_yard_lines_and_los", classify_yard_lines_and_los, (top_down_perspective, classified_yard_lines), {}),
        ("classify_formation", classify_formation, (), {}),
        ("draw_artificial_pitch", draw_positions_on_artificial_pitch, artificial_pitch, {}),
    ]

    total = len(stages)

    for idx, (name, fn, args, kwargs) in enumerate(stages, start=1):
        await emit({
            "type": "stage",
            "stage": {"index": idx, "total": total, "name": name},
            "message": f"Entering stage: {name}",
        })

        try:
            match name:
                case "detect_yard_lines":
                    logger.info(f"Starting yard line detection for job {job_id}")
                    yard_lines = await asyncio.to_thread(fn, *args, **kwargs)
                    if isinstance(yard_lines, list):
                        artifacts["yard_lines"] = yard_lines
                    else:
                        logger.error(f"Yard line detection failed for job {job_id}")
                        msg = "Yard line detection failed."
                        errors.append({"stage": name, "message": msg})
                        await emit({"type": "error", "stage": name, "message": msg})
                        aborted = True
                        break

                case "detect_hash_marks":
                    logger.info(f"Starting hash mark detection for job {job_id}")
                    hash_mark_centers = await asyncio.to_thread(fn, *args, **kwargs)
                    if isinstance(hash_mark_centers, np.ndarray):
                        artifacts["hash_mark_centers"] = hash_mark_centers
                    else:
                        logger.error(f"Hash mark detection failed for job {job_id}")
                        msg = "Hash mark detection failed."
                        errors.append({"stage": name, "message": msg})
                        await emit({"type": "error", "stage": name, "message": msg})
                        aborted = True
                        break

                case "detect_players":
                    logger.info(f"Starting player detection for job {job_id}")
                    player_detections = await asyncio.to_thread(fn, *args, **kwargs)
                    if isinstance(player_detections, Detections):
                        artifacts["player_detections"] = player_detections
                    else:
                        logger.error(f"Player detection failed for job {job_id}")
                        msg = "Player detection failed."
                        errors.append({"stage": name, "message": msg})
                        await emit({"type": "error", "stage": name, "message": msg})
                        aborted = True
                        break

                case "process_field_boundaries":
                    logger.info(f"Starting field boundary processing for job {job_id}")
                    if not all(k in artifacts for k in ("yard_lines", "hash_mark_centers", "player_detections")):
                        logger.info(f"Artifacts so far: {list(artifacts.keys())}")
                        logger.error(f"Missing prerequisites for field boundary processing in job {job_id}")
                        msg = "Cannot process field boundaries without yard lines, hash mark centers, and player detections."
                        errors.append({"stage": name, "message": msg})
                        await emit({"type": "error", "stage": name, "message": msg})
                        aborted = True
                        break
                    result = await asyncio.to_thread(
                        fn,
                        file_path,
                        field_boundaries_img,
                        yard_lines=artifacts["yard_lines"],
                        hash_mark_centers=artifacts["hash_mark_centers"],
                        player_detections=artifacts["player_detections"]
                    )
                    if isinstance(result, tuple) and len(result) == 3:
                        artifacts["field_quad"] = result[0]
                        artifacts["yard_line_amount"] = result[1]
                        artifacts["yard_distance"] = result[2]
                    else:
                        logger.error(f"Field boundary processing failed for job {job_id}")
                        msg = "Field boundary processing failed."
                        errors.append({"stage": name, "message": msg})
                        await emit({"type": "error", "stage": name, "message": msg})
                        aborted = True
                        break

                case "transform_image":
                    logger.info(f"Starting image transformation for job {job_id}")
                    if not all(k in artifacts for k in ("field_quad", "yard_distance", "player_detections")):
                        msg = "Cannot transform image without prior field data and average yard distance."
                        errors.append({"stage": name, "message": msg})
                        await emit({"type": "error", "stage": name, "message": msg})
                        aborted = True
                        break
                    result = await asyncio.to_thread(
                        fn,
                        *args,
                        field_quad=artifacts["field_quad"],
                        yard_distance=artifacts["yard_distance"],
                        player_detections=artifacts["player_detections"]
                    )
                    if isinstance(result, tuple) and len(result) == 2:
                        artifacts["top_down_perspective_path"] = top_down_perspective
                        artifacts["transformed_players"] = result[0]
                        artifacts["x_los"] = result[1]
                    else:
                        logger.error(f"Image transformation failed for job {job_id}")
                        msg = "Image transformation failed."
                        errors.append({"stage": name, "message": msg})
                        await emit({"type": "error", "stage": name, "message": msg})
                        aborted = True
                        break

                case "classify_yard_lines_and_los":
                    logger.info(f"Starting yard line and LOS classification for job {job_id}")
                    if "x_los" not in artifacts:
                        msg = "Cannot classify yard lines and LOS without transformed players."
                        errors.append({"stage": name, "message": msg})
                        await emit({"type": "error", "stage": name, "message": msg})
                        aborted = True
                        break
                    result = await asyncio.to_thread(
                        fn,
                        top_down_perspective,
                        classified_yard_lines,
                        x_los=artifacts["x_los"],
                        threshold=0.75,
                    )
                    if isinstance(result, tuple) and len(result) == 3:
                        artifacts["classified_lines"] = result[0]
                        artifacts["los_yards"] = result[1]
                        artifacts["direction"] = result[2]
                    else:
                        logger.error(f"Yard line and LOS classification failed for job {job_id}")
                        msg = "Yard line and LOS classification failed."
                        errors.append({"stage": name, "message": msg})
                        await emit({"type": "error", "stage": name, "message": msg})
                        aborted = True
                        break

                case "classify_formation":
                    logger.info(f"Starting formation classification for job {job_id}")
                    if not all(k in artifacts for k in ("transformed_players", "x_los", "los_yards")):
                        msg = "Cannot classify formation without transformed players and line of scrimmage."
                        errors.append({"stage": name, "message": msg})
                        await emit({"type": "error", "stage": name, "message": msg})
                        aborted = True
                        break
                    result = await asyncio.to_thread(
                        fn,
                        formation_result_path=formation_result,
                        player_positions=artifacts["transformed_players"],
                        x_los=artifacts["x_los"],
                        los_yards=artifacts["los_yards"],
                    )
                    if result is None:
                        logger.error(f"Formation classification failed for job {job_id}")
                        msg = "Formation classification failed."
                        errors.append({"stage": name, "message": msg})
                        await emit({"type": "error", "stage": name, "message": msg})
                        aborted = True
                        break
                    artifacts["formation"] = result

                case "draw_artificial_pitch":
                    logger.info(f"Starting artificial pitch drawing for job {job_id}")
                    if not all(k in artifacts for k in ("transformed_players", "x_los", "los_yards", "direction")):
                        logger.info(f"Artifacts so far: {list(artifacts.keys())}")
                        logger.error(f"Missing prerequisites for artificial pitch drawing in job {job_id}")
                        msg = "Cannot draw artificial pitch without transformed players, line of scrimmage, and direction."
                        errors.append({"stage": name, "message": msg})
                        await emit({"type": "error", "stage": name, "message": msg})
                        aborted = True
                        break
                    result = await asyncio.to_thread(
                        fn,
                        output_img_path=artificial_pitch,
                        player_positions=artifacts["transformed_players"],
                        x_los=artifacts["x_los"],
                        los_yards=artifacts["los_yards"],
                        direction="left" if artifacts["direction"] == 1 else "right"
                    )
                    if result is None:
                        logger.error(f"Artificial pitch drawing failed for job {job_id}")
                        msg = "Artificial pitch drawing failed."
                        errors.append({"stage": name, "message": msg})
                        await emit({"type": "error", "stage": name, "message": msg})
                        aborted = True
                        break
                    artifacts["artificial_pitch_path"] = artificial_pitch

                case _:
                    logger.error(f"Unknown stage {name} for job {job_id}")
                    msg = f"Unknown stage: {name}"
                    errors.append({"stage": name, "message": msg})
                    await emit({"type": "error", "stage": name, "message": msg})
                    aborted = True
                    break

            # Optional: notify stage completion
            await emit({
                "type": "stage",
                "stage": {"index": idx, "total": total, "name": name},
                "message": f"Completed stage: {name}",
            })

        except Exception as e:
            logger.error(f"Exception in stage {name} for job {job_id}: {e}")
            # Catch exceptions from the stage and continue to terminal emit
            tb = traceback.format_exc()
            errors.append({"stage": name, "message": str(e), "traceback": tb})
            await emit({"type": "error", "stage": name, "message": str(e)})
            aborted = True
            break

    # Always send a terminal event
    status = "ok" if not aborted and len(errors) == 0 else "error"
    await emit({
        "type": "result",
        "status": status,
        "errors": errors,
        "message": "Analyze-snap job completed." if status == "ok" else "Analyze-snap job ended with errors.",
    })
