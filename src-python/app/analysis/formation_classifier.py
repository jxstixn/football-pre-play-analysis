import json
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from collections import defaultdict

PX_PER_YARD = 10.0


@dataclass
class Player:
    x: float
    y: float
    cls: str
    id: Optional[str] = None


@dataclass
class FormationResult:
    personnel: str
    lxr: Tuple[int, int]
    te_side: str
    off_flag: bool
    label: str
    details: Dict


def _depth_relative_to_los(x, x_los, offense_side):
    if offense_side.lower() == "left":
        return max(0.0, (x_los - x))
    else:
        return max(0.0, (x - x_los))


def _on_los_band(x, x_los, offense_side, tol_yards=0.5):
    return abs(x - x_los) <= tol_yards * PX_PER_YARD


def _is_backfield(x, x_los, offense_side, min_depth_yards=1.0):
    return _depth_relative_to_los(x, x_los, offense_side) >= min_depth_yards * PX_PER_YARD


def _compute_tackles(oline, x_los, offense_side):
    los_band = [p[0] if isinstance(p, tuple) else p for p in oline if
                _on_los_band((p[0] if isinstance(p, tuple) else p).x, x_los, offense_side, tol_yards=1.0)]
    if len(los_band) < 2:
        los_band = [p[0] if isinstance(p, tuple) else p for p in oline]
    if not los_band:
        raise ValueError("No offensive linemen provided; cannot define tackle box.")
    leftmost = min(los_band, key=lambda p: p.y)
    rightmost = max(los_band, key=lambda p: p.y)
    return leftmost, rightmost


def _is_attached_te(p, lt, rt, x_los, offense_side, los_tol_yards=0.5, lateral_tol_yards=1.0):
    if not _on_los_band(p.x, x_los, offense_side, tol_yards=los_tol_yards):
        return False, ""
    if abs(p.y - lt.y) <= lateral_tol_yards * PX_PER_YARD:
        return True, "left"
    if abs(p.y - rt.y) <= lateral_tol_yards * PX_PER_YARD:
        return True, "right"
    return False, ""


def _is_wing_hback(p, lt, rt, x_los, offense_side, min_depth_yards=0.5, max_depth_yards=2.5, lateral_tol_yards=1.5):
    depth = _depth_relative_to_los(p.x, x_los, offense_side)
    if depth < min_depth_yards * PX_PER_YARD or depth > max_depth_yards * PX_PER_YARD:
        return False, ""
    if abs(p.y - lt.y) <= lateral_tol_yards * PX_PER_YARD:
        return True, "left"
    if abs(p.y - rt.y) <= lateral_tol_yards * PX_PER_YARD:
        return True, "right"
    return False, ""


def _in_tackle_box_lateral(y, lt, rt, pad_yards=0.5):
    lo = min(lt.y, rt.y) - pad_yards * PX_PER_YARD
    hi = max(lt.y, rt.y) + pad_yards * PX_PER_YARD
    return lo <= y <= hi


def _classify_roles(players, x_los, offense_side):
    groups = defaultdict(list)
    for p in players:
        groups[p.cls].append(p)
    return groups


def _find_oline_players(players, x_los, offense_side):
    oline_players = [p for p in players if getattr(p, "cls", None) == "oline"]
    if len(oline_players) >= 5:
        return oline_players

    los_tol = 1.0 * PX_PER_YARD
    los_candidates = [p for p in players if abs(p.x - x_los) < los_tol]

    if len(los_candidates) < 7:
        los_candidates = sorted(players, key=lambda p: abs(p.x - x_los))[:7]

    sorted_by_y = sorted(los_candidates, key=lambda p: p.y)
    min_spread = float('inf')
    best_group = None
    for i in range(len(sorted_by_y) - 4):
        group = sorted_by_y[i:i + 5]
        spread = group[-1].y - group[0].y
        if spread < min_spread:
            min_spread = spread
            best_group = group

    if best_group is not None:
        for p in oline_players:
            if p not in best_group:
                best_group.append(p)
        return best_group

    return oline_players


def _find_wide_receivers(players, x_los, offense_side, side="left"):
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    oline_box = get_bounding_box(_find_oline_players(players, x_los, offense_side), padding=5.0 * PX_PER_YARD)
    _, _, min_y, max_y = oline_box
    if side == "left":
        candidates = [p for p in players if p.y < min_y]
    else:
        candidates = [p for p in players if p.y > max_y]
    return candidates


def _find_qb(players, x_los, offense_side):
    qb_candidates = [p for p in players if getattr(p, "cls", None) == "qb"]
    if qb_candidates:
        return qb_candidates
    backfield_candidates = [p for p in players if _is_backfield(p.x, x_los, offense_side, min_depth_yards=1.0)]
    if not backfield_candidates:
        return []
    qb = min(backfield_candidates, key=lambda p: _depth_relative_to_los(p.x, x_los, offense_side))
    return [qb]


def get_bounding_box(players, padding=0.0):
    min_x = min(p.x for p in players) - padding
    max_x = max(p.x for p in players) + padding
    min_y = min(p.y for p in players) - padding
    max_y = max(p.y for p in players) + padding
    return min_x, max_x, min_y, max_y


def players_in_box(players, box):
    min_x, max_x, min_y, max_y = box
    return [p for p in players if min_x <= p.x <= max_x and min_y <= p.y <= max_y]


def players_in_box_oline(players, x_los, offense_side):
    """
    Gibt eine Liste von Spielern im O-Line-Bereich zurück und klassifiziert TEs als 'TE left', 'TE left off', 'TE right', 'TE right off'
    abhängig von ihrer Position relativ zur O-Line und offense_side.
    """
    oLine_box = get_bounding_box(_find_oline_players(players, x_los, offense_side), padding=5.0 * PX_PER_YARD)
    strict_oLine_box = get_bounding_box(_find_oline_players(players, x_los, offense_side))

    oline = players_in_box(players, oLine_box)
    min_x, max_x, min_y, max_y = strict_oLine_box

    result = []
    result.extend(oline)
    for p in oline:
        if p.cls in ("skill"):
            if offense_side == "left":
                if p.y > max_y and p.x < min_x:
                    result.append((p, "TE right off"))
                elif p.y > max_y and p.x > min_x:
                    result.append((p, "TE right"))
                elif p.y < min_y and p.x > min_x:
                    result.append((p, "TE left"))
                elif p.x < min_x and p.y < min_y:
                    result.append((p, "TE left off"))
            elif offense_side == "right":
                if p.y > max_y and p.x > max_x:
                    result.append((p, "TE left off"))
                elif p.y > max_y and p.x < max_x:
                    result.append((p, "TE left"))
                elif p.y < min_y and p.x < max_x:
                    result.append((p, "TE right"))
                elif p.x > max_x and p.y < min_y:
                    result.append((p, "TE right off"))
            if result and isinstance(result[-1], tuple) and result[-1][0] == p:
                result.remove(p)

    return result


def map_transformed_to_filtered_positions(transformed_player_positions):
    allowed_classes = {"skill", "oline", "qb"}
    player_positions = []
    class_counters = {}
    for player in transformed_player_positions:
        cls = str(player['class'])
        if cls not in allowed_classes:
            continue  # nur erlaubte Klassen
        x, y = player['transformed_position']
        class_counters.setdefault(cls, 0)
        class_counters[cls] += 1
        player_id = f"{cls}{class_counters[cls]}"
        player_positions.append({
            "x": x,
            "y": y,
            "cls": cls,
            "id": player_id
        })
    return player_positions


def classify_formation(formation_result_path: str, player_positions, x_los, offense_side="left"):
    filtered_players = map_transformed_to_filtered_positions(player_positions)
    ps = [Player(**p) if not isinstance(p, Player) else p for p in filtered_players]

    groups = _classify_roles(ps, x_los, offense_side)
    # oline = groups.get("oline", [])
    skill = groups.get("skill", [])
    qb = groups.get("qb", [])

    # compute the players inside the o-line box
    oline = players_in_box_oline(ps, x_los, offense_side)

    # compute the wide receiver box on the left and right side of the o-line box
    leftWR_box = get_bounding_box(_find_wide_receivers(ps, x_los, offense_side, side="left"), padding=5.0 * PX_PER_YARD)
    rightWR_box = get_bounding_box(_find_wide_receivers(ps, x_los, offense_side, side="right"),
                                   padding=5.0 * PX_PER_YARD)

    leftWR = players_in_box(ps, leftWR_box)
    rightWR = players_in_box(ps, rightWR_box)

    # compute the running backs
    qb_box = get_bounding_box(qb, padding=10.5 * PX_PER_YARD)
    qb = [p for p in players_in_box(ps, qb_box) if getattr(p, "cls", None) in ("qb", "skill") and p not in oline]

    lt, rt = _compute_tackles(oline, x_los, offense_side)

    attached_tes = []
    off_tes = []
    rbs = []
    wrs = []

    used_ids = set()

    if len(oline) > 5:
        for p in oline[5:]:
            attached_tes.append((p, ""))
            used_ids.add(id(p))

    for p in skill:
        if id(p) in used_ids:
            continue
        in_qb_box = any(
            abs(p.x - qb_p.x) <= 10.0 * PX_PER_YARD and abs(p.y - qb_p.y) <= 10.0 * PX_PER_YARD
            for qb_p in qb
        )
        if in_qb_box and all(id(p) != id(qb_p) for qb_p in qb):
            rbs.append(p)
            used_ids.add(id(p))
        else:
            wrs.append(p)

    num_te_attached = max(0, len(oline) - 5)
    num_rb = max(0, len(qb) - 1)
    num_wr_left = len(leftWR)
    num_wr_right = len(rightWR)
    outside_players = wrs[:]
    for p, _side in off_tes:
        if not _in_tackle_box_lateral(p.y, lt, rt):
            outside_players.append(p)
    offense_all = oline + qb + skill
    if not offense_all:
        center_y = (lt.y + rt.y) / 2.0
    else:
        ys = sorted([p.y if hasattr(p, "y") else p[0].y for p in offense_all])
        mid = len(ys) // 2
        center_y = ys[mid] if len(ys) % 2 == 1 else 0.5 * (ys[mid - 1] + ys[mid])
    te_sides = [side for _, side in attached_tes]
    if len(set(te_sides)) == 2:
        te_side = "both"
    elif len(te_sides) == 1:
        te_side = te_sides[0]
    else:
        te_side = ""
    off_flag = len(off_tes) > 0
    personnel = f"{num_rb}{num_te_attached}"
    lxr = (num_wr_left, num_wr_right)
    parts = [f"{personnel} – {lxr[0]}x{lxr[1]}"]
    for p in oline:
        if isinstance(p, tuple) and len(p) == 2 and p[1].startswith("TE"):
            parts.append(p[1])
    if te_side:
        parts.append(f"TE {te_side}")
    if off_flag:
        parts.append("OFF")
    label = " ".join(parts)
    details = {
        "num_rb": num_rb,
        "num_te_attached": num_te_attached,
        "left_receivers": num_wr_left,
        "right_receivers": num_wr_right,
        "te_side": te_side,
        "off_flag": off_flag,
        "attached_te_ids": [getattr(p, "id", None) for p, _ in attached_tes],
        "off_te_ids": [getattr(p, "id", None) for p, _ in off_tes],
        "rb_ids": [getattr(p, "id", None) for p in rbs],
        "wr_ids": [getattr(p, "id", None) for p in wrs],
        "center_y_used": center_y,
        "lt_y": lt.y,
        "rt_y": rt.y,
        "oline_ids": [getattr(p, "id", None) for p in oline],
        "leftWR_ids": [getattr(p, "id", None) for p in leftWR],
        "rightWR_ids": [getattr(p, "id", None) for p in rightWR],
        "qb_ids": [getattr(p, "id", None) for p in qb],
    }

    final_json = {
        "personnel": personnel,
        "lxr": lxr,
        "te_side": te_side,
        "off_flag": off_flag,
        "label": label,
        "details": details
    }

    with open(formation_result_path, "w") as f:
        json.dump(final_json, f, indent=4)

    return FormationResult(personnel, lxr, te_side, off_flag, label, details)
