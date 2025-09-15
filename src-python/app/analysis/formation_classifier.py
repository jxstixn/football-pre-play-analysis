import json
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

PX_PER_YARD = 30.0


@dataclass
class Player:
    x: float
    y: float
    cls: str
    id: Optional[str] = None


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


def get_offense_side(player_positions):
    # check player classes by labels
    defense = [p for p in player_positions if p['class'] == 'defense']
    offense = [p for p in player_positions if p['class'] in ('skill', 'oline', 'qb')]

    if not defense or not offense:
        return "left"  # default to left if we can't determine
    avg_def_x = sum(p['transformed_position'][0] for p in defense) / len(defense)
    avg_off_x = sum(p['transformed_position'][0] for p in offense) / len(offense)

    return "left" if avg_off_x < avg_def_x else "right"


def find_five_oline_near_los(
        players,
        x_los: float,
        los_band_yards: float = 1.0,
):
    """
    Return exactly five players inside a LOS band that are closest together laterally.

    Rules:
    - Only consider players with |x - x_los| <= los_band_yards.
    - No expansion beyond the LOS band; if fewer than five are available, return [].
    - Among eligible players, select the contiguous window of five (sorted by y)
      with the smallest lateral spread (y_max - y_min). Ties break on smaller
      average |x - x_los|.
    """

    if not players:
        return []

    los_tol = los_band_yards * PX_PER_YARD
    los_candidates = [p for p in players if abs(p.x - x_los) <= los_tol]

    if len(los_candidates) < 5:
        return []

    candidates_sorted_by_y = sorted(los_candidates, key=lambda p: p.y)
    best_group = None
    best_key = None  # (spread_y, avg_abs_dx)

    for i in range(len(candidates_sorted_by_y) - 4):
        group = candidates_sorted_by_y[i:i + 5]
        spread_y = group[-1].y - group[0].y
        avg_abs_dx = sum(abs(p.x - x_los) for p in group) / 5.0
        key = (spread_y, avg_abs_dx)
        if best_key is None or key < best_key:
            best_key = key
            best_group = group

    return best_group if best_group is not None else []


def get_oline_backfield_box(oline_players, offense_side: str, pad_y: float = 2.5, pad_backfield: float = 4.0):
    """
    Build a bounding box around the provided five O-Line players with padding:
    - Top/Bottom (lateral, y): 2.5 yards on each side
    - Backfield (x, relative to offense direction): 4.0 yards only toward backfield

    The backfield is defined as the direction opposite the offense movement:
    - offense_side == "left"  → offense moves to decreasing x → backfield is +x
    - offense_side == "right" → offense moves to increasing x → backfield is -x

    Returns a tuple: (min_x, max_x, min_y, max_y)
    """
    if not oline_players:
        return 0.0, 0.0, 0.0, 0.0

    pad_y = pad_y * PX_PER_YARD
    pad_backfield = pad_backfield * PX_PER_YARD

    min_x = min(p.x for p in oline_players)
    max_x = max(p.x for p in oline_players)
    min_y = min(p.y for p in oline_players)
    max_y = max(p.y for p in oline_players)

    # Apply vertical padding on both sides
    min_y -= pad_y
    max_y += pad_y

    # Apply backfield padding on x only toward backfield
    if offense_side.lower() == "right":
        # Backfield is +x
        max_x += pad_backfield
    else:
        # Backfield is -x
        min_x -= pad_backfield

    return min_x, max_x, min_y, max_y


def find_tight_ends_in_oline_box(all_players, oline_players, offense_side: str, pad_y: float = 2.5,
                                 pad_backfield: float = 4.0, line_tol_yards: float = 0.5):
    """
    Return players inside the O-Line backfield box that are not already in the O-Line.

    - The O-Line box is computed from `oline_players` using `get_oline_backfield_box` with the
      provided paddings.
    - Excludes any player present in `oline_players`.
    - No class restriction - considers all player classes.
    """
    if not all_players or not oline_players:
        return []

    box = get_oline_backfield_box(oline_players, offense_side, pad_y=pad_y, pad_backfield=pad_backfield)
    min_x, max_x, min_y, max_y = box

    mid_y = 0.5 * (min_y + max_y)

    oline_ids = set(id(p) for p in oline_players)

    candidates = [
        p for p in all_players
        if id(p) not in oline_ids
           and (min_x <= p.x <= max_x and min_y <= p.y <= max_y)
    ]

    # Classify ON/OFF LOS by comparing to average O-Line x instead of x_los
    te_with_status = []
    avg_oline_x = sum(p.x for p in oline_players) / float(len(oline_players))
    line_tol_px = line_tol_yards * PX_PER_YARD
    for p in candidates:
        status = "ON" if abs(p.x - avg_oline_x) <= line_tol_px else "OFF"
        te_with_status.append((p, status))

    te_above = [p for p in te_with_status if p[0].y > mid_y]
    te_below = [p for p in te_with_status if p[0].y <= mid_y]

    return te_above, te_below  # Return below first, then above


def find_wide_receivers_by_oline_box(all_players, oline_players, oline_box):
    """
    Split wide receivers into two groups based on O-Line box y-bounds.

    - below_wrs: players with y < min_y of the O-Line box
    - above_wrs: players with y > max_y of the O-Line box
    - Excludes players already identified as O-Line
    - Filters to likely WR candidates by class 'skill'
    Returns (below_wrs, above_wrs)
    """
    if not all_players or not oline_players:
        return [], []

    min_x, max_x, min_y, max_y = oline_box
    oline_ids = set(id(p) for p in oline_players)

    below_wrs = [
        p for p in all_players
        if id(p) not in oline_ids
           and p.y < min_y
    ]

    above_wrs = [
        p for p in all_players
        if id(p) not in oline_ids
           and p.y > max_y
    ]

    return below_wrs, above_wrs


def find_qb_and_proximity_players(all_players, exclude_players=None, proximity_yards: float = 5.0):
    """
    Find the quarterback (cls == 'qb') and other players within a proximity radius around him.

    - exclude_players: iterable of players to ignore (e.g., oline, tes, wrs)
    - proximity_yards: circular radius in yards; converted via PX_PER_YARD
    Returns (qb, nearby_players)
    """
    if not all_players:
        return None, []

    exclude_ids = set(id(p) for p in (exclude_players or []))

    # Locate QB by class; fall back to nearest to LOS among backfield if not explicitly labeled
    qb_candidates = [p for p in all_players if getattr(p, "cls", None) == "qb" and id(p) not in exclude_ids]
    qb = qb_candidates[0] if qb_candidates else None

    if qb is None:
        return None, []

    radius_px = proximity_yards * PX_PER_YARD
    r2 = radius_px * radius_px

    nearby = []
    for p in all_players:
        if id(p) in exclude_ids or id(p) == id(qb):
            continue
        dx = p.x - qb.x
        dy = p.y - qb.y
        if (dx * dx + dy * dy) <= r2:
            nearby.append(p)

    return qb, nearby


@dataclass
class FormationResult:
    personnel: str  # z.B. "21" (2 Fullbacks/RBs, 1 TE)
    lxr: Tuple[int, int]  # z.B. (2, 3)
    label: str  # z.B. "21 – 2x3 TE left"
    los: float
    details: Dict

    def __str__(self):
        return f"NewFormationResult(personnel={self.personnel}, lxr={self.lxr}, label={self.label} los={self.los}, details={json.dumps(self.details, indent=2)})"

    def to_json(self):
        return {
            "personnel": self.personnel,
            "lxr": self.lxr,
            "label": self.label,
            "los": self.los,
            "details": self.details
        }


def classify_formation(formation_result_path: str, player_positions, x_los, los_yards) -> FormationResult:
    offensive_player_count = sum(1 for p in player_positions if p['class'] in ('skill', 'oline', 'qb'))
    defensive_player_count = sum(1 for p in player_positions if p['class'] == 'defense')

    offense_side = get_offense_side(player_positions)

    filtered_players = map_transformed_to_filtered_positions(player_positions)
    ps = [Player(**p) if not isinstance(p, Player) else p for p in filtered_players]

    oline = find_five_oline_near_los(ps, x_los=x_los, los_band_yards=2.5)

    def _serialize(plrs):
        return [{"id": p.id, "cls": p.cls, "x": p.x, "y": p.y} for p in plrs]

    def _serialize_te(plrs_with_status):
        return [{"id": p.id, "cls": p.cls, "x": p.x, "y": p.y, "status": status} for p, status in plrs_with_status]

    if oline:
        oline_box = get_oline_backfield_box(oline, offense_side, pad_y=1.0, pad_backfield=2.0)
        te_above, te_below = find_tight_ends_in_oline_box(ps, oline, offense_side, pad_y=1.0, pad_backfield=2.0,
                                                          line_tol_yards=1.0)
        tes = te_above + te_below
        wrs_below, wrs_above = find_wide_receivers_by_oline_box(ps, oline, oline_box)
        excluded = (list(oline) + [p for p, _ in tes] + list(wrs_below) + list(wrs_above))
        qb, nearby = find_qb_and_proximity_players(ps, exclude_players=excluded, proximity_yards=5.0)

        assigned_ids = set(id(p) for p in excluded)
        if qb:
            assigned_ids.add(id(qb))
        for p in nearby:
            assigned_ids.add(id(p))
        remaining = [p for p in ps if id(p) not in assigned_ids]

        wrs_right = wrs_above if offense_side == "left" else wrs_below
        wrs_left = wrs_below if offense_side == "left" else wrs_above

        te_right = te_above if offense_side == "left" else te_below
        te_left = te_below if offense_side == "left" else te_above

        details = {
            "offensive_player_count": offensive_player_count,
            "defensive_player_count": defensive_player_count,
            "offense_side": offense_side,
            "oline_box": oline_box,
            "oline": _serialize(oline),
            "te_left": _serialize_te(te_left),
            "te_right": _serialize_te(te_right),
            "wr_left": _serialize(wrs_left),
            "wr_right": _serialize(wrs_right),
            "qb": {"id": qb.id, "cls": qb.cls, "x": qb.x, "y": qb.y} if qb else None,
            "qb_nearby": _serialize(nearby),
            "remaining": _serialize(remaining),
        }

        num_te = len(te_left) + len(te_right)
        num_rb = len(nearby)
        num_wr_left = len(wrs_left)
        num_wr_right = len(wrs_right)
        personnel = f"{num_rb}{num_te}"
        lxr = (num_wr_left, num_wr_right)

        label = f"{personnel} – {lxr[0]}x{lxr[1]}"
        te_sides = []
        if te_left:
            te_sides.append("left")
        if te_right:
            te_sides.append("right")
        if len(te_sides) == 2:
            label += " TE both"
        elif len(te_sides) == 1:
            label += f" TE {te_sides[0]}"
        off_flag = any("OFF" in status for _, status in tes)
        if off_flag:
            label += " [OFF]"

        result = FormationResult(personnel, lxr, label, los_yards, details)
    else:
        result = FormationResult(
            personnel="00",
            lxr=(0, 0),
            label="00 – 0x0",
            los=los_yards,
            details={
                "offensive_player_count": offensive_player_count,
                "defensive_player_count": defensive_player_count,
                "offense_side": offense_side,
                "oline_box": None,
                "oline": [],
                "te_left": [],
                "te_right": [],
                "wr_left": [],
                "wr_right": [],
                "qb": None,
                "qb_nearby": [],
                "remaining": _serialize(ps),
            }
        )

    with open(formation_result_path, "w") as f:
        json.dump(result.to_json(), f, indent=4)

    return result