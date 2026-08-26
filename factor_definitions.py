"""Shared factor definitions for NFL board explanations and confluence views."""

from __future__ import annotations

import math
from collections.abc import Mapping


FACTOR_DEFINITIONS = (
    {"key": "snap_route_volume", "label": "Snap / route volume", "fields": ("snap_share_ewm", "route_participation_ewm"), "threshold": 0.75, "aggregation": "max", "description": "At least 75% snap share or route participation."},
    {"key": "red_zone_role", "label": "Red-zone role", "fields": ("red_zone_opp_share_ewm",), "threshold": 0.20, "aggregation": "max", "description": "At least 20% of team red-zone opportunities."},
    {"key": "inside10_role", "label": "Inside-10 role", "fields": ("inside10_opp_share_ewm",), "threshold": 0.20, "aggregation": "max", "description": "At least 20% of team opportunities inside the 10."},
    {"key": "goal_line_end_zone_role", "label": "Goal-line / end-zone role", "fields": ("goal_line_rush_share_ewm", "end_zone_target_share_ewm"), "threshold": 0.25, "aggregation": "max", "description": "At least 25% goal-line rush share or end-zone target share."},
    {"key": "stable_role", "label": "Stable role", "fields": ("role_stability",), "threshold": 0.75, "aggregation": "max", "description": "Role-stability score of at least 75%."},
    {"key": "strong_scoring_environment", "label": "Strong scoring environment", "fields": ("team_implied_total",), "threshold": 24.0, "aggregation": "max", "description": "Team implied total of at least 24 points."},
    {"key": "vulnerable_red_zone_defense", "label": "Vulnerable red-zone defense", "fields": ("def_red_zone_td_rate_prior",), "threshold": 0.22, "aggregation": "max", "description": "Opponent prior red-zone TD rate of at least 22%."},
)


def public_factor_definitions() -> list[dict]:
    return [{**factor, "fields": list(factor["fields"])} for factor in FACTOR_DEFINITIONS]


def factor_value(row: Mapping, factor: Mapping) -> float | None:
    values = []
    for field in factor["fields"]:
        try:
            value = float(row.get(field))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return max(values) if values else None


def matched_factor_labels(row: Mapping) -> list[str]:
    return [
        factor["label"]
        for factor in FACTOR_DEFINITIONS
        if (value := factor_value(row, factor)) is not None and value >= factor["threshold"]
    ]


def factor_read(row: Mapping) -> str:
    labels = matched_factor_labels(row)
    if not labels:
        return f"0/{len(FACTOR_DEFINITIONS)} factors matched"
    return f"{len(labels)}/{len(FACTOR_DEFINITIONS)} | " + " | ".join(labels)
