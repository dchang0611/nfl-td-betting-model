from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from factor_definitions import factor_read, matched_factor_labels, public_factor_definitions


ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
BOARDS = ROOT / "boards"


def clean(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def records(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    available = [column for column in columns if column in frame.columns]
    return [
        {key: clean(value) for key, value in row.items()}
        for row in frame[available].to_dict("records")
    ]


def latest_board() -> Path:
    boards = sorted(BOARDS.glob("anytime_td_board_????_week_??.csv"))
    if not boards:
        fallback = ROOT / "outputs" / "latest_board.csv"
        if fallback.exists():
            return fallback
        raise FileNotFoundError("No saved anytime-touchdown board was produced.")
    return boards[-1]


def backtest_payload() -> dict:
    metrics_path = ROOT / "outputs" / "pregame_backtest_2025_metrics.csv"
    if not metrics_path.exists():
        metrics_path = ROOT / "backtests" / "pregame_backtest_2025_metrics.csv"
    summary_path = ROOT / "outputs" / "pregame_backtest_2025_summary.json"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    preferred = [
        "legacy_features", "legacy_plus_rookie_specialist", "availability",
        "rules_environment_only", "rookie_pathway_only", "history_2024_plus",
        "history_recency_weighted", "expert_role",
    ]
    if not metrics.empty and "variant" in metrics:
        metrics = metrics.loc[metrics["variant"].isin(preferred)].copy()
        metrics["display_order"] = metrics["variant"].map({name: i for i, name in enumerate(preferred)})
        metrics = metrics.sort_values("display_order").drop(columns="display_order")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return {
        "metrics": records(metrics, list(metrics.columns)),
        "summary": summary,
        "weeks": weekly_backtest_payload(),
    }


def weekly_backtest_payload() -> list[dict]:
    """Publish the selected model's fixed 2025 boards with actual results."""
    snapshot = ROOT / "backtests" / "pregame_backtest_2025_weeks.json"
    snapshot_dir = ROOT / "backtests" / "weekly_results_2025"
    source_paths = sorted((BOARDS / "history").glob("2025_week_??_legacy_features.csv"))
    split_snapshots = sorted(snapshot_dir.glob("week_??.json"))
    if not source_paths and split_snapshots:
        return [json.loads(path.read_text(encoding="utf-8")) for path in split_snapshots]
    if not source_paths and snapshot.exists():
        return json.loads(snapshot.read_text(encoding="utf-8"))

    weeks = []
    columns = [
        "board_rank", "player_id", "player_name", "position", "team",
        "opponent_team", "model_probability", "confidence", "played", "void",
        "scored_td", "td_count", "snap_share_ewm", "route_participation_ewm",
        "red_zone_opp_share_ewm", "inside10_opp_share_ewm",
        "goal_line_rush_share_ewm", "end_zone_target_share_ewm",
        "role_stability", "team_implied_total", "def_red_zone_td_rate_prior",
    ]
    for path in source_paths:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["board_rank"] = pd.to_numeric(frame["board_rank"], errors="coerce")
        frame = frame.loc[frame["board_rank"].le(60)].sort_values("board_rank").copy()
        for column in ["played", "void", "scored_td", "td_count"]:
            frame[column] = pd.to_numeric(frame.get(column), errors="coerce").fillna(0)
        week = int(pd.to_numeric(frame["week"], errors="coerce").dropna().iloc[0])
        top5 = frame.loc[frame["board_rank"].le(5)]
        top10 = frame.loc[frame["board_rank"].le(10)]

        def result_summary(sample: pd.DataFrame) -> dict:
            graded = sample.loc[sample["void"].ne(1)]
            return {
                "hits": int(graded["scored_td"].eq(1).sum()),
                "graded": int(len(graded)),
                "voids": int(sample["void"].eq(1).sum()),
            }

        weeks.append(
            {
                "season": 2025,
                "week": week,
                "top5": result_summary(top5),
                "top10": result_summary(top10),
                "rows": records(frame, columns),
            }
        )
    if weeks:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(json.dumps(weeks, separators=(",", ":")), encoding="utf-8")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for item in weeks:
            (snapshot_dir / f"week_{item['week']:02d}.json").write_text(
                json.dumps(item, separators=(",", ":")), encoding="utf-8"
            )
    return weeks


def board_history() -> list[dict]:
    history = []
    for path in sorted(BOARDS.glob("anytime_td_board_????_week_??.csv"), reverse=True):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        history.append(
            {
                "season": int(frame["target_season"].iloc[0]),
                "week": int(frame["target_week"].iloc[0]),
                "players": int(len(frame)),
                "generatedAt": clean(frame.get("generated_at_utc", pd.Series([None])).iloc[0]),
                "file": path.name,
            }
        )
    return history


def main() -> None:
    path = latest_board()
    frame = pd.read_csv(path).sort_values("ranking")
    matched = frame.apply(matched_factor_labels, axis=1)
    frame["matched_factors"] = matched.map(" | ".join)
    frame["factor_count"] = matched.map(len)
    frame["model_note"] = frame.apply(factor_read, axis=1)
    columns = [
        "ranking", "target_season", "target_week", "nfl_game_id", "kickoff",
        "player_id", "player_name", "headshot_url", "team", "opponent_team",
        "position", "model_group", "home_away", "model_probability", "confidence",
        "raw_model_probability", "ranking_score", "expert_rank_percentile", "ecr",
        "rookie_flag", "rookie_season", "draft_round", "draft_pick",
        "rookie_specialist_probability", "snapshot_status", "planned_snapshot_utc",
        "actual_snapshot_utc", "board_cutoff", "scheduled_kickoff",
        "model_note", "matched_factors", "factor_count", "games_prior", "depth_rank", "roster_status", "new_team_flag",
        "injury", "practice_status", "game_status", "availability_probability",
        "current_team_games", "role_stability", "role_score", "recent_opportunities",
        "high_value_role_score", "snap_share_ewm", "route_participation_ewm",
        "rush_share_ewm", "target_share_ewm", "red_zone_opp_share_ewm",
        "inside10_opp_share_ewm", "inside5_opp_share_ewm",
        "goal_line_rush_share_ewm", "end_zone_target_share_ewm",
        "team_implied_total", "spread", "game_total", "team_epa_per_play_prior",
        "team_success_rate_prior", "def_epa_allowed_prior",
        "def_success_allowed_prior", "def_red_zone_td_rate_prior",
        "player_epa_per_opp_ewm", "generated_at_utc",
    ]
    season = int(frame["target_season"].iloc[0])
    week = int(frame["target_week"].iloc[0])
    payload = {
        "season": season,
        "week": week,
        "label": f"{season} Week {week}",
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "factorDefinitions": public_factor_definitions(),
        "rows": records(frame, columns),
        "history": board_history(),
        "backtest": backtest_payload(),
    }
    data = SITE / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "board.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    frame.to_csv(data / "latest-board.csv", index=False)
    position_group = frame["position"].replace({"FB": "RB"})
    for position in ["RB", "WR", "TE", "QB"]:
        frame.loc[position_group.eq(position)].to_csv(
            data / f"latest-board-{position.lower()}.csv", index=False
        )
    print(f"Built website data for {season} Week {week}.")


if __name__ == "__main__":
    main()
