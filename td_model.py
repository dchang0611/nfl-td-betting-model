#!/usr/bin/env python3
"""Build the hosted NFL anytime-touchdown probability board.

This is the production entry point.  It reuses the leakage-safe historical
feature engine in ``anytime_td_model_v1.py``, appends a synthetic pregame row
for every eligible player on the upcoming slate, trains position-specific
models, and saves the exact scored board for publishing and later grading.
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

import anytime_td_model_v1 as core


ROOT = Path(__file__).resolve().parent
TRAIN_START_SEASON = int(os.getenv("TRAIN_START_SEASON", "2022"))
TARGET_DATE = os.getenv("TARGET_DATE", date.today().isoformat())
TARGET_SEASON = os.getenv("TARGET_SEASON", "").strip()
TARGET_WEEK = os.getenv("TARGET_WEEK", "").strip()
MODEL_TYPE = os.getenv("MODEL_TYPE", "logistic").strip().lower()
BOARD_SIZE = int(os.getenv("BOARD_SIZE", "60"))

EVENT_COLUMNS = [
    "player_name_pbp", "opportunities", "rush_attempts_pbp", "targets_pbp",
    "red_zone_opps", "inside10_opps", "inside5_opps", "goal_line_rushes",
    "red_zone_targets", "end_zone_targets", "receiving_opps", "rush_tds", "receiving_tds",
    "player_epa", "air_yards_pbp", "qb_designed_rushes", "td_count",
    "scored_td",
]


def first_column(frame: pd.DataFrame, names: Iterable[str], default=np.nan) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series(default, index=frame.index)


def regular_season(frame: pd.DataFrame) -> pd.DataFrame:
    for column in ("game_type", "season_type"):
        if column in frame.columns:
            return frame.loc[frame[column].astype(str).eq("REG")].copy()
    return frame.copy()


def load_schedules() -> pd.DataFrame:
    import nflreadpy as nfl

    schedules = core._to_pandas(nfl.load_schedules(True))
    schedules = regular_season(schedules)
    schedules["season"] = pd.to_numeric(schedules["season"], errors="coerce")
    schedules["week"] = pd.to_numeric(schedules["week"], errors="coerce")
    schedules["gameday"] = pd.to_datetime(
        first_column(schedules, ["gameday", "game_date"]), errors="coerce"
    )
    return schedules.dropna(subset=["season", "week", "gameday"]).copy()


def resolve_target(schedules: pd.DataFrame) -> tuple[int, int]:
    if TARGET_SEASON and TARGET_WEEK:
        return int(TARGET_SEASON), int(TARGET_WEEK)

    as_of = pd.Timestamp(TARGET_DATE)
    future = schedules.loc[schedules["gameday"].ge(as_of)].sort_values(
        ["gameday", "season", "week"]
    )
    if future.empty:
        latest = schedules.sort_values(["season", "week"]).iloc[-1]
        return int(latest["season"]), int(latest["week"])
    next_game = future.iloc[0]
    return int(next_game["season"]), int(next_game["week"])


def completed_history_seasons(schedules: pd.DataFrame, target_season: int, target_week: int) -> list[int]:
    played = schedules.loc[
        pd.to_numeric(first_column(schedules, ["home_score"]), errors="coerce").notna()
        | pd.to_numeric(first_column(schedules, ["result"]), errors="coerce").notna()
    ].copy()
    played = played.loc[
        (played["season"] < target_season)
        | ((played["season"] == target_season) & (played["week"] < target_week))
    ]
    end = int(played["season"].max()) if not played.empty else target_season - 1
    if end < TRAIN_START_SEASON:
        raise RuntimeError("Not enough completed seasons are available to train the model.")
    return list(range(TRAIN_START_SEASON, end + 1))


def load_history(
    seasons: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import nflreadpy as nfl

    print(f"Loading completed NFL history for {seasons[0]}-{seasons[-1]}...")
    # Pull one season at a time and keep only columns used by the feature engine.
    # This avoids holding four full-width play-by-play releases in runner memory.
    pbp_columns = {
        "season_type", "season", "week", "game_id", "posteam", "defteam",
        "yardline_100", "epa", "success", "rush_attempt", "pass_attempt",
        "drive",
        "touchdown", "rush_touchdown", "pass_touchdown", "rusher_player_id",
        "rusher_id", "rusher_player_name", "rusher", "receiver_player_id",
        "receiver_id", "receiver_player_name", "receiver", "air_yards",
        "qb_scramble", "position", "game_seconds_remaining",
    }
    pbp_parts = []
    stats_parts = []
    snap_parts = []
    for season in seasons:
        raw_pbp = nfl.load_pbp(season)
        if hasattr(raw_pbp, "select"):
            raw_pbp = raw_pbp.select([c for c in raw_pbp.columns if c in pbp_columns])
        pbp_parts.append(core._to_pandas(raw_pbp))
        stats_parts.append(
            core._to_pandas(nfl.load_player_stats(season, summary_level="week"))
        )
        try:
            snap_parts.append(core._to_pandas(nfl.load_snap_counts(season)))
        except Exception as exc:
            print(f"Snap counts unavailable for {season}; using role proxies ({exc}).")
    pbp = pd.concat(pbp_parts, ignore_index=True, sort=False)
    stats = pd.concat(stats_parts, ignore_index=True, sort=False)
    snaps = pd.concat(snap_parts, ignore_index=True, sort=False) if snap_parts else pd.DataFrame()
    if not snaps.empty:
        snaps = regular_season(snaps)
        players = core._to_pandas(nfl.load_players())
        player_map = players[[c for c in ["pfr_id", "gsis_id"] if c in players.columns]].copy()
        if {"pfr_id", "gsis_id"}.issubset(player_map.columns):
            player_map = player_map.dropna().drop_duplicates("pfr_id")
            snaps = snaps.merge(
                player_map, left_on="pfr_player_id", right_on="pfr_id", how="left"
            )
        snaps["player_id"] = first_column(snaps, ["gsis_id", "player_id"])
        snaps["offense_snaps"] = pd.to_numeric(
            first_column(snaps, ["offense_snaps"], 0), errors="coerce"
        ).fillna(0)
        snaps = snaps.loc[snaps["player_id"].notna() & snaps["offense_snaps"].gt(0)].copy()
    return pbp, stats, core._to_pandas(nfl.load_schedules(seasons)), snaps


def current_candidates(target_season: int) -> pd.DataFrame:
    import nflreadpy as nfl

    roster = core._to_pandas(nfl.load_rosters(target_season))
    roster["player_id"] = first_column(roster, ["gsis_id", "player_id"])
    roster["player_name"] = first_column(
        roster, ["full_name", "player_name", "player_display_name"]
    )
    roster["position"] = first_column(
        roster, ["position", "depth_chart_position"], "UNK"
    ).astype(str).str.upper().replace({"HB": "RB"})
    roster["team"] = core.normalize_team(first_column(roster, ["team", "club_code"]))
    roster["roster_status"] = first_column(roster, ["status"], "Unknown").astype(str)
    roster["headshot_url"] = first_column(roster, ["headshot_url"], np.nan)
    valid = set().union(*core.POSITION_GROUPS.values())
    roster = roster.loc[
        roster["position"].isin(valid) & roster["player_id"].notna() & roster["team"].notna()
    ].copy()
    inactive = roster["roster_status"].str.contains(
        r"practice|reserve|injured reserve|suspend|retired|waived", case=False, na=False
    )
    roster = roster.loc[~inactive].copy()

    # Current depth charts narrow the board to realistic game-day candidates.
    try:
        depth = core._to_pandas(nfl.load_depth_charts(target_season))
        depth["player_id"] = first_column(depth, ["gsis_id", "player_id"])
        depth["depth_rank"] = pd.to_numeric(
            first_column(depth, ["pos_rank", "depth_team"]), errors="coerce"
        )
        if "dt" in depth.columns:
            depth["dt"] = pd.to_datetime(depth["dt"], errors="coerce", utc=True)
            depth = depth.sort_values("dt").drop_duplicates("player_id", keep="last")
        else:
            depth = depth.sort_values(
                [c for c in ["season", "week"] if c in depth.columns]
            ).drop_duplicates("player_id", keep="last")
        roster = roster.merge(depth[["player_id", "depth_rank"]], on="player_id", how="left")
        roster = roster.loc[roster["depth_rank"].le(3) | roster["depth_rank"].isna()].copy()
    except Exception as exc:
        print(f"Depth charts unavailable; using active roster candidates ({exc}).")
        roster["depth_rank"] = np.nan

    return roster.drop_duplicates("player_id", keep="last")


def target_games(schedules: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    games = schedules.loc[
        schedules["season"].eq(season) & schedules["week"].eq(week)
    ].copy()
    if games.empty:
        raise RuntimeError(f"No regular-season schedule found for {season} Week {week}.")
    games["home_team"] = core.normalize_team(first_column(games, ["home_team"]))
    games["away_team"] = core.normalize_team(first_column(games, ["away_team"]))
    return games


def make_forward_rows(
    roster: pd.DataFrame, games: pd.DataFrame, season: int, week: int
) -> pd.DataFrame:
    home = games[["home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent_team"}
    )
    away = games[["away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent_team"}
    )
    slate = pd.concat([home, away], ignore_index=True)
    forward = roster.merge(slate, on="team", how="inner")
    forward["season"] = season
    forward["week"] = week
    forward["game_id"] = f"{season}_{week:02d}_" + forward["team"].astype(str)
    forward["posteam"] = forward["team"]
    forward["defteam"] = forward["opponent_team"]
    forward["player_name_pbp"] = forward["player_name"]
    for column in EVENT_COLUMNS:
        if column not in forward.columns:
            forward[column] = np.nan if column == "scored_td" else 0.0
    forward["is_forward"] = True
    return forward


def extend_team_week(team_week: pd.DataFrame, games: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    teams = pd.unique(pd.concat([games["home_team"], games["away_team"]], ignore_index=True))
    future = pd.DataFrame({"season": season, "week": week, "team": teams})
    for column in team_week.columns:
        if column not in future.columns:
            future[column] = np.nan
    return pd.concat([team_week, future[team_week.columns]], ignore_index=True)


def build_feature_frame(
    pbp: pd.DataFrame,
    stats: pd.DataFrame,
    snaps: pd.DataFrame,
    history_schedules: pd.DataFrame,
    all_schedules: pd.DataFrame,
    roster: pd.DataFrame,
    season: int,
    week: int,
    paths: core.Paths,
    save_dataset: bool = True,
    player_pbp_precomputed: pd.DataFrame | None = None,
    team_week_precomputed: pd.DataFrame | None = None,
    player_stats_prepared: pd.DataFrame | None = None,
    player_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    player_pbp = (
        core.build_player_week_from_pbp(pbp)
        if player_pbp_precomputed is None else player_pbp_precomputed.copy()
    )
    team_week = (
        core.build_team_week(pbp)
        if team_week_precomputed is None else team_week_precomputed.copy()
    )
    player_stats = (
        core.prepare_player_stats(stats)
        if player_stats_prepared is None else player_stats_prepared.copy()
    )
    if snaps.empty:
        historical = player_pbp.copy()
    else:
        appearances = snaps.copy()
        appearances["snap_player_name"] = first_column(appearances, ["player"])
        appearances["snap_position"] = first_column(appearances, ["position"], "UNK")
        appearances["snap_team"] = core.normalize_team(first_column(appearances, ["team"]))
        appearances["snap_opponent"] = core.normalize_team(first_column(appearances, ["opponent"]))
        appearances["offense_pct"] = pd.to_numeric(
            first_column(appearances, ["offense_pct"]), errors="coerce"
        )
        keys = ["season", "week", "game_id", "player_id"]
        appearances = appearances[
            keys + [
                "snap_player_name", "snap_position", "snap_team", "snap_opponent",
                "offense_snaps", "offense_pct",
            ]
        ].drop_duplicates(keys)
        historical = player_pbp.merge(appearances, on=keys, how="outer")
        historical["posteam"] = historical["posteam"].fillna(historical["snap_team"])
        historical["defteam"] = historical["defteam"].fillna(historical["snap_opponent"])
        historical["player_name_pbp"] = historical["player_name_pbp"].fillna(
            historical["snap_player_name"]
        )
        for column in EVENT_COLUMNS:
            if column != "player_name_pbp":
                historical[column] = pd.to_numeric(
                    historical[column], errors="coerce"
                ).fillna(0)
    historical = historical.merge(
        player_stats, on=["season", "week", "player_id"], how="left"
    )
    historical["team"] = core.normalize_team(historical["team"].fillna(historical["posteam"]))
    historical["opponent_team"] = core.normalize_team(
        historical["opponent_team"].fillna(historical["defteam"])
    )
    historical["player_name"] = historical["player_name"].fillna(
        historical["player_name_pbp"]
    )
    if "snap_position" in historical.columns:
        historical["position"] = historical["position"].fillna(historical["snap_position"])
    historical["position"] = historical["position"].fillna("UNK")
    historical["is_forward"] = False

    games = target_games(all_schedules, season, week)
    forward = make_forward_rows(roster, games, season, week)
    combined = pd.concat([historical, forward], ignore_index=True, sort=False)

    team_opps = combined.groupby(["season", "week", "team"])["opportunities"].transform("sum")
    opportunity_share = core.safe_div(combined["opportunities"], team_opps).fillna(0)
    actual_snap_share = pd.to_numeric(
        first_column(combined, ["offense_pct"]), errors="coerce"
    )
    actual_snap_share = actual_snap_share.where(actual_snap_share.le(1), actual_snap_share / 100)
    combined["snap_share"] = actual_snap_share.combine_first(opportunity_share).clip(0, 1)
    team_targets = combined.groupby(["season", "week", "team"])["targets_pbp"].transform("sum")
    combined["routes_run_proxy"] = np.maximum(
        combined["targets_pbp"].fillna(0), team_targets.fillna(0) * combined["snap_share"]
    )
    combined["route_participation"] = core.safe_div(
        combined["routes_run_proxy"], team_targets
    ).fillna(0).clip(0, 1)

    schedules = pd.concat([history_schedules, games], ignore_index=True, sort=False)
    schedules = schedules.drop_duplicates(
        [c for c in ["season", "week", "home_team", "away_team"] if c in schedules.columns],
        keep="last",
    )
    combined = core.add_schedule_context(combined, schedules)
    combined = core.add_player_history(combined)
    if player_metadata is not None and not player_metadata.empty:
        metadata = player_metadata.copy()
        metadata["player_id"] = first_column(metadata, ["gsis_id", "player_id"])
        keep = ["player_id"] + [
            column for column in [
                "rookie_season", "draft_year", "draft_round", "draft_pick",
                "college_name", "birth_date",
            ] if column in metadata.columns
        ]
        metadata = metadata[keep].dropna(subset=["player_id"]).drop_duplicates("player_id")
        combined = combined.merge(metadata, on="player_id", how="left", suffixes=("", "_meta"))
        for column in keep[1:]:
            meta = f"{column}_meta"
            if meta in combined:
                combined[column] = combined.get(column, pd.Series(np.nan, index=combined.index)).fillna(combined[meta])
                combined.drop(columns=meta, inplace=True)
    if "rookie_season" not in combined and "rookie_year" in combined:
        combined["rookie_season"] = combined["rookie_year"]
    rookie_season = pd.to_numeric(first_column(combined, ["rookie_season", "rookie_year"]), errors="coerce")
    combined["rookie_flag"] = rookie_season.eq(pd.to_numeric(combined["season"], errors="coerce")).astype(int)
    combined["draft_round"] = pd.to_numeric(first_column(combined, ["draft_round"]), errors="coerce")
    combined["draft_pick"] = pd.to_numeric(first_column(combined, ["draft_pick"]), errors="coerce")
    combined = core.add_team_priors(
        combined, extend_team_week(team_week, games, season, week)
    )
    combined = core.merge_optional_scheme(combined, None)
    combined["position_group_model"] = combined["position"].map(core.get_group)
    combined = combined.loc[combined["position_group_model"].notna()].copy()
    combined = combined.sort_values(["season", "week", "game_id", "player_id"])

    if save_dataset:
        dataset = paths.processed / "player_game_features.parquet"
        combined.to_parquet(dataset, index=False)
        print(f"Saved {len(combined):,} historical and forward rows to {dataset}")
    return combined


def kickoff_lookup(games: pd.DataFrame) -> pd.DataFrame:
    game_id = first_column(games, ["game_id", "old_game_id"], "")
    gametime = first_column(games, ["gametime"], "00:00").fillna("00:00").astype(str)
    kickoff_local = pd.to_datetime(
        games["gameday"].dt.strftime("%Y-%m-%d") + " " + gametime,
        errors="coerce",
    )
    # nflverse schedule `gametime` is Eastern; publish an absolute UTC timestamp
    # so each dashboard viewer sees the correct local kickoff time.
    kickoff = kickoff_local.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="shift_forward"
    ).dt.tz_convert("UTC")
    home = pd.DataFrame({"team": games["home_team"], "nfl_game_id": game_id, "kickoff": kickoff})
    away = pd.DataFrame({"team": games["away_team"], "nfl_game_id": game_id, "kickoff": kickoff})
    return pd.concat([home, away], ignore_index=True)


def explain_board(board: pd.DataFrame) -> pd.DataFrame:
    role_fields = ["rush_share_ewm", "target_share_ewm", "snap_share_ewm"]
    high_value = ["red_zone_opp_share_ewm", "inside10_opp_share_ewm", "inside5_opp_share_ewm"]
    board["role_score"] = board[[c for c in role_fields if c in board]].mean(axis=1)
    board["high_value_role_score"] = board[[c for c in high_value if c in board]].mean(axis=1)
    board["recent_opportunities"] = np.where(
        board["position_group_model"].eq("RB"),
        board["rush_share_ewm"],
        board["target_share_ewm"],
    )
    board["model_note"] = np.select(
        [
            board["high_value_role_score"].ge(0.30),
            board["role_score"].ge(0.25),
            board["games_prior"].lt(3),
        ],
        ["Strong goal-line/red-zone role", "Strong recent workload", "Limited NFL sample"],
        default="Matchup and scoring environment",
    )
    return board


def freeze_game_snapshots(
    board: pd.DataFrame, season: int, week: int, paths: core.Paths
) -> pd.DataFrame:
    """Freeze each game's full candidate set at the first run on/after T-minus 24h."""
    now_text = os.getenv("SNAPSHOT_AT_UTC", "").strip()
    now = pd.Timestamp(now_text) if now_text else pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    snapshot_dir = paths.root / "boards" / "snapshots" / f"{season}_week_{week:02d}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    pieces = []
    for game_id, game in board.groupby("nfl_game_id", dropna=False):
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(game_id or "unknown"))
        snapshot_path = snapshot_dir / f"{safe_id}.csv"
        if snapshot_path.exists():
            frozen = pd.read_csv(snapshot_path)
            frozen["snapshot_status"] = "frozen"
            pieces.append(frozen)
            continue
        candidate = game.copy()
        kickoff = pd.to_datetime(candidate["kickoff"].iloc[0], errors="coerce", utc=True)
        cutoff = kickoff - pd.Timedelta("24h") if pd.notna(kickoff) else pd.NaT
        candidate["planned_snapshot_utc"] = cutoff.isoformat() if pd.notna(cutoff) else None
        candidate["actual_snapshot_utc"] = None
        candidate["snapshot_status"] = "provisional"
        if pd.notna(kickoff) and now >= cutoff and now < kickoff:
            candidate["actual_snapshot_utc"] = now.isoformat()
            candidate["snapshot_status"] = "frozen"
            candidate.to_csv(snapshot_path, index=False)
        pieces.append(candidate)
    return pd.concat(pieces, ignore_index=True, sort=False)


def train_and_score(
    frame: pd.DataFrame,
    games: pd.DataFrame,
    season: int,
    week: int,
    paths: core.Paths,
    expert_ranks: pd.DataFrame | None = None,
) -> Path:
    eligible_history = frame.loc[
        ~frame["is_forward"].fillna(False)
        & frame["scored_td"].notna()
        & ((frame["season"] < season) | ((frame["season"] == season) & (frame["week"] < week)))
    ].sort_values(["season", "week"]).copy()
    train = eligible_history.loc[eligible_history["games_prior"].ge(2)].copy()
    score = frame.loc[frame["is_forward"].fillna(False)].copy()
    rookie_train_source = eligible_history.copy()
    rookie_score_source = score.copy()
    # The 2025 T-minus-24h ablation favored the established feature set. Keep
    # rule-era and draft fields available for monitoring without letting them
    # distort the production ranking until they validate out of sample.
    experimental = [
        "league_offensive_tds_prior", "league_drives_prior",
        "league_red_zone_trips_prior", "league_plays_prior",
        "league_start_field_position_prior", "dynamic_kickoff_era",
        "touchback_35_era", "both_teams_overtime_era",
        "rookie_flag", "draft_round", "draft_pick",
    ]
    for column in experimental:
        if column in train:
            train[column] = np.nan
    scored = []
    for group in core.POSITION_GROUPS:
        group_train = train.loc[train["position_group_model"].eq(group)].copy()
        group_score = score.loc[score["position_group_model"].eq(group)].copy()
        if group_score.empty or len(group_train) < 500:
            continue
        model, numeric, categorical = core.fit_calibrated(group_train, group, MODEL_TYPE)
        group_score["model_probability"] = core.predict_group(
            model, group_score, numeric, categorical
        )
        group_score["model_group"] = group
        group_score["confidence"] = core.confidence_score(group_score)
        scored.append(group_score)
        joblib.dump(
            {"model": model, "numeric": numeric, "categorical": categorical},
            paths.models / f"td_model_{group}_{season}_week_{week:02d}.joblib",
        )
    if not scored:
        raise RuntimeError("No forward predictions were generated.")

    board = pd.concat(scored, ignore_index=True)
    board["raw_model_probability"] = board["model_probability"]
    board["ranking_score"] = board["model_probability"]
    rookie_train = rookie_train_source.loc[
        rookie_train_source.get("rookie_flag", pd.Series(0, index=rookie_train_source.index)).eq(1)
    ].copy()
    rookie_score = rookie_score_source.loc[
        rookie_score_source.get("rookie_flag", pd.Series(0, index=rookie_score_source.index)).eq(1)
    ].copy()
    rule_fields = experimental[:8]
    for column in rule_fields:
        if column in rookie_train:
            rookie_train[column] = np.nan
        if column in rookie_score:
            rookie_score[column] = np.nan
    if len(rookie_train) >= 500 and not rookie_score.empty:
        rookie_model, rookie_num, rookie_cat = core.fit_calibrated(
            rookie_train.sort_values(["season", "week"]), "ROOKIE", MODEL_TYPE
        )
        rookie_score["rookie_specialist_probability"] = core.predict_group(
            rookie_model, rookie_score, rookie_num, rookie_cat
        )
        board = board.merge(
            rookie_score[["player_id", "rookie_specialist_probability"]],
            on="player_id", how="left",
        )
    if expert_ranks is not None and not expert_ranks.empty:
        import pregame

        board = pregame.attach_expert_rankings(board, expert_ranks)
    board = board.merge(kickoff_lookup(games), on="team", how="left")
    board = explain_board(board)
    board["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    board = freeze_game_snapshots(board, season, week, paths)
    board = board.sort_values(
        ["ranking_score", "confidence"], ascending=[False, False]
    ).head(BOARD_SIZE).reset_index(drop=True)
    board.insert(0, "ranking", np.arange(1, len(board) + 1))
    board.insert(1, "target_season", season)
    board.insert(2, "target_week", week)

    paths.root.joinpath("boards").mkdir(parents=True, exist_ok=True)
    output = paths.root / "boards" / f"anytime_td_board_{season}_week_{week:02d}.csv"
    board.to_csv(output, index=False)
    board.to_csv(paths.outputs / "latest_board.csv", index=False)
    print(f"Saved {len(board)} ranked players to {output}")
    return output


def run() -> Path:
    if MODEL_TYPE not in {"logistic", "histgb"}:
        raise ValueError("MODEL_TYPE must be logistic or histgb.")
    paths = core.Paths(ROOT)
    paths.ensure()
    schedules = load_schedules()
    season, week = resolve_target(schedules)
    print(f"Target slate: {season} Week {week}")
    seasons = completed_history_seasons(schedules, season, week)
    pbp, stats, history_schedules, snaps = load_history(seasons)
    games = target_games(schedules, season, week)
    try:
        import nflreadpy as nfl
        import pregame

        depth = core._to_pandas(nfl.load_depth_charts(season))
        roster = pregame.weekly_candidates(
            season, week, games, depth, core.Paths(ROOT).raw, freeze_friday=False
        )
        print(f"Using {len(roster)} roster/depth/injury-aware candidates.")
    except Exception as exc:
        print(f"Pregame availability layer unavailable; using current roster ({exc}).")
        roster = current_candidates(season)
    try:
        import nflreadpy as nfl
        player_metadata = core._to_pandas(nfl.load_players())
    except Exception as exc:
        print(f"Player draft metadata unavailable; rookie layer will be limited ({exc}).")
        player_metadata = None
    frame = build_feature_frame(
        pbp, stats, snaps, history_schedules, schedules, roster, season, week, paths,
        player_metadata=player_metadata,
    )
    try:
        import pregame

        expert_ranks = pregame.load_expert_rankings()
    except Exception as exc:
        print(f"Expert role rankings unavailable; using model plus injuries ({exc}).")
        expert_ranks = None
    return train_and_score(frame, games, season, week, paths, expert_ranks=expert_ranks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hosted NFL anytime-TD probability board")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="Build the next weekly board (default)")
    backtest = sub.add_parser("backtest", help="Replay a completed season")
    backtest.add_argument("--season", type=int, default=2025)
    backtest.add_argument("--train-end", type=int, default=2024)
    backtest.add_argument("--model-type", choices=["logistic", "histgb"], default="logistic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command in (None, "run"):
        run()
        return
    paths = core.Paths(ROOT)
    paths.ensure()
    import pregame

    pregame.run_fixed_board_backtest(
        ROOT,
        test_season=args.season,
        train_start=TRAIN_START_SEASON,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
