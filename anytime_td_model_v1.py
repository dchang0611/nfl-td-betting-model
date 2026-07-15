#!/usr/bin/env python3
"""
Anytime Touchdown Model v1
==========================

Purpose
-------
Build a leakage-safe player-game dataset from nflverse data, train position-specific
TD classifiers on seasons before a holdout season, and replay the holdout season
week by week.

Default experiment
------------------
Train: 2022-2024 regular seasons
Test:  2025 regular season, strict walk-forward replay
Target: player scores at least one rushing or receiving TD

Important design choices
------------------------
* Historical features are shifted one game before rolling calculations.
* Team context belongs to the current team, not the player's prior team.
* Early-season team estimates shrink toward prior-season and league priors.
* Obvious redundant variables are omitted by design.
* RB, WR/TE, and QB models are trained separately.
* Public-data core works without coverage data; optional scheme CSV can be merged.
* Market odds are optional and are never required for the football-only backtest.

Install
-------
pip install nflreadpy pandas numpy scikit-learn pyarrow joblib

Examples
--------
python anytime_td_model_v1.py build-data --start-season 2022 --end-season 2025
python anytime_td_model_v1.py backtest --train-end 2024 --test-season 2025
python anytime_td_model_v1.py audit --dataset outputs/player_game_features.parquet

Optional scheme file schema
---------------------------
season,week,defteam,man_rate,two_high_rate,blitz_rate
Additional player-vs-coverage fields may be supplied if available:
player_id,player_vs_man_score,player_vs_zone_score
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
POSITION_GROUPS = {
    "RB": {"RB", "FB"},
    "REC": {"WR", "TE"},
    "QB": {"QB"},
}

# Deliberately compact and nonredundant. Each feature answers a distinct question.
BASE_FEATURES = [
    # Player availability / role
    "games_prior",
    "snap_share_ewm",
    "route_participation_ewm",
    "rush_share_ewm",
    "target_share_ewm",
    # High-value opportunity
    "red_zone_opp_share_ewm",
    "inside10_opp_share_ewm",
    "inside5_opp_share_ewm",
    "end_zone_target_share_ewm",
    # Team scoring and style, assigned to current team
    "team_implied_total",           # optional market input
    "spread",                       # optional market/game-script input
    "team_epa_per_play_prior",
    "team_success_rate_prior",
    "team_red_zone_pass_rate_prior",
    "team_seconds_per_play_prior",
    # Opponent quality
    "def_epa_allowed_prior",
    "def_success_allowed_prior",
    "def_red_zone_td_rate_prior",
    # Player efficiency summaries
    "player_epa_per_opp_ewm",
    "targets_per_route_ewm",
    # New-team / uncertainty controls
    "new_team_flag",
    "current_team_games",
    "role_stability",
    # Optional scheme layer
    "def_man_rate",
    "def_two_high_rate",
    "def_blitz_rate",
    "coverage_matchup_score",
    # League environment and explicit rules eras
    "league_offensive_tds_prior",
    "league_drives_prior",
    "league_red_zone_trips_prior",
    "league_plays_prior",
    "league_start_field_position_prior",
    "dynamic_kickoff_era",
    "touchback_35_era",
    "both_teams_overtime_era",
    # Rookie / low-sample information
    "rookie_flag",
    "draft_round",
    "draft_pick",
]

POSITION_EXTRA_FEATURES = {
    "RB": ["goal_line_rush_share_ewm", "receiving_opp_share_ewm"],
    "REC": ["air_yards_share_ewm", "red_zone_target_share_ewm"],
    "QB": ["qb_inside5_rush_share_ewm", "qb_designed_rush_share_ewm"],
    "ROOKIE": [
        "goal_line_rush_share_ewm", "receiving_opp_share_ewm",
        "air_yards_share_ewm", "red_zone_target_share_ewm",
        "qb_inside5_rush_share_ewm", "qb_designed_rush_share_ewm",
    ],
}

CATEGORICAL_FEATURES = ["position", "home_away"]


@dataclass
class Paths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def models(self) -> Path:
        return self.root / "models"

    def ensure(self) -> None:
        for p in (self.raw, self.processed, self.outputs, self.models):
            p.mkdir(parents=True, exist_ok=True)


def _to_pandas(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def load_nflverse(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load play-by-play, player stats, and schedules using nflreadpy."""
    try:
        import nflreadpy as nfl
    except ImportError as exc:
        raise SystemExit(
            "nflreadpy is not installed. Run: pip install nflreadpy pandas numpy "
            "scikit-learn pyarrow joblib"
        ) from exc

    print(f"Loading nflverse data for {seasons[0]}-{seasons[-1]}...")
    pbp = _to_pandas(nfl.load_pbp(seasons))
    stats = _to_pandas(nfl.load_player_stats(seasons, summary_level="week"))
    schedules = _to_pandas(nfl.load_schedules(seasons))
    return pbp, stats, schedules


def numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def first_existing(df: pd.DataFrame, names: Iterable[str], default=np.nan) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num.astype(float).div(den.replace(0, np.nan).astype(float))


def normalize_team(team: pd.Series) -> pd.Series:
    mapping = {"OAK": "LV", "SD": "LAC", "STL": "LA", "JAX": "JAC"}
    return team.astype(str).replace(mapping)


def prepare_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    p = pbp.copy()
    p = p.loc[first_existing(p, ["season_type"]).astype(str).eq("REG")].copy()
    p["season"] = numeric(p, "season").astype(int)
    p["week"] = numeric(p, "week").astype(int)
    p["posteam"] = normalize_team(first_existing(p, ["posteam"]))
    p["defteam"] = normalize_team(first_existing(p, ["defteam"]))
    p["yardline_100"] = numeric(p, "yardline_100", 99)
    p["epa"] = pd.to_numeric(first_existing(p, ["epa"]), errors="coerce")
    p["success"] = pd.to_numeric(first_existing(p, ["success"]), errors="coerce")
    p["play"] = (
        numeric(p, "rush_attempt")
        .add(numeric(p, "pass_attempt"))
        .gt(0)
        .astype(int)
    )
    p["rz_play"] = ((p["yardline_100"] <= 20) & (p["play"] == 1)).astype(int)
    p["rz_pass"] = ((p["yardline_100"] <= 20) & (numeric(p, "pass_attempt") == 1)).astype(int)
    p["rz_td"] = ((p["yardline_100"] <= 20) & (numeric(p, "touchdown") == 1)).astype(int)
    p["offensive_td"] = (
        numeric(p, "rush_touchdown").add(numeric(p, "pass_touchdown")).gt(0).astype(int)
    )
    return p


def build_player_week_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """Create player-week opportunity/TD summaries from PBP."""
    p = prepare_pbp(pbp)

    rush = p.loc[numeric(p, "rush_attempt").eq(1)].copy()
    rush["player_id"] = first_existing(rush, ["rusher_player_id", "rusher_id"])
    rush["player_name_pbp"] = first_existing(rush, ["rusher_player_name", "rusher"])
    rush["opportunity"] = 1
    rush["rush_opp"] = 1
    rush["target_opp"] = 0
    rush["red_zone_opp"] = rush["yardline_100"].le(20).astype(int)
    rush["inside10_opp"] = rush["yardline_100"].le(10).astype(int)
    rush["inside5_opp"] = rush["yardline_100"].le(5).astype(int)
    rush["goal_line_rush"] = rush["yardline_100"].le(5).astype(int)
    rush["red_zone_target"] = 0
    rush["end_zone_target"] = 0
    rush["receiving_opp"] = 0
    rush["rush_td"] = numeric(rush, "rush_touchdown")
    rush["rec_td"] = 0
    rush["player_epa"] = rush["epa"]
    rush["air_yards"] = 0.0
    rush["qb_designed_rush"] = (
        first_existing(rush, ["qb_scramble"], 0).fillna(0).astype(float).eq(0)
    ).astype(int)

    tgt = p.loc[numeric(p, "pass_attempt").eq(1) & first_existing(p, ["receiver_player_id", "receiver_id"]).notna()].copy()
    tgt["player_id"] = first_existing(tgt, ["receiver_player_id", "receiver_id"])
    tgt["player_name_pbp"] = first_existing(tgt, ["receiver_player_name", "receiver"])
    tgt["opportunity"] = 1
    tgt["rush_opp"] = 0
    tgt["target_opp"] = 1
    tgt["red_zone_opp"] = tgt["yardline_100"].le(20).astype(int)
    tgt["inside10_opp"] = tgt["yardline_100"].le(10).astype(int)
    tgt["inside5_opp"] = tgt["yardline_100"].le(5).astype(int)
    tgt["goal_line_rush"] = 0
    tgt["red_zone_target"] = tgt["yardline_100"].le(20).astype(int)
    # Air yards at/above distance to goal is a public-data proxy for an end-zone target.
    tgt["end_zone_target"] = numeric(tgt, "air_yards").ge(tgt["yardline_100"]).astype(int)
    tgt["receiving_opp"] = 1
    tgt["rush_td"] = 0
    tgt["rec_td"] = numeric(tgt, "pass_touchdown")
    tgt["player_epa"] = tgt["epa"]
    tgt["air_yards"] = numeric(tgt, "air_yards")
    tgt["qb_designed_rush"] = 0

    cols = [
        "season", "week", "game_id", "posteam", "defteam", "player_id",
        "player_name_pbp", "opportunity", "rush_opp", "target_opp",
        "red_zone_opp", "inside10_opp", "inside5_opp", "goal_line_rush", "red_zone_target",
        "end_zone_target", "receiving_opp", "rush_td", "rec_td",
        "player_epa", "air_yards", "qb_designed_rush",
    ]
    events = pd.concat([rush[cols], tgt[cols]], ignore_index=True)
    events = events.loc[events["player_id"].notna()].copy()

    agg = events.groupby(
        ["season", "week", "game_id", "posteam", "defteam", "player_id"],
        as_index=False,
    ).agg(
        player_name_pbp=("player_name_pbp", "first"),
        opportunities=("opportunity", "sum"),
        rush_attempts_pbp=("rush_opp", "sum"),
        targets_pbp=("target_opp", "sum"),
        red_zone_opps=("red_zone_opp", "sum"),
        inside10_opps=("inside10_opp", "sum"),
        inside5_opps=("inside5_opp", "sum"),
        goal_line_rushes=("goal_line_rush", "sum"),
        red_zone_targets=("red_zone_target", "sum"),
        end_zone_targets=("end_zone_target", "sum"),
        receiving_opps=("receiving_opp", "sum"),
        rush_tds=("rush_td", "sum"),
        receiving_tds=("rec_td", "sum"),
        player_epa=("player_epa", "sum"),
        air_yards_pbp=("air_yards", "sum"),
        qb_designed_rushes=("qb_designed_rush", "sum"),
    )
    agg["td_count"] = agg["rush_tds"] + agg["receiving_tds"]
    agg["scored_td"] = agg["td_count"].gt(0).astype(int)
    return agg


def build_team_week(pbp: pd.DataFrame) -> pd.DataFrame:
    p = prepare_pbp(pbp)
    plays = p.loc[p["play"].eq(1) & p["posteam"].notna()].copy()

    offense = plays.groupby(["season", "week", "posteam"], as_index=False).agg(
        team_plays=("play", "sum"),
        team_epa=("epa", "sum"),
        team_successes=("success", "sum"),
        red_zone_plays=("rz_play", "sum"),
        red_zone_passes=("rz_pass", "sum"),
        team_offensive_tds=("offensive_td", "sum"),
    ).rename(columns={"posteam": "team"})
    offense["team_epa_per_play"] = safe_div(offense["team_epa"], offense["team_plays"])
    offense["team_success_rate"] = safe_div(offense["team_successes"], offense["team_plays"])
    offense["team_red_zone_pass_rate"] = safe_div(offense["red_zone_passes"], offense["red_zone_plays"])

    # nflfastR has game_seconds_remaining; elapsed seconds / plays is a stable pace proxy.
    game_elapsed = 3600 - numeric(plays, "game_seconds_remaining", 1800)
    plays["elapsed"] = game_elapsed
    pace = plays.groupby(["season", "week", "posteam"], as_index=False).agg(
        max_elapsed=("elapsed", "max"), team_plays2=("play", "sum")
    ).rename(columns={"posteam": "team"})
    pace["team_seconds_per_play"] = safe_div(pace["max_elapsed"], pace["team_plays2"])
    offense = offense.merge(pace[["season", "week", "team", "team_seconds_per_play"]],
                            on=["season", "week", "team"], how="left")

    if "drive" in plays.columns:
        drives = plays.loc[pd.to_numeric(plays["drive"], errors="coerce").notna()].copy()
        drives["start_field_position"] = 100 - drives["yardline_100"]
        drive_level = drives.sort_values(["season", "week", "game_id", "posteam", "drive"]).groupby(
            ["season", "week", "game_id", "posteam", "drive"], as_index=False
        ).agg(
            start_field_position=("start_field_position", "first"),
            reached_red_zone=("rz_play", "max"),
        )
        possession = drive_level.groupby(["season", "week", "posteam"], as_index=False).agg(
            team_drives=("drive", "size"),
            team_red_zone_trips=("reached_red_zone", "sum"),
            team_start_field_position=("start_field_position", "mean"),
        ).rename(columns={"posteam": "team"})
        offense = offense.merge(possession, on=["season", "week", "team"], how="left")
    else:
        offense["team_drives"] = np.nan
        offense["team_red_zone_trips"] = np.nan
        offense["team_start_field_position"] = np.nan

    defense = plays.groupby(["season", "week", "defteam"], as_index=False).agg(
        def_plays=("play", "sum"),
        def_epa_allowed=("epa", "sum"),
        def_successes_allowed=("success", "sum"),
        def_rz_plays=("rz_play", "sum"),
        def_rz_tds=("rz_td", "sum"),
    ).rename(columns={"defteam": "team"})
    defense["def_epa_allowed"] = safe_div(defense["def_epa_allowed"], defense["def_plays"])
    defense["def_success_allowed"] = safe_div(defense["def_successes_allowed"], defense["def_plays"])
    defense["def_red_zone_td_rate"] = safe_div(defense["def_rz_tds"], defense["def_rz_plays"])

    return offense.merge(
        defense[["season", "week", "team", "def_epa_allowed", "def_success_allowed", "def_red_zone_td_rate"]],
        on=["season", "week", "team"], how="outer"
    )


def prepare_player_stats(stats: pd.DataFrame) -> pd.DataFrame:
    s = stats.copy()
    if "season_type" in s.columns:
        s = s.loc[s["season_type"].astype(str).eq("REG")].copy()
    s["season"] = numeric(s, "season").astype(int)
    s["week"] = numeric(s, "week").astype(int)
    s["player_id"] = first_existing(s, ["player_id", "gsis_id"])
    s["player_name"] = first_existing(s, ["player_display_name", "player_name"])
    s["position"] = first_existing(s, ["position", "position_group"]).astype(str)
    s["team"] = normalize_team(first_existing(s, ["recent_team", "team"])).astype(str)
    s["opponent_team"] = normalize_team(first_existing(s, ["opponent_team", "opponent"])).astype(str)

    keep = ["season", "week", "player_id", "player_name", "position", "team", "opponent_team"]
    optional = [
        "carries", "targets", "target_share", "air_yards_share", "wopr",
        "rushing_tds", "receiving_tds", "receptions", "receiving_yards",
        "rushing_yards", "fantasy_points", "pacr", "racr",
    ]
    for c in optional:
        if c in s.columns:
            keep.append(c)
    return s[keep].drop_duplicates(["season", "week", "player_id"])


def add_schedule_context(df: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    sch = schedules.copy()
    if "game_type" in sch.columns:
        sch = sch.loc[sch["game_type"].astype(str).eq("REG")].copy()
    sch["season"] = numeric(sch, "season").astype(int)
    sch["week"] = numeric(sch, "week").astype(int)
    sch["home_team"] = normalize_team(first_existing(sch, ["home_team"]))
    sch["away_team"] = normalize_team(first_existing(sch, ["away_team"]))

    # nflverse schedules frequently contain closing spread and total.
    sch["spread_line"] = pd.to_numeric(first_existing(sch, ["spread_line"]), errors="coerce")
    sch["total_line"] = pd.to_numeric(first_existing(sch, ["total_line"]), errors="coerce")

    home = sch[["season", "week", "home_team", "away_team", "spread_line", "total_line"]].copy()
    home.columns = ["season", "week", "team", "opponent_team_sch", "spread", "game_total"]
    home["home_away"] = "home"
    # nflverse spread_line is positive when the home team is favored.
    home["team_implied_total"] = home["game_total"] / 2 + home["spread"] / 2

    away = sch[["season", "week", "away_team", "home_team", "spread_line", "total_line"]].copy()
    away.columns = ["season", "week", "team", "opponent_team_sch", "home_spread", "game_total"]
    away["spread"] = -away["home_spread"]
    away.drop(columns="home_spread", inplace=True)
    away["home_away"] = "away"
    away["team_implied_total"] = away["game_total"] / 2 + away["spread"] / 2

    game_ctx = pd.concat([home, away], ignore_index=True)
    d = df.merge(game_ctx, on=["season", "week", "team"], how="left")
    d["dynamic_kickoff_era"] = d["season"].ge(2024).astype(int)
    d["touchback_35_era"] = d["season"].ge(2025).astype(int)
    d["both_teams_overtime_era"] = d["season"].ge(2025).astype(int)
    return d


def shifted_ewm(df: pd.DataFrame, group: str, col: str, halflife: float = 4.0) -> pd.Series:
    return df.groupby(group, group_keys=False)[col].apply(
        lambda x: x.shift(1).ewm(halflife=halflife, min_periods=1, adjust=False).mean()
    )


def add_player_history(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["player_id", "season", "week"]).copy()
    d["games_prior"] = d.groupby("player_id").cumcount()

    # Weekly shares: denominators are team-week opportunity totals.
    for col in ["rush_attempts_pbp", "targets_pbp", "red_zone_opps", "inside10_opps",
                "inside5_opps", "goal_line_rushes", "end_zone_targets", "receiving_opps"]:
        team_total = d.groupby(["season", "week", "team"])[col].transform("sum")
        d[col.replace("_pbp", "") + "_share"] = safe_div(d[col], team_total).fillna(0)

    d["player_epa_per_opp"] = safe_div(d["player_epa"], d["opportunities"]).fillna(0)
    d["targets_per_route"] = safe_div(d["targets_pbp"], d["routes_run_proxy"]).fillna(0)
    d["air_yards_share"] = safe_div(
        d["air_yards_pbp"], d.groupby(["season", "week", "team"])["air_yards_pbp"].transform("sum")
    ).fillna(0)
    d["red_zone_target_share"] = safe_div(
        d["red_zone_targets"],
        d.groupby(["season", "week", "team"])["red_zone_targets"].transform("sum")
    ).fillna(0)
    d["qb_inside5_rush_share"] = np.where(
        d["position"].eq("QB"), d["inside5_opps_share"], 0.0
    )
    d["qb_designed_rush_share"] = safe_div(
        d["qb_designed_rushes"],
        d.groupby(["season", "week", "team"])["qb_designed_rushes"].transform("sum")
    ).fillna(0)

    # Snap and routes are proxied if participation data is unavailable.
    # These remain distinct: snaps = plays with any opportunity proxy, routes = targets + team dropbacks proxy.
    for col in [
        "snap_share", "route_participation", "rush_attempts_share", "targets_share",
        "red_zone_opps_share", "inside10_opps_share", "inside5_opps_share",
        "goal_line_rushes_share", "end_zone_targets_share", "receiving_opps_share",
        "player_epa_per_opp", "targets_per_route", "air_yards_share",
        "red_zone_target_share", "qb_inside5_rush_share", "qb_designed_rush_share",
    ]:
        source = col
        out = {
            "rush_attempts_share": "rush_share_ewm",
            "targets_share": "target_share_ewm",
            "red_zone_opps_share": "red_zone_opp_share_ewm",
            "inside10_opps_share": "inside10_opp_share_ewm",
            "inside5_opps_share": "inside5_opp_share_ewm",
            "goal_line_rushes_share": "goal_line_rush_share_ewm",
            "end_zone_targets_share": "end_zone_target_share_ewm",
            "receiving_opps_share": "receiving_opp_share_ewm",
            "player_epa_per_opp": "player_epa_per_opp_ewm",
            "targets_per_route": "targets_per_route_ewm",
            "air_yards_share": "air_yards_share_ewm",
            "red_zone_target_share": "red_zone_target_share_ewm",
            "qb_inside5_rush_share": "qb_inside5_rush_share_ewm",
            "qb_designed_rush_share": "qb_designed_rush_share_ewm",
        }.get(col, f"{col}_ewm")
        d[out] = shifted_ewm(d, "player_id", source)

    # Role stability: lower rolling variability means more confidence.
    prior_role = d.groupby("player_id")["opportunities"].shift(1)
    rolling_std = prior_role.groupby(d["player_id"]).rolling(4, min_periods=2).std().reset_index(level=0, drop=True)
    rolling_mean = prior_role.groupby(d["player_id"]).rolling(4, min_periods=1).mean().reset_index(level=0, drop=True)
    d["role_stability"] = (1 / (1 + safe_div(rolling_std, rolling_mean).fillna(1))).clip(0, 1)

    d["previous_team"] = d.groupby("player_id")["team"].shift(1)
    d["new_team_flag"] = (
        d["previous_team"].notna() & d["team"].ne(d["previous_team"])
    ).astype(int)
    d["current_team_games"] = d.groupby(["player_id", "team"]).cumcount()
    return d


def add_team_priors(df: pd.DataFrame, team_week: pd.DataFrame) -> pd.DataFrame:
    """Create current-team priors with early-season shrinkage."""
    tw = team_week.sort_values(["team", "season", "week"]).copy()
    metrics = [
        "team_epa_per_play", "team_success_rate", "team_red_zone_pass_rate",
        "team_seconds_per_play", "def_epa_allowed", "def_success_allowed",
        "def_red_zone_td_rate",
    ]

    # Prior season team mean; fallback to prior season league mean.
    season_team = tw.groupby(["season", "team"], as_index=False)[metrics].mean()
    prior_team = season_team.copy()
    prior_team["season"] += 1
    prior_team = prior_team.rename(columns={m: f"{m}_prior_season" for m in metrics})
    tw = tw.merge(prior_team, on=["season", "team"], how="left")

    for m in metrics:
        current_prior = tw.groupby(["season", "team"])[m].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )
        league_week = tw.groupby(["season", "week"], as_index=False)[m].mean()
        league_week[f"{m}_league_prior"] = league_week.groupby("season")[m].transform(
            lambda values: values.shift(1).expanding(min_periods=1).mean()
        )
        tw = tw.merge(
            league_week[["season", "week", f"{m}_league_prior"]],
            on=["season", "week"], how="left",
        )
        old = tw[f"{m}_prior_season"].fillna(tw[f"{m}_league_prior"])
        games = tw.groupby(["season", "team"]).cumcount()
        # Empirical-Bayes style smooth transition, no arbitrary week cliffs.
        current_weight = games / (games + 4.0)
        tw[f"{m}_prior"] = current_weight * current_prior.fillna(old) + (1 - current_weight) * old

    off_cols = ["season", "week", "team"] + [
        "team_epa_per_play_prior", "team_success_rate_prior",
        "team_red_zone_pass_rate_prior", "team_seconds_per_play_prior",
    ]
    d = df.merge(tw[off_cols], on=["season", "week", "team"], how="left")

    def_cols = ["season", "week", "team", "def_epa_allowed_prior",
                "def_success_allowed_prior", "def_red_zone_td_rate_prior"]
    opp = tw[def_cols].rename(columns={"team": "opponent_team_sch"})
    d = d.merge(opp, on=["season", "week", "opponent_team_sch"], how="left")

    league_sources = {
        "team_offensive_tds": "league_offensive_tds_prior",
        "team_drives": "league_drives_prior",
        "team_red_zone_trips": "league_red_zone_trips_prior",
        "team_plays": "league_plays_prior",
        "team_start_field_position": "league_start_field_position_prior",
    }
    available = [column for column in league_sources if column in tw.columns]
    if available:
        weekly = tw.groupby(["season", "week"], as_index=False)[available].mean()
        for source in available:
            season_prior = weekly.groupby("season")[source].transform(
                lambda values: values.shift(1).expanding(min_periods=1).mean()
            )
            prior_lookup = (
                weekly.groupby("season", as_index=False)[source].mean()
                .assign(season=lambda x: x["season"] + 1)
                .rename(columns={source: "prior_value"})
            )
            weekly = weekly.merge(prior_lookup, on="season", how="left")
            weekly[league_sources[source]] = season_prior.fillna(weekly["prior_value"])
            weekly.drop(columns="prior_value", inplace=True)
        keep = ["season", "week"] + [league_sources[x] for x in available]
        d = d.merge(weekly[keep], on=["season", "week"], how="left")
    return d


def merge_optional_scheme(df: pd.DataFrame, scheme_csv: Optional[str]) -> pd.DataFrame:
    d = df.copy()
    for c in ["def_man_rate", "def_two_high_rate", "def_blitz_rate", "coverage_matchup_score"]:
        d[c] = np.nan
    if not scheme_csv:
        return d
    sc = pd.read_csv(scheme_csv)
    required = {"season", "week", "defteam"}
    missing = required - set(sc.columns)
    if missing:
        raise ValueError(f"Scheme CSV missing required columns: {sorted(missing)}")
    sc["defteam"] = normalize_team(sc["defteam"])
    d = d.drop(columns=["def_man_rate", "def_two_high_rate", "def_blitz_rate", "coverage_matchup_score"])
    d = d.merge(sc, left_on=["season", "week", "opponent_team_sch"],
                right_on=["season", "week", "defteam"], how="left")
    if "coverage_matchup_score" not in d.columns:
        player_man = d.get("player_vs_man_score", pd.Series(np.nan, index=d.index))
        player_zone = d.get("player_vs_zone_score", pd.Series(np.nan, index=d.index))
        d["coverage_matchup_score"] = (
            d.get("def_man_rate", 0) * player_man +
            (1 - d.get("def_man_rate", 0)) * player_zone
        )
    return d


def build_dataset(start_season: int, end_season: int, paths: Paths,
                  scheme_csv: Optional[str] = None) -> Path:
    seasons = list(range(start_season, end_season + 1))
    pbp, stats, schedules = load_nflverse(seasons)
    player_pbp = build_player_week_from_pbp(pbp)
    team_week = build_team_week(pbp)
    ps = prepare_player_stats(stats)

    d = player_pbp.merge(
        ps, left_on=["season", "week", "player_id"],
        right_on=["season", "week", "player_id"], how="left"
    )
    d["team"] = normalize_team(d["team"].fillna(d["posteam"]))
    d["opponent_team"] = normalize_team(d["opponent_team"].fillna(d["defteam"]))
    d["player_name"] = d["player_name"].fillna(d["player_name_pbp"])
    d["position"] = d["position"].fillna("UNK")

    # Public PBP does not contain every route/snap. Use conservative proxies unless
    # the user later merges participation data.
    team_opps = d.groupby(["season", "week", "team"])["opportunities"].transform("sum")
    d["snap_share"] = safe_div(d["opportunities"], team_opps).fillna(0)
    team_targets = d.groupby(["season", "week", "team"])["targets_pbp"].transform("sum")
    d["routes_run_proxy"] = np.maximum(d["targets_pbp"], team_targets * d["snap_share"])
    d["route_participation"] = safe_div(d["routes_run_proxy"], team_targets).fillna(0).clip(0, 1)

    d = add_schedule_context(d, schedules)
    d = add_player_history(d)
    d = add_team_priors(d, team_week)
    d = merge_optional_scheme(d, scheme_csv)

    valid_positions = set().union(*POSITION_GROUPS.values())
    d = d.loc[d["position"].isin(valid_positions)].copy()
    d = d.sort_values(["season", "week", "game_id", "player_id"])

    out = paths.processed / "player_game_features.parquet"
    d.to_parquet(out, index=False)
    d.to_csv(paths.processed / "player_game_features.csv", index=False)
    print(f"Saved {len(d):,} player-game rows to {out}")
    return out


def make_pipeline(model_type: str, numeric_features: list[str], categorical_features: list[str]):
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)),
    ])
    pre = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])

    if model_type == "logistic":
        estimator = LogisticRegression(
            C=0.35, penalty="l2", solver="liblinear", class_weight=None,
            max_iter=2000, random_state=RANDOM_STATE,
        )
    else:
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.045,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=60,
            l2_regularization=2.0,
            random_state=RANDOM_STATE,
        )
    return Pipeline([("pre", pre), ("model", estimator)])


def get_group(position: str) -> Optional[str]:
    for group, positions in POSITION_GROUPS.items():
        if position in positions:
            return group
    return None


def feature_list(group: str, df: pd.DataFrame) -> tuple[list[str], list[str]]:
    num = [f for f in BASE_FEATURES + POSITION_EXTRA_FEATURES[group] if f in df.columns]
    cat = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    # Drop completely missing optional inputs rather than manufacturing information.
    num = [f for f in num if df[f].notna().any()]
    return num, cat


def fit_calibrated(train: pd.DataFrame, group: str, model_type: str):
    num, cat = feature_list(group, train)
    X = train[num + cat]
    y = train["scored_td"].astype(int)
    if y.nunique() < 2:
        raise ValueError(f"Not enough target variation for {group}")

    base = make_pipeline(model_type, num, cat)
    # Time-respecting calibration split: oldest 80% fits, newest 20% calibrates.
    cutoff = max(1, int(len(train) * 0.80))
    fit_idx = np.arange(cutoff)
    cal_idx = np.arange(cutoff, len(train))
    fit_kwargs = {}
    if "_sample_weight" in train.columns:
        fit_kwargs["model__sample_weight"] = pd.to_numeric(
            train["_sample_weight"].iloc[fit_idx], errors="coerce"
        ).fillna(1.0).to_numpy()
    base.fit(X.iloc[fit_idx], y.iloc[fit_idx], **fit_kwargs)

    if len(cal_idx) >= 100 and y.iloc[cal_idx].nunique() == 2:
        try:
            from sklearn.frozen import FrozenEstimator

            calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
        except ImportError:
            # scikit-learn 1.5 compatibility; 1.6+ uses FrozenEstimator above.
            calibrated = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
        calibrated.fit(X.iloc[cal_idx], y.iloc[cal_idx])
        return calibrated, num, cat
    return base, num, cat


def predict_group(model, frame: pd.DataFrame, num: list[str], cat: list[str]) -> np.ndarray:
    return model.predict_proba(frame[num + cat])[:, 1]


def confidence_score(df: pd.DataFrame) -> pd.Series:
    sample = (np.log1p(df["games_prior"].fillna(0)) / np.log(18)).clip(0, 1)
    current = (df["current_team_games"].fillna(0) / 6).clip(0, 1)
    stable = df["role_stability"].fillna(0.5).clip(0, 1)
    new_team_penalty = 1 - 0.20 * df["new_team_flag"].fillna(0)
    scheme_known = np.where(df.get("coverage_matchup_score", pd.Series(np.nan, index=df.index)).notna(), 1.0, 0.9)
    return (100 * (0.35 * sample + 0.30 * current + 0.35 * stable) * new_team_penalty * scheme_known).clip(1, 99)


def evaluate_predictions(pred: pd.DataFrame, label: str) -> dict:
    y = pred["scored_td"].astype(int)
    p = pred["model_probability"].clip(1e-5, 1 - 1e-5)
    result = {
        "label": label,
        "n": int(len(pred)),
        "td_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else None,
    }
    for n in [5, 10, 20, 30]:
        daily = []
        for _, wk in pred.groupby(["season", "week"]):
            top = wk.nlargest(n, "model_probability")
            daily.append({"precision": top["scored_td"].mean(), "any_hit": int(top["scored_td"].sum() > 0)})
        result[f"top{n}_precision"] = float(np.mean([x["precision"] for x in daily]))
        result[f"top{n}_any_hit_rate"] = float(np.mean([x["any_hit"] for x in daily]))
    return result


def calibration_table(pred: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    d = pred.copy()
    d["prob_bin"] = pd.cut(d["model_probability"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    return d.groupby("prob_bin", observed=False).agg(
        n=("scored_td", "size"),
        avg_pred=("model_probability", "mean"),
        actual_rate=("scored_td", "mean"),
    ).reset_index()


def backtest(dataset: str, train_end: int, test_season: int, paths: Paths,
             model_type: str = "logistic", min_games: int = 2) -> None:
    d = pd.read_parquet(dataset)
    d = d.loc[d["games_prior"].ge(min_games)].copy()
    d["position_group_model"] = d["position"].map(get_group)
    d = d.loc[d["position_group_model"].notna()].copy()

    predictions = []
    fitted = {}
    for week in sorted(d.loc[d["season"].eq(test_season), "week"].unique()):
        print(f"Replaying {test_season} Week {week}...")
        # Expanding walk-forward: no games from this week or future are in training.
        train = d.loc[
            (d["season"] < test_season) |
            ((d["season"] == test_season) & (d["week"] < week))
        ].copy()
        train = train.loc[train["season"].le(train_end) | train["season"].eq(test_season)]
        test = d.loc[(d["season"] == test_season) & (d["week"] == week)].copy()

        for group in POSITION_GROUPS:
            tr = train.loc[train["position_group_model"].eq(group)].sort_values(["season", "week"]).copy()
            te = test.loc[test["position_group_model"].eq(group)].copy()
            if te.empty or len(tr) < 500:
                continue
            model, num, cat = fit_calibrated(tr, group, model_type)
            te["model_probability"] = predict_group(model, te, num, cat)
            te["model_group"] = group
            te["confidence"] = confidence_score(te)
            predictions.append(te)
            fitted[group] = (model, num, cat)

    if not predictions:
        raise RuntimeError("No predictions were generated. Check season coverage and minimum games.")
    pred = pd.concat(predictions, ignore_index=True)
    pred["market_implied_probability"] = np.nan
    pred["edge"] = np.nan

    pred_out = paths.outputs / f"backtest_{test_season}_predictions.csv"
    pred.to_csv(pred_out, index=False)
    pred.to_parquet(paths.outputs / f"backtest_{test_season}_predictions.parquet", index=False)

    overall = evaluate_predictions(pred, f"{test_season} overall")
    metrics = [overall]
    for group, g in pred.groupby("model_group"):
        metrics.append(evaluate_predictions(g, f"{test_season} {group}"))
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(paths.outputs / f"backtest_{test_season}_metrics.csv", index=False)
    calibration_table(pred).to_csv(paths.outputs / f"backtest_{test_season}_calibration.csv", index=False)

    # Save final expanding models for future scoring.
    for group, (model, num, cat) in fitted.items():
        joblib.dump({"model": model, "numeric": num, "categorical": cat},
                    paths.models / f"td_model_{group}_{test_season}.joblib")

    print("\n=== BACKTEST SUMMARY ===")
    print(metrics_df.to_string(index=False))
    print(f"\nPredictions: {pred_out}")


def audit(dataset: str, paths: Paths, season_max: int = 2024,
          model_type: str = "logistic") -> None:
    d = pd.read_parquet(dataset)
    d = d.loc[d["season"].le(season_max) & d["games_prior"].ge(2)].copy()
    d["position_group_model"] = d["position"].map(get_group)
    reports = []

    for group in POSITION_GROUPS:
        g = d.loc[d["position_group_model"].eq(group)].sort_values(["season", "week"]).copy()
        if len(g) < 500:
            continue
        num, cat = feature_list(group, g)

        corr = g[num].corr(method="spearman").abs()
        pairs = []
        for i, a in enumerate(num):
            for b in num[i + 1:]:
                if corr.loc[a, b] >= 0.80:
                    pairs.append({"group": group, "feature_a": a, "feature_b": b,
                                  "spearman_abs": corr.loc[a, b]})
        pd.DataFrame(pairs).sort_values("spearman_abs", ascending=False).to_csv(
            paths.outputs / f"redundancy_{group}.csv", index=False
        )

        split = int(len(g) * 0.8)
        train, valid = g.iloc[:split], g.iloc[split:]
        model, num, cat = fit_calibrated(train, group, model_type)
        base_p = predict_group(model, valid, num, cat)
        base_loss = log_loss(valid["scored_td"], base_p, labels=[0, 1])

        # Feature-group permutation, done on raw columns to preserve interpretability.
        rng = np.random.default_rng(RANDOM_STATE)
        for feature in num + cat:
            shuffled = valid.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].values)
            p = predict_group(model, shuffled, num, cat)
            reports.append({
                "group": group,
                "feature": feature,
                "base_log_loss": base_loss,
                "shuffled_log_loss": log_loss(valid["scored_td"], p, labels=[0, 1]),
                "importance_log_loss": log_loss(valid["scored_td"], p, labels=[0, 1]) - base_loss,
            })

    report = pd.DataFrame(reports).sort_values(["group", "importance_log_loss"], ascending=[True, False])
    report.to_csv(paths.outputs / "feature_audit.csv", index=False)
    print(report.groupby("group").head(20).to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leakage-safe NFL anytime TD model")
    p.add_argument("--project-dir", default=".", help="Project directory")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-data")
    b.add_argument("--start-season", type=int, default=2022)
    b.add_argument("--end-season", type=int, default=2025)
    b.add_argument("--scheme-csv", default=None)

    bt = sub.add_parser("backtest")
    bt.add_argument("--dataset", default="data/processed/player_game_features.parquet")
    bt.add_argument("--train-end", type=int, default=2024)
    bt.add_argument("--test-season", type=int, default=2025)
    bt.add_argument("--model-type", choices=["logistic", "histgb"], default="logistic")
    bt.add_argument("--min-games", type=int, default=2)

    a = sub.add_parser("audit")
    a.add_argument("--dataset", default="data/processed/player_game_features.parquet")
    a.add_argument("--season-max", type=int, default=2024)
    a.add_argument("--model-type", choices=["logistic", "histgb"], default="logistic")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths(Path(args.project_dir).resolve())
    paths.ensure()

    if args.command == "build-data":
        build_dataset(args.start_season, args.end_season, paths, args.scheme_csv)
    elif args.command == "backtest":
        dataset = Path(args.dataset)
        if not dataset.is_absolute():
            dataset = paths.root / dataset
        backtest(str(dataset), args.train_end, args.test_season, paths,
                 args.model_type, args.min_games)
    elif args.command == "audit":
        dataset = Path(args.dataset)
        if not dataset.is_absolute():
            dataset = paths.root / dataset
        audit(str(dataset), paths, args.season_max, args.model_type)


if __name__ == "__main__":
    main()
