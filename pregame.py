"""Pregame candidate boards, availability adjustments, and honest grading.

The legacy replay ranked only players who later recorded an offensive snap. This
module reconstructs each game at kickoff minus 24 hours, preserves later
inactives as voids, and never backfills after games have been played.
"""

from __future__ import annotations

import html as html_lib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import anytime_td_model_v1 as core


TEAM_ABBR = {
    "Cardinals": "ARI", "Falcons": "ATL", "Ravens": "BAL", "Bills": "BUF",
    "Panthers": "CAR", "Bears": "CHI", "Bengals": "CIN", "Browns": "CLE",
    "Cowboys": "DAL", "Broncos": "DEN", "Lions": "DET", "Packers": "GB",
    "Texans": "HOU", "Colts": "IND", "Jaguars": "JAC", "Chiefs": "KC",
    "Raiders": "LV", "Chargers": "LAC", "Rams": "LA", "Dolphins": "MIA",
    "Vikings": "MIN", "Patriots": "NE", "Saints": "NO", "Giants": "NYG",
    "Jets": "NYJ", "Eagles": "PHI", "Steelers": "PIT", "49ers": "SF",
    "Seahawks": "SEA", "Buccaneers": "TB", "Titans": "TEN",
    "Commanders": "WAS",
}

ROLE_FIELDS = {
    "snap_share_ewm": "all",
    "route_participation_ewm": "receiving",
    "rush_share_ewm": "rushing",
    "target_share_ewm": "receiving",
    "red_zone_opp_share_ewm": "all",
    "inside10_opp_share_ewm": "all",
    "inside5_opp_share_ewm": "all",
    "end_zone_target_share_ewm": "receiving",
    "goal_line_rush_share_ewm": "rushing",
    "receiving_opp_share_ewm": "receiving",
    "air_yards_share_ewm": "receiving",
    "red_zone_target_share_ewm": "receiving",
    "qb_inside5_rush_share_ewm": "rushing",
    "qb_designed_rush_share_ewm": "rushing",
}


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", text)
    return re.sub(r"[^a-z0-9]", "", text)


def _cell_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html_lib.unescape(value)).strip()


def parse_injury_report(page: str, season: int, week: int) -> pd.DataFrame:
    records: list[dict] = []
    sections = re.findall(
        r'd3-o-section-sub-title[^>]*>\s*<span>([^<]+)</span>.*?<tbody>(.*?)</tbody>',
        page,
        flags=re.I | re.S,
    )
    for team_name, body in sections:
        team = TEAM_ABBR.get(_cell_text(team_name))
        if not team:
            continue
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, flags=re.I | re.S):
            cells = [_cell_text(x) for x in re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.I | re.S)]
            if len(cells) < 5:
                continue
            records.append({
                "season": season, "week": week, "team": team,
                "player_name_injury": cells[0], "injury_position": cells[1],
                "injury": cells[2], "practice_status": cells[3],
                "game_status": cells[4], "name_key": normalize_name(cells[0]),
            })
    return pd.DataFrame(records)


def injury_report(season: int, week: int, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"injuries_{season}_week_{week:02d}.csv"
    if cached.exists():
        return pd.read_csv(cached)
    url = f"https://www.nfl.com/injuries/league/{season}/reg{week}"
    try:
        response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        report = parse_injury_report(response.text, season, week)
        if not report.empty:
            report.to_csv(cached, index=False)
        return report
    except Exception as exc:
        print(f"Official injury report unavailable for {season} Week {week}: {exc}")
        return pd.DataFrame()


def availability_probability(game_status: pd.Series, practice_status: pd.Series) -> pd.Series:
    game = game_status.fillna("").astype(str).str.lower()
    practice = practice_status.fillna("").astype(str).str.lower()
    result = pd.Series(1.0, index=game.index)
    result = result.mask(game.str.contains("out|reserve|suspend", regex=True), 0.0)
    result = result.mask(game.str.contains("doubtful"), 0.15)
    result = result.mask(game.str.contains("questionable"), 0.70)
    result = result.mask(game.eq("") & practice.str.contains("did not participate"), 0.65)
    result = result.mask(game.eq("") & practice.str.contains("limited"), 0.90)
    return result


def game_cutoffs(games: pd.DataFrame, hours_before: int = 24) -> pd.DataFrame:
    """Return one immutable, game-relative information cutoff per team."""
    gametime = games.get("gametime", pd.Series("00:00", index=games.index)).fillna("00:00").astype(str)
    local = pd.to_datetime(
        pd.to_datetime(games["gameday"], errors="coerce").dt.strftime("%Y-%m-%d")
        + " " + gametime,
        errors="coerce",
    ).dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
    kickoff = local.dt.tz_convert("UTC")
    game_id = games.get("game_id", pd.Series("", index=games.index))
    home = pd.DataFrame({
        "team": games["home_team"], "snapshot_game_id": game_id,
        "scheduled_kickoff": kickoff, "board_cutoff_utc": kickoff - pd.Timedelta(hours=hours_before),
    })
    away = pd.DataFrame({
        "team": games["away_team"], "snapshot_game_id": game_id,
        "scheduled_kickoff": kickoff, "board_cutoff_utc": kickoff - pd.Timedelta(hours=hours_before),
    })
    return pd.concat([home, away], ignore_index=True)


def weekly_candidates(
    season: int,
    week: int,
    games: pd.DataFrame,
    depth: pd.DataFrame,
    cache_dir: Path,
    freeze_friday: bool = True,
) -> pd.DataFrame:
    import nflreadpy as nfl

    roster = core._to_pandas(nfl.load_rosters_weekly(season))
    roster = roster.loc[pd.to_numeric(roster["week"], errors="coerce").eq(week)].copy()
    roster["player_id"] = roster.get("gsis_id", roster.get("player_id"))
    roster["player_name"] = roster.get("full_name", roster.get("player_name"))
    roster["position"] = roster.get("position", "UNK").astype(str).str.upper().replace({"HB": "RB"})
    roster["team"] = core.normalize_team(roster.get("team", roster.get("club_code")))
    roster["roster_status"] = roster.get("status", "Unknown").astype(str)
    roster["headshot_url"] = roster.get("headshot", roster.get("headshot_url", np.nan))
    valid = set().union(*core.POSITION_GROUPS.values())
    roster = roster.loc[roster["position"].isin(valid) & roster["player_id"].notna()].copy()
    # INA can only be known on game day, so retain it to create an honest void.
    # CUT/RET/RES/PRA rows can remain in the weekly archive and are not candidates.
    roster = roster.loc[roster["roster_status"].str.upper().isin(["ACT", "INA"])].copy()

    cutoffs = game_cutoffs(games, 24)
    if not freeze_friday:
        cutoffs["board_cutoff_utc"] = pd.Timestamp.now(tz="UTC")
    roster = roster.merge(cutoffs, on="team", how="inner")

    roster["depth_rank"] = np.nan
    if not depth.empty:
        dep = depth.copy()
        dep["player_id"] = dep.get("gsis_id", dep.get("player_id"))
        dep["depth_rank"] = pd.to_numeric(dep.get("pos_rank"), errors="coerce")
        dep["dt"] = pd.to_datetime(dep.get("dt"), errors="coerce", utc=True)
        dep = dep.loc[dep["player_id"].isin(roster["player_id"])].copy()
        matched = roster[["player_id", "board_cutoff_utc"]].merge(
            dep[["player_id", "depth_rank", "dt"]], on="player_id", how="left"
        )
        matched = matched.loc[matched["dt"].le(matched["board_cutoff_utc"])].sort_values("dt")
        matched = matched.drop_duplicates(["player_id", "board_cutoff_utc"], keep="last")
        roster = roster.drop(columns="depth_rank").merge(
            matched[["player_id", "board_cutoff_utc", "depth_rank"]],
            on=["player_id", "board_cutoff_utc"], how="left",
        )
    roster = roster.loc[roster["depth_rank"].le(3) | roster["depth_rank"].isna()].copy()

    report = injury_report(season, week, cache_dir)
    roster["name_key"] = roster["player_name"].map(normalize_name)
    if not report.empty:
        roster = roster.merge(
            report[["team", "name_key", "injury", "practice_status", "game_status"]],
            on=["team", "name_key"], how="left",
        )
    else:
        roster["injury"] = ""
        roster["practice_status"] = ""
        roster["game_status"] = ""
    roster["availability_probability"] = availability_probability(
        roster["game_status"], roster["practice_status"]
    )
    # Players known to be OUT at the reconstructed cutoff are not candidates.
    roster = roster.loc[roster["availability_probability"].gt(0)].copy()
    roster["board_cutoff"] = roster["board_cutoff_utc"].map(
        lambda value: value.isoformat() if pd.notna(value) else None
    )
    return roster.drop_duplicates("player_id", keep="last")


def _depth_prior(position: pd.Series, rank: pd.Series, field: str) -> pd.Series:
    rank = pd.to_numeric(rank, errors="coerce").fillna(3).clip(1, 4).astype(int)
    values = []
    for pos, r in zip(position.astype(str), rank):
        if field == "rush_share_ewm":
            table = {"RB": [0.55, 0.28, 0.12], "FB": [0.08, 0.04, 0.02], "QB": [0.18, 0.14, 0.10]}
        elif field in {"target_share_ewm", "receiving_opp_share_ewm", "air_yards_share_ewm", "red_zone_target_share_ewm", "end_zone_target_share_ewm"}:
            table = {"WR": [0.23, 0.18, 0.12], "TE": [0.17, 0.10, 0.05], "RB": [0.12, 0.08, 0.04], "FB": [0.03, 0.02, 0.01]}
        elif field == "route_participation_ewm":
            table = {"WR": [0.88, 0.76, 0.60], "TE": [0.80, 0.55, 0.32], "RB": [0.48, 0.33, 0.20], "FB": [0.20, 0.12, 0.08]}
        else:
            table = {"QB": [0.98, 0.10, 0.05], "RB": [0.58, 0.34, 0.18], "WR": [0.86, 0.73, 0.58], "TE": [0.78, 0.52, 0.30], "FB": [0.28, 0.18, 0.10]}
        arr = table.get(pos, [0.10, 0.05, 0.02])
        values.append(arr[min(r, 3) - 1])
    return pd.Series(values, index=position.index, dtype=float)


def project_roles(frame: pd.DataFrame, use_availability: bool, use_depth: bool) -> pd.DataFrame:
    """Turn trailing role into a pregame workload projection and redistribute absences."""
    d = frame.copy()
    available = pd.to_numeric(d.get("availability_probability", 1.0), errors="coerce").fillna(1.0)
    for field, pool in ROLE_FIELDS.items():
        if field not in d.columns:
            continue
        base = pd.to_numeric(d[field], errors="coerce")
        d[f"{field}_base"] = base
        projected = base.copy()
        if use_depth:
            prior = _depth_prior(d["position"], d.get("depth_rank", pd.Series(3, index=d.index)), field)
            history_weight = (pd.to_numeric(d.get("games_prior", 0), errors="coerce").fillna(0) / 4).clip(0, 1)
            projected = history_weight * base.fillna(prior) + (1 - history_weight) * prior
        if use_availability:
            projected = projected.fillna(0) * available
            if pool == "rushing":
                eligible = d["position"].isin(["QB", "RB", "FB"])
            elif pool == "receiving":
                eligible = d["position"].isin(["RB", "FB", "WR", "TE"])
            else:
                eligible = pd.Series(True, index=d.index)
            before = projected.where(eligible, 0).groupby([d["season"], d["week"], d["team"]]).transform("sum")
            original = base.fillna(0).where(eligible, 0).groupby([d["season"], d["week"], d["team"]]).transform("sum")
            scale = core.safe_div(original, before).replace([np.inf, -np.inf], np.nan).fillna(1).clip(1, 2)
            projected = projected.where(~eligible, projected * scale)
        d[field] = projected.clip(0, 1)
    d["role_projection_version"] = (
        "depth+availability" if use_depth and use_availability else
        "depth" if use_depth else "availability" if use_availability else "baseline"
    )
    return d


def load_expert_rankings() -> pd.DataFrame:
    """Load timestamped pregame consensus position ranks for audit purposes."""
    import nflreadpy as nfl

    raw = nfl.load_ff_rankings("all")
    wanted = ["weekly-qb", "weekly-rb", "weekly-wr", "weekly-te"]
    if hasattr(raw, "filter"):
        import polars as pl

        raw = raw.filter(pl.col("page_type").is_in(wanted)).select(
            ["page_type", "player", "team", "tm", "ecr", "scrape_date"]
        )
    ranks = core._to_pandas(raw)
    ranks = ranks.loc[ranks["page_type"].isin(wanted)].copy()
    ranks["position"] = ranks["page_type"].str.replace("weekly-", "", regex=False).str.upper()
    ranks["team"] = core.normalize_team(ranks["team"].fillna(ranks["tm"]))
    ranks["name_key"] = ranks["player"].map(normalize_name)
    ranks["scrape_date"] = pd.to_datetime(ranks["scrape_date"], errors="coerce").dt.normalize()
    ranks["ecr"] = pd.to_numeric(ranks["ecr"], errors="coerce")
    ranks = ranks.dropna(subset=["scrape_date", "ecr", "name_key"])
    group = ranks.groupby(["scrape_date", "position"])["ecr"]
    size = group.transform("count").sub(1).clip(lower=1)
    ranks["expert_rank_percentile"] = 1 - group.rank(method="average").sub(1).div(size)
    return ranks.sort_values("ecr").drop_duplicates(
        ["scrape_date", "position", "name_key"], keep="first"
    )


def attach_expert_rankings(frame: pd.DataFrame, ranks: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    d["name_key"] = d["player_name"].map(normalize_name)
    if "board_cutoff" in d:
        cutoff = pd.to_datetime(
            d["board_cutoff"], errors="coerce", utc=True, format="mixed"
        ).dt.tz_localize(None).dt.normalize()
    else:
        cutoff = pd.Series(pd.Timestamp.now().normalize(), index=d.index)
    d["ranking_cutoff_date"] = cutoff
    d["_ranking_row"] = np.arange(len(d))
    use = ranks[["scrape_date", "position", "name_key", "expert_rank_percentile", "ecr"]]
    pairs = d[["_ranking_row", "position", "name_key", "ranking_cutoff_date"]].merge(
        use, on=["position", "name_key"], how="left"
    )
    age = pairs["ranking_cutoff_date"] - pairs["scrape_date"]
    pairs = pairs.loc[age.between(pd.Timedelta(0), pd.Timedelta(days=3))].sort_values("scrape_date")
    latest = pairs.drop_duplicates("_ranking_row", keep="last")
    d = d.merge(
        latest[["_ranking_row", "scrape_date", "expert_rank_percentile", "ecr"]],
        on="_ranking_row", how="left",
    )
    return d.drop(columns="_ranking_row")


def fixed_board_metrics(predictions: pd.DataFrame, board_sizes=(5, 10, 20, 30)) -> dict:
    played = predictions.loc[predictions["played"].eq(1)].copy()
    y = played["scored_td"].astype(int)
    p = played["model_probability"].clip(1e-5, 1 - 1e-5)
    result = {
        "candidate_predictions": int(len(predictions)),
        "graded_predictions": int(len(played)),
        "void_predictions": int((predictions["played"].eq(0)).sum()),
        "candidate_auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else None,
        "candidate_brier": float(brier_score_loss(y, p)),
        "candidate_log_loss": float(log_loss(y, p, labels=[0, 1])),
    }
    rookie_indicator = pd.to_numeric(
        played.get("rookie_flag", pd.Series(np.nan, index=played.index)), errors="coerce"
    )
    if "rookie_season" in played:
        rookie_indicator = rookie_indicator.fillna(
            pd.to_numeric(played["rookie_season"], errors="coerce").eq(
                pd.to_numeric(played["season"], errors="coerce")
            ).astype(int)
        )
    rookies = played.loc[rookie_indicator.eq(1)].copy()
    if not rookies.empty:
        rookie_y = rookies["scored_td"].astype(int)
        rookie_p = rookies["model_probability"].clip(1e-5, 1 - 1e-5)
        result.update({
            "rookie_graded": int(len(rookies)),
            "rookie_td_rate": float(rookie_y.mean()),
            "rookie_auc": float(roc_auc_score(rookie_y, rookie_p)) if rookie_y.nunique() == 2 else None,
            "rookie_brier": float(brier_score_loss(rookie_y, rookie_p)),
            "rookie_log_loss": float(log_loss(rookie_y, rookie_p, labels=[0, 1])),
        })
    for n in board_sizes:
        selected = predictions.loc[predictions["board_rank"].le(n)]
        graded = selected.loc[selected["played"].eq(1)]
        result[f"top{n}_selected"] = int(len(selected))
        result[f"top{n}_graded"] = int(len(graded))
        result[f"top{n}_voids"] = int((selected["played"].eq(0)).sum())
        result[f"top{n}_void_rate"] = float(1 - len(graded) / len(selected)) if len(selected) else None
        result[f"top{n}_hits"] = int(graded["scored_td"].sum())
        result[f"top{n}_precision"] = float(graded["scored_td"].mean()) if len(graded) else None
    return result


def _score_variants(
    frame: pd.DataFrame,
    season: int,
    week: int,
    model_type: str,
    expert_ranks: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    train = frame.loc[
        ~frame["is_forward"].fillna(False)
        & frame["scored_td"].notna()
        & (frame["games_prior"].ge(2) | frame.get("rookie_flag", pd.Series(0, index=frame.index)).eq(1))
        & ((frame["season"] < season) | ((frame["season"] == season) & (frame["week"] < week)))
    ].sort_values(["season", "week"]).copy()
    score = frame.loc[frame["is_forward"].fillna(False)].copy()
    # Market/closing lines are intentionally omitted until the separate market
    # phase can preserve a timestamped price snapshot for every historical board.
    for column in ["spread", "team_implied_total", "game_total"]:
        if column in train:
            train[column] = np.nan
        if column in score:
            score[column] = np.nan

    variants = {
        "baseline": project_roles(score, use_availability=False, use_depth=False),
        "availability": project_roles(score, use_availability=True, use_depth=False),
        "projected_role": project_roles(score, use_availability=True, use_depth=True),
    }
    output: dict[str, list[pd.DataFrame]] = {name: [] for name in variants}
    output["legacy_features"] = []
    output["rules_environment_only"] = []
    output["rookie_pathway_only"] = []
    output["history_2024_plus"] = []
    output["history_recency_weighted"] = []
    for group in core.POSITION_GROUPS:
        tr = train.loc[train["position_group_model"].eq(group)].copy()
        if len(tr) < 500:
            continue
        model, numeric, categorical = core.fit_calibrated(tr, group, model_type)
        for name, candidate_frame in variants.items():
            candidates = candidate_frame.loc[candidate_frame["position_group_model"].eq(group)].copy()
            if candidates.empty:
                continue
            candidates["model_probability"] = core.predict_group(
                model, candidates, numeric, categorical
            )
            candidates["model_group"] = group
            candidates["confidence"] = core.confidence_score(candidates)
            output[name].append(candidates)
        base_candidates = variants["baseline"].loc[
            variants["baseline"]["position_group_model"].eq(group)
        ].copy()
        recent = tr.loc[tr["season"].ge(2024)].copy()
        if len(recent) >= 500 and not base_candidates.empty:
            recent_model, recent_num, recent_cat = core.fit_calibrated(
                recent, group, model_type
            )
            recent_score = base_candidates.copy()
            recent_score["model_probability"] = core.predict_group(
                recent_model, recent_score, recent_num, recent_cat
            )
            recent_score["model_group"] = group
            recent_score["confidence"] = core.confidence_score(recent_score)
            output["history_2024_plus"].append(recent_score)
        weighted = tr.copy()
        weighted["_sample_weight"] = 0.65 ** (
            season - pd.to_numeric(weighted["season"], errors="coerce")
        )
        if not base_candidates.empty:
            weighted_model, weighted_num, weighted_cat = core.fit_calibrated(
                weighted, group, model_type
            )
            weighted_score = base_candidates.copy()
            weighted_score["model_probability"] = core.predict_group(
                weighted_model, weighted_score, weighted_num, weighted_cat
            )
            weighted_score["model_group"] = group
            weighted_score["confidence"] = core.confidence_score(weighted_score)
            output["history_recency_weighted"].append(weighted_score)

        rule_columns = [
            "league_offensive_tds_prior", "league_drives_prior",
            "league_red_zone_trips_prior", "league_plays_prior",
            "league_start_field_position_prior", "dynamic_kickoff_era",
            "touchback_35_era", "both_teams_overtime_era",
        ]
        rookie_columns = ["rookie_flag", "draft_round", "draft_pick"]

        def score_ablation(
            name: str,
            ablation_train: pd.DataFrame,
            missing_columns: list[str],
        ) -> None:
            ablation_score = base_candidates.copy()
            for column in missing_columns:
                if column in ablation_train:
                    ablation_train[column] = np.nan
                if column in ablation_score:
                    ablation_score[column] = np.nan
            if len(ablation_train) < 500 or ablation_score.empty:
                return
            ablation_model, ablation_num, ablation_cat = core.fit_calibrated(
                ablation_train, group, model_type
            )
            ablation_score["model_probability"] = core.predict_group(
                ablation_model, ablation_score, ablation_num, ablation_cat
            )
            ablation_score["model_group"] = group
            ablation_score["confidence"] = core.confidence_score(ablation_score)
            output[name].append(ablation_score)

        old_sample = tr.loc[tr["games_prior"].ge(2)].copy()
        score_ablation(
            "legacy_features", old_sample.copy(), rule_columns + rookie_columns
        )
        score_ablation(
            "rules_environment_only", old_sample.copy(), rookie_columns
        )
        score_ablation(
            "rookie_pathway_only", tr.copy(), rule_columns
        )
    combined = {
        name: pd.concat(parts, ignore_index=True) for name, parts in output.items() if parts
    }
    if "legacy_features" in combined:
        rookie_train = train.loc[train.get("rookie_flag", pd.Series(0, index=train.index)).eq(1)].copy()
        rookie_source = score.loc[score.get("rookie_flag", pd.Series(0, index=score.index)).eq(1)].copy()
        rule_columns = [
            "league_offensive_tds_prior", "league_drives_prior",
            "league_red_zone_trips_prior", "league_plays_prior",
            "league_start_field_position_prior", "dynamic_kickoff_era",
            "touchback_35_era", "both_teams_overtime_era",
        ]
        for column in rule_columns:
            if column in rookie_train:
                rookie_train[column] = np.nan
            if column in rookie_source:
                rookie_source[column] = np.nan
        if len(rookie_train) >= 500 and not rookie_source.empty:
            rookie_model, rookie_num, rookie_cat = core.fit_calibrated(
                rookie_train.sort_values(["season", "week"]), "ROOKIE", model_type
            )
            rookie_source["rookie_specialist_probability"] = core.predict_group(
                rookie_model, rookie_source, rookie_num, rookie_cat
            )
            specialist = combined["legacy_features"].copy()
            specialist = specialist.merge(
                rookie_source[["player_id", "rookie_specialist_probability"]],
                on="player_id", how="left",
            )
            use_rookie = specialist["rookie_specialist_probability"].notna()
            specialist.loc[use_rookie, "model_probability"] = specialist.loc[
                use_rookie, "rookie_specialist_probability"
            ]
            specialist["model_group"] = np.where(
                use_rookie, "ROOKIE", specialist["model_group"]
            )
            combined["legacy_plus_rookie_specialist"] = specialist
    def add_reranks(source_name: str, output_name: str) -> None:
        if source_name not in combined:
            return
        source = combined[source_name]
        ranked = attach_expert_rankings(source, expert_ranks) if expert_ranks is not None and not expert_ranks.empty else source.copy()
        expert_factor = 1 + 0.30 * (
            ranked.get("expert_rank_percentile", pd.Series(0.5, index=ranked.index)).fillna(0.5) - 0.5
        )
        ranked["ranking_score"] = (
            ranked["model_probability"]
            * expert_factor
            * pd.to_numeric(ranked.get("availability_probability", 1), errors="coerce").fillna(1).pow(0.20)
        )
        ranked["role_projection_version"] = "expert+injury-rerank"
        combined[output_name] = ranked

    if "baseline" in combined:
        baseline = combined["baseline"]
        injury = baseline.copy()
        injury["ranking_score"] = injury["model_probability"] * pd.to_numeric(
            injury.get("availability_probability", 1), errors="coerce"
        ).fillna(1).pow(0.20)
        injury["role_projection_version"] = "injury-rerank"
        combined["injury_rerank"] = injury
        if expert_ranks is not None and not expert_ranks.empty:
            expert = attach_expert_rankings(baseline, expert_ranks)
            expert_factor = 1 + 0.30 * (expert["expert_rank_percentile"].fillna(0.5) - 0.5)
            expert["ranking_score"] = expert["model_probability"] * expert_factor
            expert["role_projection_version"] = "expert-rerank"
            combined["expert_role"] = expert
            recommended = expert.copy()
            recommended["ranking_score"] *= pd.to_numeric(
                recommended.get("availability_probability", 1), errors="coerce"
            ).fillna(1).pow(0.20)
            recommended["role_projection_version"] = "expert+injury-rerank"
            combined["recommended"] = recommended
    add_reranks("history_2024_plus", "history_2024_plus_recommended")
    add_reranks("history_recency_weighted", "history_recency_weighted_recommended")
    return combined


def _attach_outcomes(
    scored: pd.DataFrame,
    week_outcomes: pd.DataFrame,
    week_snaps: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    outcome = week_outcomes[["player_id", "scored_td", "td_count"]].drop_duplicates("player_id")
    played = week_snaps[["player_id", "offense_snaps"]].copy()
    played["offense_snaps"] = pd.to_numeric(played["offense_snaps"], errors="coerce").fillna(0)
    played = played.groupby("player_id", as_index=False)["offense_snaps"].sum()
    result = scored.merge(outcome, on="player_id", how="left", suffixes=("", "_actual"))
    if "scored_td_actual" in result:
        result["scored_td"] = result["scored_td_actual"].fillna(0).astype(int)
        result.drop(columns="scored_td_actual", inplace=True)
    else:
        result["scored_td"] = result["scored_td"].fillna(0).astype(int)
    if "td_count_actual" in result:
        result["td_count"] = result["td_count_actual"].fillna(0)
        result.drop(columns="td_count_actual", inplace=True)
    result = result.merge(played, on="player_id", how="left", suffixes=("", "_actual"))
    snap_col = "offense_snaps_actual" if "offense_snaps_actual" in result else "offense_snaps"
    result["played"] = pd.to_numeric(result.get(snap_col, 0), errors="coerce").fillna(0).gt(0).astype(int)
    result["void"] = 1 - result["played"]
    result["variant"] = variant
    rank_column = "ranking_score" if "ranking_score" in result else "model_probability"
    result = result.sort_values([rank_column, "confidence"], ascending=False).reset_index(drop=True)
    result["board_rank"] = np.arange(1, len(result) + 1)
    return result


def run_fixed_board_backtest(
    root: Path,
    test_season: int = 2025,
    train_start: int = 2022,
    model_type: str = "logistic",
) -> dict:
    """Replay best-available kickoff-minus-24h boards and test each layer."""
    import nflreadpy as nfl
    import td_model

    paths = core.Paths(root)
    paths.ensure()
    all_schedules = td_model.load_schedules()
    pbp, stats, history_schedules, snaps = td_model.load_history(
        list(range(train_start, test_season + 1))
    )
    depth = core._to_pandas(nfl.load_depth_charts(test_season))
    player_metadata = core._to_pandas(nfl.load_players())
    full_outcomes = core.build_player_week_from_pbp(pbp)
    full_team_week = core.build_team_week(pbp)
    full_player_stats = core.prepare_player_stats(stats)
    test_weeks = sorted(
        all_schedules.loc[
            all_schedules["season"].eq(test_season)
            & pd.to_numeric(td_model.first_column(all_schedules, ["home_score"]), errors="coerce").notna(),
            "week",
        ].dropna().astype(int).unique()
    )
    if not test_weeks:
        test_weeks = sorted(full_outcomes.loc[full_outcomes["season"].eq(test_season), "week"].unique())

    predictions: dict[str, list[pd.DataFrame]] = {}
    try:
        expert_ranks = load_expert_rankings()
    except Exception as exc:
        print(f"Archived expert rankings unavailable; skipping that ablation ({exc}).")
        expert_ranks = None
    history_dir = root / "boards" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    for week in test_weeks:
        print(f"Freezing {test_season} Week {week} pregame board...")
        games = td_model.target_games(all_schedules, test_season, int(week))
        roster = weekly_candidates(test_season, int(week), games, depth, paths.raw)
        prior_mask_pbp = (pd.to_numeric(pbp["season"], errors="coerce") < test_season) | (
            pd.to_numeric(pbp["season"], errors="coerce").eq(test_season)
            & pd.to_numeric(pbp["week"], errors="coerce").lt(week)
        )
        prior_mask_stats = (pd.to_numeric(stats["season"], errors="coerce") < test_season) | (
            pd.to_numeric(stats["season"], errors="coerce").eq(test_season)
            & pd.to_numeric(stats["week"], errors="coerce").lt(week)
        )
        if snaps.empty:
            prior_snaps = snaps
            week_snaps = snaps
        else:
            snap_season = pd.to_numeric(snaps["season"], errors="coerce")
            snap_week = pd.to_numeric(snaps["week"], errors="coerce")
            prior_snaps = snaps.loc[(snap_season < test_season) | (snap_season.eq(test_season) & snap_week.lt(week))]
            week_snaps = snaps.loc[snap_season.eq(test_season) & snap_week.eq(week)]
        schedule_season = pd.to_numeric(history_schedules["season"], errors="coerce")
        schedule_week = pd.to_numeric(history_schedules["week"], errors="coerce")
        prior_schedules = history_schedules.loc[
            (schedule_season < test_season) | (schedule_season.eq(test_season) & schedule_week.lt(week))
        ]
        frame = td_model.build_feature_frame(
            pbp.loc[prior_mask_pbp], stats.loc[prior_mask_stats], prior_snaps,
            prior_schedules, all_schedules, roster, test_season, int(week), paths,
            save_dataset=False,
            player_pbp_precomputed=full_outcomes.loc[
                (full_outcomes["season"] < test_season)
                | (full_outcomes["season"].eq(test_season) & full_outcomes["week"].lt(week))
            ],
            team_week_precomputed=full_team_week.loc[
                (full_team_week["season"] < test_season)
                | (full_team_week["season"].eq(test_season) & full_team_week["week"].lt(week))
            ],
            player_stats_prepared=full_player_stats.loc[
                (full_player_stats["season"] < test_season)
                | (full_player_stats["season"].eq(test_season) & full_player_stats["week"].lt(week))
            ],
            player_metadata=player_metadata,
        )
        variant_scores = _score_variants(
            frame, test_season, int(week), model_type, expert_ranks=expert_ranks
        )
        week_outcomes = full_outcomes.loc[
            full_outcomes["season"].eq(test_season) & full_outcomes["week"].eq(week)
        ]
        slate_teams = set(pd.concat([games["home_team"], games["away_team"]]))
        week_outcomes = week_outcomes.loc[week_outcomes["posteam"].isin(slate_teams)]
        week_snaps = week_snaps.loc[td_model.first_column(week_snaps, ["team"]).isin(slate_teams)]
        for variant, scored in variant_scores.items():
            graded = _attach_outcomes(scored, week_outcomes, week_snaps, variant)
            predictions.setdefault(variant, []).append(graded)
            graded.to_csv(
                history_dir / f"{test_season}_week_{week:02d}_{variant}.csv", index=False
            )

    all_predictions = []
    metrics = []
    for variant, parts in predictions.items():
        if not parts:
            continue
        pred = pd.concat(parts, ignore_index=True)
        all_predictions.append(pred)
        metric = fixed_board_metrics(pred)
        metric["variant"] = variant
        metrics.append(metric)
    combined = pd.concat(all_predictions, ignore_index=True)
    metrics_frame = pd.DataFrame(metrics)
    combined.to_csv(paths.outputs / f"pregame_backtest_{test_season}_predictions.csv", index=False)
    combined.to_parquet(paths.outputs / f"pregame_backtest_{test_season}_predictions.parquet", index=False)
    metrics_frame.to_csv(paths.outputs / f"pregame_backtest_{test_season}_metrics.csv", index=False)
    manifest = {
        "test_season": test_season,
        "board_timing": "Each game frozen 24 hours before its scheduled kickoff",
        "historical_snapshot_limit": "Best available reconstruction; some archived roster/injury fields are final weekly state",
        "void_policy": "Selected inactive/DNP players remain voids; no backfill",
        "market_policy": "Historical closing spread and total omitted",
        "variants": metrics,
    }
    write_manifest(paths.outputs / f"pregame_backtest_{test_season}_summary.json", manifest)
    print("\n=== FIXED PREGAME BOARD BACKTEST ===")
    print(metrics_frame.to_string(index=False))
    return manifest


def write_manifest(path: Path, payload: dict) -> None:
    payload = {**payload, "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
