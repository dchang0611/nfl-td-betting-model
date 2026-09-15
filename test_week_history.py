import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import build_site


class SavedWeekTests(unittest.TestCase):
    def test_outcomes_require_matching_game_and_frozen_pregame_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            boards = root / "boards"
            boards.mkdir()
            processed = root / "data" / "processed"
            processed.mkdir(parents=True)
            base = dict(target_season=2026, target_week=1, nfl_game_id="game1",
                        kickoff="2026-09-10T20:00:00Z", actual_snapshot_utc="2026-09-09T20:00:00Z",
                        snapshot_status="frozen", model_probability=0.4)
            rows = [dict(base, player_id=str(i), ranking=i) for i in range(1, 8)]
            rows[3]["snapshot_status"] = "provisional"
            rows[4]["actual_snapshot_utc"] = "2026-09-11T20:00:00Z"
            path = boards / "anytime_td_board_2026_week_01.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            original = path.read_bytes()
            actual = [dict(season=2026, week=1, player_id=i, game_id="game1", scored_td=1 if i == 1 else 0,
                           td_count=1 if i == 1 else 0, offense_snaps=0 if i == 3 else 40) for i in range(1, 8)]
            actual[5]["game_id"] = "wronggame"
            actual.append(actual[-1].copy())
            pd.DataFrame(actual).to_parquet(processed / "player_game_features.parquet")
            with patch.object(build_site, "ROOT", root), patch.object(build_site, "BOARDS", boards):
                result = build_site.saved_week_payloads()[0]["rows"]
            self.assertEqual([r["result_status"] for r in result], ["hit", "miss", "void", "pending", "pending", "pending", "pending"])
            self.assertEqual([r["ranking"] for r in result], list(range(1, 8)))
            self.assertTrue(all(r["model_probability"] == 0.4 for r in result))
            self.assertEqual(path.read_bytes(), original)

    def test_missing_outcomes_stay_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame([dict(target_season=2026, target_week=2, ranking=1, player_id="p")]).to_csv(
                root / "anytime_td_board_2026_week_02.csv", index=False)
            with patch.object(build_site, "ROOT", root), patch.object(build_site, "BOARDS", root):
                row = build_site.saved_week_payloads()[0]["rows"][0]
            self.assertEqual(row["result_status"], "pending")
            self.assertIsNone(row["scored_td"])
            self.assertIsNone(row["void"])


if __name__ == "__main__":
    unittest.main()
