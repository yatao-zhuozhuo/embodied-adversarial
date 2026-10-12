import json
from pathlib import Path
import tempfile
import unittest

from scripts.compress_unified_expert_sft import accepted_episodes


class UnifiedCompressionSelectionTest(unittest.TestCase):
    def prepare(self, root, outcome_success, count=1):
        rows = []
        for index in range(count):
            case = {"case_id": f"metaworld/reach-v3/sft_train/{index:03d}",
                    "bench": "metaworld", "task_id": "reach-v3", "seed": index}
            rows.append((case, {}, None))
            outcome = root / "results" / (case["case_id"] + ".json")
            outcome.parent.mkdir(parents=True, exist_ok=True)
            outcome.write_text(json.dumps({"success": outcome_success}))
            episode = root / "episodes/metaworld/reach-v3" / f"seed-{index}"
            episode.mkdir(parents=True)
            (episode / "episode.json").write_text(json.dumps({"success": True,
                "stop_reason": "official_success", "unified_case": case}))
            (episode / "samples.jsonl").write_text("{}\n")
        return rows

    def test_all_successes_selected_without_legacy_quota(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.prepare(root, True, 61)
            self.assertEqual(len(accepted_episodes(root, rows, "metaworld")), 61)

    def test_uncommitted_episode_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.prepare(root, False)
            self.assertEqual(accepted_episodes(root, rows, "metaworld"), [])

    def test_changed_initial_binding_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.prepare(root, True)
            rows[0][0]["seed"] = 999
            old = root / "episodes/metaworld/reach-v3/seed-0"
            old.rename(old.with_name("seed-999"))
            with self.assertRaises(ValueError):
                accepted_episodes(root, rows, "metaworld")


if __name__ == "__main__":
    unittest.main()
