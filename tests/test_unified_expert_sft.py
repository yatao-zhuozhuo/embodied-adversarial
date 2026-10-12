import copy
import json
from pathlib import Path
import unittest

from scripts.collect_unified_expert_sft import validate_case


class UnifiedExpertCasesTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "data_unified"
        manifest = json.loads((root / "manifests/sft_train.json").read_text())
        self.case = next(case for case in manifest["cases"] if case["bench"] == "libero")
        self.task = next(task for task in manifest["tasks"] if task["task_key"] == self.case["task_key"])
        self.reference = next(ref for ref in map(json.loads, (root / "source_refs.jsonl").read_text().splitlines())
                              if ref["source_ref_id"] == self.case["initial_state"]["source_ref_id"])

    def test_frozen_reference_accepted(self):
        validate_case(self.case, self.task, self.reference)

    def test_evaluation_case_rejected(self):
        case = {**self.case, "phase": "seen_eval"}
        with self.assertRaises(ValueError):
            validate_case(case, self.task, self.reference)

    def test_wrong_demo_rejected(self):
        reference = {**self.reference, "reference_demo_index": -1}
        with self.assertRaises(ValueError):
            validate_case(self.case, self.task, reference)

    def test_cross_phase_reference_rejected(self):
        reference = {**self.reference, "phase": "rl_train"}
        with self.assertRaises(ValueError):
            validate_case(self.case, self.task, reference)

    def test_wrong_fingerprint_rejected(self):
        case = copy.deepcopy(self.case)
        case["initial_state"]["state_fingerprint"] = "changed"
        with self.assertRaises(ValueError):
            validate_case(case, self.task, self.reference)

    def test_all_selected_maniskill_tasks_have_matching_camera_profiles(self):
        from agent.runtime.two_tool_input import make_profile
        root = Path(__file__).resolve().parents[1] / "data_unified"
        manifest = json.loads((root / "manifests/sft_train.json").read_text())
        for task in manifest["tasks"]:
            if task["bench"] != "maniskill":
                continue
            with self.subTest(task=task["task_id"]):
                dimension = task["expected_action_dim"]
                native = {"slots": ["slot"] * dimension,
                          "mode": task["environment_config"]["control_mode"]}
                profile = make_profile("maniskill", task["task_id"],
                                       {"input_control_contract": native}, {"action_dim": dimension})
                self.assertEqual(profile["primary"], ["base_camera"])
                self.assertEqual(bool(profile["wrist"]), task["robot"] == "panda_wristcam")
                if profile["wrist"]:
                    self.assertEqual(profile["camera_dimensions"]["hand_camera"], [128, 128])


if __name__ == "__main__":
    unittest.main()
