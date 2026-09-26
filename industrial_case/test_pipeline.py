"""Run with: python -m unittest industrial_case.test_pipeline -v."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .analysis import checked_records, hypervolume
from .baselines import batch_order, rule
from .data import PUBLISHED, fit_cost, validate_data
from .simulation import audit, replay, run_policy


class IndustrialCaseTests(unittest.TestCase):
    def test_published_inputs_and_cost_identifiability(self):
        validate_data()
        removal, addition, errors = fit_cost(PUBLISHED["switch_cost"])
        np.testing.assert_allclose(removal, [1.125, 2.125, 2.5, 2.25])
        np.testing.assert_allclose(addition, [1.375, .375, 3.5, 2.75])
        self.assertAlmostEqual(removal.sum(), addition.sum())
        self.assertAlmostEqual(errors["mae_thousand_cny"], 17/24)

    def test_full_source_case_baseline_and_replay(self):
        report = run_policy("full", rule("SPT"))
        self.assertTrue(report["feasible"], report["errors"])
        self.assertEqual(report["operations"], 80)
        low, nominal, high = [replay(report["events"], "full", f) for f in [.5, 1, 2]]
        self.assertLessEqual(low["makespan"], nominal["makespan"])
        self.assertLessEqual(nominal["makespan"], high["makespan"])
        self.assertLessEqual(nominal["makespan"], report["makespan"] + 1e-6)
        self.assertEqual(low["cost"], high["cost"])

    def test_auditor_rejects_tampering(self):
        report = run_policy("smoke", rule("SPT"))
        events = copy.deepcopy(report["events"])
        processing = next(e for e in events if e["kind"] == "process")
        processing["end"] += .5
        self.assertFalse(audit(events, "smoke")["feasible"])
        events = copy.deepcopy(report["events"])
        events.remove(next(e for e in events if e["kind"] == "install"))
        self.assertFalse(audit(events, "smoke")["feasible"])
        self.assertFalse(audit(report["events"][:-1], "smoke")["complete"])

    def test_two_stage_keeps_prespecified_family_order(self):
        report = run_policy("full", rule("TwoStage"))
        self.assertTrue(report["feasible"], report["errors"])
        for station in range(5):
            products = [e["product"] for e in report["events"] if e["machine"] == station and e["kind"] == "process"]
            unique = [p for i, p in enumerate(products) if i == 0 or products[i-1] != p]
            self.assertEqual(unique, list(batch_order()))

    def test_hypervolume_and_duplicate_points(self):
        self.assertAlmostEqual(hypervolume([[1, 2], [2, 1], [1, 2], [2.5, 2.5]], [3, 3]), 3.)
        self.assertEqual(hypervolume([], [3, 3]), 0)

    def test_missing_full_results_and_smoke_export_are_rejected(self):
        from .export import export_paper
        with tempfile.TemporaryDirectory(prefix="industrial_gate_") as directory:
            with self.assertRaisesRegex(ValueError, "Missing"):
                checked_records("full", directory)
            with self.assertRaisesRegex(ValueError, "Smoke"):
                export_paper("smoke", directory)

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        import torch
        from .learning import train
        with tempfile.TemporaryDirectory(prefix="industrial_resume_") as directory:
            direct = Path(directory) / "direct"
            resumed = Path(directory) / "resumed"
            train("smoke", direct, methods=["HCMAGRL"])
            train("smoke", resumed, stop_after=1, methods=["HCMAGRL"])
            train("smoke", resumed, resume=True, methods=["HCMAGRL"])
            checkpoints = [torch.load(p / "smoke/checkpoints/HCMAGRL_w0.5_s42.pt", map_location="cpu", weights_only=False)
                           for p in [direct, resumed]]
            self.assertTrue(all(p["completed"] and p["iteration"] == 2 for p in checkpoints))
            self.assertTrue(checkpoints[0]["optimizers"])
            for key, tensor in checkpoints[0]["model_state_dict"].items():
                torch.testing.assert_close(tensor, checkpoints[1]["model_state_dict"][key], rtol=0, atol=0)
            # A changed profile cannot masquerade as an existing run.
            from .common import initialize_run, read_json, write_json
            manifest = resumed / "smoke/manifest.json"
            state = read_json(manifest)
            state["signature"] = "mismatched"
            write_json(manifest, state)
            with self.assertRaisesRegex(ValueError, "different"):
                initialize_run("smoke", resumed)

    def test_windows_spawn_workers_match_serial_rollouts(self):
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        from .learning import configure, make_agent, worker
        reference = run_policy("smoke", rule("SPT"))
        refs = {key: reference[key] for key in ["makespan", "cost"]}
        configure(.5, refs)
        agent = make_agent("smoke", "HCMAGRL", "cpu")
        state = {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()}
        arguments = [("smoke", "HCMAGRL", .5, refs, state, seed) for seed in [901, 902]]
        serial = [worker(*args)[1] for args in arguments]
        with ProcessPoolExecutor(2, mp_context=get_context("spawn")) as pool:
            futures = [pool.submit(worker, *args) for args in arguments]
            parallel = [future.result()[1] for future in futures]
        for a, b in zip(serial, parallel):
            for key in ["feasible", "makespan", "cost", "reward", "steps"]:
                self.assertEqual(a[key], b[key])


if __name__ == "__main__":
    unittest.main()
