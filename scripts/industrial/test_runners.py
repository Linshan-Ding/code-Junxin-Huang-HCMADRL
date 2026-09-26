"""Entry contracts, including a real isolated smoke run; never full training.

Run: python -m unittest discover -s scripts/industrial -p test_runners.py -v
"""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import _bootstrap as runner
import _settings


HERE = Path(__file__).resolve().parent


def hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file()}


class RunnerTests(unittest.TestCase):
    def settings(self, directory):
        return runner.Settings(Path(directory) / "result with spaces", "cpu", Path(directory) / "paper with spaces")

    def test_command_contracts_keep_existing_cli_and_protocol(self):
        settings = self.settings(".")
        commands = {s: runner.cli_commands(s, settings) for s in runner.STAGES if s != "08"}
        self.assertEqual(commands["00"][:2], [["prepare"], ["validate"]])
        self.assertIn("smoke", commands["00"][2])
        self.assertEqual(commands["02"][0][-2:], ["--methods", "HCMAGRL"])
        self.assertEqual(commands["03"][0][-3:], ["--methods", "Flat", "MLP"])
        for stage in ["00", "02", "03", "04", "05"]:
            self.assertIn("--resume", commands[stage][-1])
        for stage in ["00", "02", "03", "05"]:
            args = commands[stage][-1]
            self.assertEqual(args[args.index("--device") + 1], "cpu")
        for stage in ["02", "03", "04", "05", "06", "07"]:
            self.assertIn("full", commands[stage][0])
            self.assertIn(str(runner.absolute(settings.result_root)), commands[stage][0])
        self.assertNotIn("--draft", commands["07"][0])
        self.assertEqual(runner.DATA_STAGES, ("00", "01", "02", "03", "04", "05", "06"))

    def test_zero_argument_scripts_dispatch_their_stage(self):
        scripts = sorted(HERE.glob("run_*.py"))
        self.assertEqual(len(scripts), 10)
        for script in scripts:
            stage = "all" if script.stem == "run_all" else script.stem.split("_")[1]
            with patch.object(runner, "entry") as entry:
                runpy.run_path(str(script), run_name="__main__")
                entry.assert_called_once_with(stage)
        with patch.object(sys, "argv", ["run_all.py", "--help"]), patch.object(runner, "main") as main:
            with self.assertRaises(SystemExit):
                runner.entry("all")
            main.assert_not_called()

    def test_run_all_stops_on_failure_and_preserves_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = self.settings(directory)
            called = []
            def fail(stage, settings, log):
                called.append(stage)
                self.assertEqual(Path.cwd(), runner.ROOT)
                if stage == "02":
                    raise runner.StepError("deliberate failure", 7)
            previous = Path.cwd()
            with patch.object(runner, "execute_stage", side_effect=fail), redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    runner.main("all", settings)
            self.assertEqual(error.exception.code, 7)
            self.assertEqual(called, ["00", "01", "02"])
            self.assertEqual(Path.cwd(), previous)
            log = next((settings.result_root / "runner_logs").glob("*.log")).read_text(encoding="utf-8")
            self.assertIn("deliberate failure", log)
            self.assertIn("[RECOVERY]", log)

    def test_run_all_success_includes_only_data_stages(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "execute_stage") as execute, redirect_stdout(io.StringIO()):
            runner.main("all", self.settings(directory))
            self.assertEqual([call.args[0] for call in execute.call_args_list], list(runner.DATA_STAGES))

    def test_subprocess_paths_output_and_nonzero_status(self):
        with tempfile.TemporaryDirectory(prefix="industrial runner ") as directory:
            target = Path(directory) / "script with spaces.py"
            target.write_text("import sys\nprint('visible-output', flush=True)\nsys.exit(7)\n", encoding="utf-8")
            log = io.StringIO()
            with redirect_stdout(io.StringIO()), self.assertRaises(runner.StepError) as error:
                runner.run_command([sys.executable, "-B", str(target)], directory, log)
            self.assertEqual(error.exception.code, 7)
            self.assertIn("visible-output", log.getvalue())

    def test_compiler_priority_and_missing_compiler(self):
        with tempfile.TemporaryDirectory() as directory:
            explicit = Path(directory) / "tectonic.exe"
            explicit.touch()
            bundled = Path(directory) / "bundled" / "tectonic.exe"
            bundled.parent.mkdir()
            bundled.touch()
            with patch.object(runner.shutil, "which", return_value="path-tectonic") as which:
                self.assertEqual(runner.find_compiler(str(explicit)), (explicit, "tectonic"))
                which.assert_not_called()
                self.assertEqual(runner.find_compiler(), (Path("path-tectonic"), "tectonic"))
            with patch.object(runner.shutil, "which", side_effect=lambda name: "path-latexmk" if name == "latexmk" else None):
                with patch.object(runner, "bundled_tectonics", return_value=[bundled]):
                    self.assertEqual(runner.find_compiler(), (bundled, "tectonic"))
                with patch.object(runner, "bundled_tectonics", return_value=[]):
                    self.assertEqual(runner.find_compiler(), (Path("path-latexmk"), "latexmk"))
            with patch.object(runner.shutil, "which", return_value=None), patch.object(runner, "bundled_tectonics", return_value=[]):
                with self.assertRaisesRegex(runner.StepError, "No Tectonic"):
                    runner.find_compiler()
            with self.assertRaisesRegex(runner.StepError, "does not exist"):
                runner.find_compiler(Path(directory) / "missing.exe")

    def test_compile_keeps_logs_and_checks_only_after_success(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = self.settings(directory)
            settings.paper_root.mkdir()
            (settings.paper_root / "main.tex").write_text("draft", encoding="utf-8")
            with patch.object(runner, "find_compiler", return_value=(Path("tectonic"), "tectonic")):
                commands = runner.compile_commands(settings)
                self.assertIn("--keep-logs", commands[0][0])
                self.assertIn("--keep-intermediates", commands[0][0])
                self.assertEqual(commands[0][1], settings.paper_root)
                self.assertEqual(commands[1][0][0], sys.executable)
                self.assertTrue(any(x.endswith("check.py") for x in commands[1][0]))
                with patch.object(runner, "run_command", side_effect=runner.StepError("compile failed")) as command, redirect_stdout(io.StringIO()):
                    with self.assertRaises(runner.StepError):
                        runner.execute_stage("08", settings, io.StringIO())
                    self.assertEqual(command.call_count, 1)

    def test_expected_full_output_counts(self):
        with runner.repository_context():
            settings = self.settings(".")
            ours = runner.required_outputs("02", settings)
            ablations = runner.required_outputs("03", settings)
            baselines = runner.required_outputs("04", settings)
            evaluated = runner.required_outputs("05", settings)
        self.assertEqual(sum(p.suffix == ".pt" for p in ours), 25)
        self.assertEqual(sum(p.suffix == ".pt" for p in ablations), 50)
        self.assertEqual(sum(p.parent.name == "evaluations" for p in baselines), 32)
        self.assertEqual(sum(p.parent.name == "evaluations" for p in evaluated), 75)
        self.assertEqual(sum(p.parent.name == "nsga" and p.suffix == ".json" for p in baselines), 5)

    def test_missing_outputs_fail_a_successful_child(self):
        with tempfile.TemporaryDirectory() as directory, runner.repository_context(), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(runner.StepError, "Expected outputs missing"):
                runner.verify_outputs("02", self.settings(directory), io.StringIO())

    def test_real_smoke_entry_resume_and_export_guards(self):
        with tempfile.TemporaryDirectory(prefix="industrial runner integration ") as directory:
            settings = self.settings(directory)
            settings.paper_root.mkdir()
            (settings.paper_root / "main.tex").write_text("Draft must remain untouched.", encoding="utf-8")
            script = HERE / "run_00_smoke.py"
            previous = Path.cwd()
            # Run the actual public script from an unrelated cwd, with isolated
            # settings injected in memory; no repository settings are edited.
            with patch.object(_settings, "RESULT_ROOT", settings.result_root), patch.object(_settings, "PAPER_ROOT", settings.paper_root), patch.object(sys, "argv", [str(script)]), redirect_stdout(io.StringIO()):
                import os
                os.chdir(directory)
                try:
                    runpy.run_path(str(script), run_name="__main__")
                    before = hashes(settings.result_root / "smoke")
                    runpy.run_path(str(script), run_name="__main__")
                    self.assertEqual(before, hashes(settings.result_root / "smoke"))
                finally:
                    os.chdir(previous)
                smoke = settings.result_root / "smoke"
                self.assertEqual(len(list((smoke / "evaluations").glob("*.json"))), 7)
                self.assertTrue(all(json.loads(p.read_text())["feasible"] for p in (smoke / "evaluations").glob("*.json")))
                self.assertEqual(len(list((settings.result_root / "runner_logs").glob("*.log"))), 2)
                metadata = json.loads((smoke / "aggregate/manifest.json").read_text())
                self.assertEqual(metadata["expected_evaluations"], 7)
                manifest = smoke / "manifest.json"
                altered = json.loads(manifest.read_text())
                altered["signature"] = "deliberately-incompatible-test"
                manifest.write_text(json.dumps(altered), encoding="utf-8")
                mismatched = hashes(smoke)
                with self.assertRaises(SystemExit):
                    runpy.run_path(str(script), run_name="__main__")
                self.assertEqual(mismatched, hashes(smoke))
                paper_before = hashes(settings.paper_root)
                with self.assertRaises(SystemExit):
                    runner.main("07", settings)
                self.assertEqual(paper_before, hashes(settings.paper_root))
                self.assertFalse((settings.result_root / "full/checkpoints").exists())
                log = io.StringIO()
                with self.assertRaises(runner.StepError):
                    runner.run_command([sys.executable, "-B", "-m", "industrial_case", "export-paper",
                                        "--profile", "smoke", "--output", str(settings.result_root),
                                        "--paper", str(settings.paper_root)], runner.ROOT, log)
                self.assertEqual(paper_before, hashes(settings.paper_root))
