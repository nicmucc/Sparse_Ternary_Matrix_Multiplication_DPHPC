"""CPU-only checks for GPU campaign accounting, correctness gating and plotting data."""
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "bench" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


runner = module("run_gpu_compare")
plotter = module("plot_gpu_compare")


def samples(version, M=1, K=17, N=7, k=3, zero=.5, count=3):
    scale = 2 if version in ("v1", "v2") else 1
    result = []
    for metric in ("prepare_wall", *plotter.METRICS):
        for sample in range(1 if metric == "prepare_wall" else count):
            result.append(dict(version=version, M=str(M), K=str(K), N=str(N), k=str(k),
                               zero_fraction=str(zero), seed="42", metric=metric, sample=str(sample),
                               ms=str(scale * (sample + 1)), max_abs_error="0"))
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture(root, versions=("v2", "v3")):
    rows = [r for version in versions for M in (1, 33) for r in samples(version, M)]
    write_csv(root / "raw_samples.csv", rows)
    (root / "environment.json").write_text(json.dumps(dict(versions=versions, status="complete",
        suite="smoke", samples=3, configurations_per_version=2, gpu="Synthetic unit-test data")))
    (root / "failures.json").write_text("[]")
    runner.summarize(rows, root, versions)
    return rows


class GPUComparisonTest(unittest.TestCase):
    def test_pair_labels_speedup_and_summary_validation(self):
        for candidate in ("v3", "v4"):
            with self.subTest(candidate=candidate), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                fixture(root, ("v2", candidate))
                env, versions, workloads, _, raw = plotter.load_results(root)
                self.assertEqual(versions, ("v2", candidate))
                self.assertEqual(len(workloads), 2)
                self.assertEqual(len(raw), 40)
                with (root / "speedups.csv").open() as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 6)
                self.assertTrue(all(float(r["speedup"]) == 2 for r in rows))
                self.assertIn(f"{candidate}_ms", rows[0])
                html = (root / "comparison.html").read_text()
                self.assertIn(f"Above 1 favors {candidate}", html)
                self.assertNotIn("v1", html)
                self.assertNotIn("v3" if candidate == "v4" else "v4", html)

    def test_previous_v1_v2_pair(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = samples("v1") + samples("v2")
            speedups = runner.summarize(rows, root, ("v1", "v2"))
            self.assertTrue(all(float(r["speedup"]) == 1 for r in speedups))
            self.assertIn("v1_ms", speedups[0])

    def test_incomplete_or_failed_campaign_rejected(self):
        for status, failures in (("benchmarking", []), ("complete", [{"case": "failed"}])):
            with self.subTest(status=status, failures=failures), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                fixture(root)
                env = json.loads((root / "environment.json").read_text())
                env["status"] = status
                (root / "environment.json").write_text(json.dumps(env))
                (root / "failures.json").write_text(json.dumps(failures))
                with self.assertRaises(ValueError):
                    plotter.load_results(root)

    def test_missing_version_samples_and_duplicate_ids_rejected(self):
        for kind in ("missing_version", "duplicate_id", "missing_workload"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                rows = fixture(root)
                if kind == "missing_version":
                    rows = [r for r in rows if not (r["version"] == "v3" and r["M"] == "33")]
                elif kind == "duplicate_id":
                    rows[2]["sample"] = rows[1]["sample"]
                else:
                    rows = [r for r in rows if r["M"] != "33"]
                write_csv(root / "raw_samples.csv", rows)
                with self.assertRaises(ValueError):
                    plotter.load_results(root)

    def test_inconsistent_summary_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root)
            rows = plotter.read_csv(root / "timing_summary.csv")
            rows[0]["q1_ms"] = "999"
            write_csv(root / "timing_summary.csv", rows)
            with self.assertRaisesRegex(ValueError, "differs"):
                plotter.load_results(root)

    def test_launch_instrumentation(self):
        for version in ("v2", "v3", "v4"):
            source = (ROOT / "gpu" / "spr" / f"spr_{version}.cu").read_text()
            result = runner.instrument(source)
            self.assertEqual(result.count("spr_bench::mark_start();"), 1)
            self.assertEqual(result.count("spr_bench::mark_stop();"), 1)

    def campaign(self, root, fail_correctness=False, versions=("v2", "v3")):
        calls = []
        def command(argv, **kwargs):
            argv = [str(a) for a in argv]
            calls.append(argv)
            program = Path(argv[0]).name
            if fail_correctness and program == "test_gpu_correctness":
                return subprocess.CompletedProcess(argv, 1, "", "reference mismatch")
            if program in ("bench_v2", "bench_v3", "bench_v4"):
                version, M, K, N, k, zero, seed, count, profile = argv[1:]
                output = io.StringIO()
                rows = samples(version, int(M), int(K), int(N), int(k), float(zero), int(count))
                writer = csv.DictWriter(output, list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
                return subprocess.CompletedProcess(argv, 0, output.getvalue(), "PASS")
            return subprocess.CompletedProcess(argv, 0, "PASS", "")
        args = ["run_gpu_compare.py", "--versions", *versions, "--suite", "smoke",
                "--samples", "3", "--output", str(root)]
        with patch.object(sys, "argv", args), patch.object(runner, "command", side_effect=command), contextlib.redirect_stdout(io.StringIO()):
            if fail_correctness:
                with self.assertRaisesRegex(RuntimeError, "Command failed"):
                    runner.main()
            else:
                self.assertEqual(runner.main(), 0)
        return calls

    def test_smoke_campaign_accounting(self):
        for candidate in ("v3", "v4"):
            with self.subTest(candidate=candidate), tempfile.TemporaryDirectory() as d:
                root = Path(d) / "run"
                calls = self.campaign(root, versions=("v2", candidate))
                checks = [c for c in calls if Path(c[0]).name in ("test_correctness", "test_gpu_correctness")]
                self.assertEqual(len(checks), 4)
                self.assertEqual({c[1] for c in checks}, {"spr.v2.cuda", f"spr.{candidate}.cuda"})
                self.assertEqual(sum(Path(c[0]).name in ("bench_v2", f"bench_{candidate}") for c in calls), 12)
                env, versions, workloads, _, raw = plotter.load_results(root)
                self.assertEqual(versions, ("v2", candidate))
                self.assertEqual(len(workloads), 6)
                self.assertEqual(len(raw), 120)
                self.assertEqual(env["status"], "complete")

    def test_correctness_failure_prevents_benchmarking(self):
        for candidate in ("v3", "v4"):
            with self.subTest(candidate=candidate), tempfile.TemporaryDirectory() as d:
                root = Path(d) / "run"
                calls = self.campaign(root, fail_correctness=True, versions=("v2", candidate))
                self.assertFalse(any(Path(c[0]).name in ("bench_v2", "bench_v3", "bench_v4") or
                                     (Path(c[0]).name == "nvcc" and "--version" not in c) for c in calls))
                self.assertFalse((root / "comparison.html").exists())
                self.assertFalse((root / "raw_samples.csv").exists())


if __name__ == "__main__":
    unittest.main()
