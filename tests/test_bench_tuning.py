"""Regression tests for verified, two-stage tuning. Run after make build."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import tune

class TuningTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw = Path(self.tmp.name)/"raw.csv"
        self.cache = Path(self.tmp.name)/"cache.csv"

    def write(self, rows):
        with self.raw.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    def row(self, kernel, ms, k="k=4", **extra):
        return dict(tag="tuning", machine="test", compiler="gcc", threads="4", M="64", K="128", N="256",
                    sparsity="0.5", kernel=kernel, param=k, median_ms=str(ms), ok="1", max_err="0", **extra)

    def test_v8_variant_and_k_selected_together_but_baselines_kept(self):
        self.write([self.row("spr.v8.b16", 3, "k=2"), self.row("spr.v8.b32", 1, "k=4"),
                    self.row("spr.v8.b32.l2", 2, "k=8"), self.row("rsr.rsr_pp_inference", 4)])
        tune.add(self.raw, self.cache, include_spr=True)
        selection = tune.get(self.cache, "64", "128", "256", "0.5")
        self.assertIn("=spr.v8.b32@4", selection)
        self.assertNotIn("spr.v8.b16", selection)
        self.assertIn("=rsr.rsr_pp_inference@4", selection)
        self.assertIn("=spr.v8.b32@4", tune.get(self.cache, "64", "128", "256", "0.5", "opt"))
        self.assertEqual(tune.get(self.cache, "1", "128", "256", "0.5"), "")

    def test_unverified_wrong_nan_or_empty_data_cannot_be_cached(self):
        for field, value in [("max_err", "-1"), ("max_err", "nan"), ("ok", "0"), ("median_ms", "nan"), ("median_ms", "inf"), ("max_err", "inf")]:
            row=self.row("spr.v8.b32", 1);row[field]=value;self.write([row])
            with self.assertRaises(ValueError): tune.add(self.raw, self.cache, True)
        self.raw.write_text("kernel,ok,max_err,median_ms\n")
        with self.assertRaises(ValueError): tune.check(self.raw)

    def test_thread_contexts_cannot_be_pooled(self):
        rows=[self.row("spr.v8.b16", 2), self.row("spr.v8.b32", 1)]
        rows[1]["threads"]="1";self.write(rows)
        with self.assertRaisesRegex(ValueError,"mixed"): tune.add(self.raw,self.cache,True)

    def test_gemv_and_dense_upstream_grouping(self):
        self.assertEqual(tune.method("rsr.vector_rsr_pp_inference"), "RSR++ · scalar")
        self.assertEqual(tune.method("tcsc.GEMM_CPU_FP32_colMajor_Direct_OpenMP"), "Dense · upstream")

    def test_boundary_optimum_is_reported(self):
        self.write([self.row("spr.v8.b16", 3, "k=2"), self.row("spr.v8.b32", 1, "k=8")])
        out=io.StringIO()
        with contextlib.redirect_stdout(out): tune.edges(self.raw)
        self.assertIn("k=8 at boundary", out.getvalue())

    def test_scaling_freezes_one_thread_tuning_choice(self):
        root=Path(self.tmp.name)
        source=root/"comparison"/"run";source.mkdir(parents=True)
        (source/"summary.md").touch()
        (source/"context.json").write_text(json.dumps({"settings":"threads=1|test"}))
        self.write([self.row("spr.v8.b16",1,"k=2"),self.row("spr.v8.b32",2,"k=4")])
        tune.add(self.raw, source/"selected.csv", True)
        for thread, times in [("1",(10,8)),("4",(5,1))]:
            path=root/"roofline"/thread;path.mkdir(parents=True)
            for name in ("report.md","roofline.png","runtime.png"): (path/name).touch()
            rows=[self.row("spr.v8.b16",times[0],"k=2"),self.row("spr.v8.b32",times[1],"k=4")]
            for row in rows: row["threads"]=thread
            self.write(rows);(path/"raw.csv").write_bytes(self.raw.read_bytes())
        with contextlib.redirect_stdout(io.StringIO()): tune.scaling(root)
        rows=tune.read_rows(root/"scaling.csv")
        result=next(r for r in rows if r["threads"]=="4")
        self.assertEqual(result["kernel"],"spr.v8.b16")
        self.assertEqual(float(result["speedup"]),2.0)

    def test_actual_driver_samples_and_wrong_result_detection(self):
        binary=os.environ.get("SPR_BENCH_BIN", str(ROOT/"build/release/bench_cpu"))
        samples=Path(self.tmp.name)/"samples.csv"
        common=[binary,"--M","8","--K","17","--N","16","--k","2","--threads","1",
                "--ref-rows","-1","--rounds","3","--warmup","1","--out",str(self.raw)]
        subprocess.run(common+["--kernels","=spr.v8.b16","--samples",str(samples)],check=True,stdout=subprocess.DEVNULL)
        row=tune.check(self.raw)[0]
        with samples.open() as f: times=sorted(float(r["ms"]) for r in csv.DictReader(f))
        self.assertEqual(len(times),3)
        self.assertAlmostEqual(float(row["median_ms"]),times[1],places=5)
        self.raw.unlink()
        subprocess.run(common+["--kernels","=ssr.ssr_inference"],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        with self.assertRaises(ValueError): tune.check(self.raw)

if __name__ == "__main__": unittest.main()
