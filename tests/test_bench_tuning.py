"""Regression tests for the fixed-tile benchmark adapter. Run after make build."""
import csv
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FixedTileTest(unittest.TestCase):
    def test_registered_tile_and_adapter_agree(self):
        # Reuse the same explicit config once through the upstream registry
        # (substring filter) and once through the benchmark factory (exact name).
        binary = os.environ.get("SPR_BENCH_BIN", ROOT/"build/release/bench_cpu")
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            args = [str(binary), "--M","64","--K","512","--N","64","--s","0.5",
                    "--seed","43","--threads","4","--ref-rows","-1","--rounds","1",
                    "--warmup","0","--k","1","--case-dir",d]
            rows = []
            for i,prefix in enumerate(("", "=")):
                path = tmp/f"result{i}.csv"
                args_i = ["--samples",str(tmp/"samples.csv")] if i else []
                subprocess.run(args+["--out",str(path),"--kernels",prefix+"spr.v8.b32.kc256.g8"]+args_i,
                               check=True, stdout=subprocess.DEVNULL)
                with path.open() as f:
                    rows.append(next(csv.DictReader(f)))
            upstream, adapter = rows
            self.assertEqual(upstream["ok"], "1")
            for key in ("max_err","ok","bytes","effective_threads"):
                self.assertEqual(adapter[key],upstream[key])
            # With k=1 each distinct nonzero pattern is a distinct sign.
            data = next(tmp.glob("*.bin")).read_bytes()
            w = data[4*64*512:4*64*512+512*64]
            segments = sum(len(set(w[col::64])-{0}) for col in range(64))
            self.assertEqual(float(adapter["adds_per_row"])-float(upstream["adds_per_row"]), segments)


if __name__ == "__main__":
    unittest.main()
