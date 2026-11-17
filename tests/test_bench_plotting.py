"""A partially completed run must never appear as a complete comparison."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"bench"))
from campaign_plots import plot


class PlotCoverageTest(unittest.TestCase):
    def test_missing_held_out_seed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/"evaluation.csv").write_text(
                "M,K,N,sparsity,threads,seed,method,selected,fixed_scaling\n"
                "64,128,256,0.5,1,43,SPR v8,1,1\n")
            (root/"context.json").write_text(json.dumps(dict(cpu="",hostname="local",settings="pilot")))
            (root/"peaks.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"Missing held-out measurement"):
                plot(root, root/"plots")


if __name__ == "__main__":
    unittest.main()
