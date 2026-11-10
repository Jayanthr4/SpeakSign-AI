"""Small regression tests for V2.6 sequence QA (standard library only)."""
import json
import tempfile
import unittest
from pathlib import Path

from ml.person_tracking.sequence_engine import process


class SequenceEngineTests(unittest.TestCase):
    def test_multiple_signers_and_missing_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/"input.jsonl"
            rows=[
                {"frame":i,"time_seconds":i/10,
                 "signers":[{"track_id":tid,"visible":True,
                             "hands":[],"pose":[],"face":[]}
                            for tid in ([1,3,8] if i!=2 else [1,8])]}
                for i in range(5)
            ]
            source.write_text("".join(json.dumps(row)+"\n" for row in rows))
            process(source,root/"out",3,1,0.6)
            summary=json.loads((root/"out/sequence_quality.json").read_text())
            self.assertEqual(set(summary["signers"]),{"1","3","8"})
            self.assertEqual(summary["signers"]["3"]["visible_frames"],4)
            windows=[json.loads(x) for x in (root/"out/temporal_windows.jsonl").read_text().splitlines()]
            self.assertEqual(len(windows),9)
            missing=[x for x in windows if x["track_id"]==3 and x["start_frame"]==1][0]
            self.assertFalse(missing["frames"][1]["visible"])
            self.assertEqual(missing["frames"][1]["hands"],[])

    def test_invalid_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError):
                process(root/"absent",root/"out",0,1,0.5)


if __name__=="__main__":
    unittest.main()
