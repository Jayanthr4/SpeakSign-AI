import json
import tempfile
import unittest
from pathlib import Path
from ml.recognition.inspect_gesturesign import inspect

class GestureSignMetadataTests(unittest.TestCase):
    def test_rejects_incompatible_schema_before_loading_model(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            model=root/"model.onnx"
            model.write_bytes(b"placeholder")
            meta=root/"meta.json"
            meta.write_text(json.dumps({"featureVersion":1,"featureSet":"hands_pose","featureDim":100}))
            with self.assertRaisesRegex(ValueError,"Unsupported"):
                inspect(model,meta)

if __name__=="__main__":
    unittest.main()
