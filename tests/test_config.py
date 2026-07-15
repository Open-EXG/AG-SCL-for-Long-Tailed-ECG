from pathlib import Path
import unittest

from ag_scl import AGSCLConfig


class ConfigTests(unittest.TestCase):
    def test_public_yaml(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = AGSCLConfig.from_yaml(root / "configs" / "ptbxl_ag_scl.yaml")
        self.assertEqual(config.method, "ag_scl")
        self.assertEqual(config.data.num_labels, 6)
        self.assertEqual(config.data.window_length, 20)
        self.assertEqual(config.augmentation.transforms.signal_negation, 0.1)
        self.assertEqual(config.augmentation.transforms.frequency_masking, 0.5)
        self.assertEqual(config.loss.label_state_priors[3][1], 0.01)


if __name__ == "__main__":
    unittest.main()

