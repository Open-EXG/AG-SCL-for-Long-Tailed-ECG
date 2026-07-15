import io
import unittest

import torch

from ag_scl import AGSCL
from tests.common import tiny_config


class MethodTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(13)
        self.config = tiny_config()

    def test_training_shapes_and_normalized_projections(self) -> None:
        method = AGSCL(self.config).train()
        ecg = torch.randn(4, 1, 32)
        labels = torch.tensor([[1, 0], [0, 1], [0, 0], [1, 1]])
        views = method.augmenter(ecg, labels)
        logits, projections = method.model.forward_views(views)
        self.assertEqual(tuple(logits.shape), (4, 2, 2))
        self.assertEqual(tuple(projections.shape), (2, 4, 2, 8))
        self.assertTrue(torch.allclose(projections.norm(dim=-1), torch.ones(2, 4, 2), atol=1e-5))
        output = method.training_step(ecg, labels)
        self.assertEqual(tuple(output.raw_logits.shape), (4, 2, 2))
        self.assertTrue(torch.isfinite(output.loss))

    def test_ala_parameters_belong_to_top_level_module(self) -> None:
        method = AGSCL(self.config)
        parameter_ids = {id(parameter) for parameter in method.parameters()}
        for label_loss in method.criterion.label_losses:
            self.assertIn(id(label_loss.ala.gamma), parameter_ids)

    def test_inference_does_not_update_moments(self) -> None:
        method = AGSCL(self.config).eval()
        counts_before = [loss.scorer.history.count.clone() for loss in method.criterion.label_losses]
        with torch.no_grad():
            logits = method(torch.randn(3, 1, 32))
        self.assertEqual(tuple(logits.shape), (3, 2, 2))
        for before, label_loss in zip(counts_before, method.criterion.label_losses):
            self.assertTrue(torch.equal(before, label_loss.scorer.history.count))

    def test_complete_checkpoint_round_trip(self) -> None:
        method = AGSCL(self.config).train()
        method.begin_epoch()
        method.training_step(
            torch.randn(4, 1, 32),
            torch.tensor([[1, 0], [0, 1], [0, 0], [1, 1]]),
        )
        buffer = io.BytesIO()
        torch.save(method.state_dict(), buffer)
        buffer.seek(0)
        restored = AGSCL(self.config)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        for key, value in method.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)


if __name__ == "__main__":
    unittest.main()
