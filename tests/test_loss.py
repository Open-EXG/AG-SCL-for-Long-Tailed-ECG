import io
import unittest

import torch

from ag_scl.loss import AGSCLLoss, AdaptiveLogitAdjustment
from ag_scl.statistics import CountBasedCovarianceShrinkage, OnlineClassMomentEstimator
from tests.common import tiny_config


class LossTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.config = tiny_config()

    def test_ala_initial_scale_and_bounds(self) -> None:
        ala = AdaptiveLogitAdjustment((0.9, 0.1))
        self.assertTrue(torch.allclose(ala.scales, torch.ones_like(ala.scales)))
        with torch.no_grad():
            ala.gamma.fill_(100)
        self.assertTrue(torch.all(ala.scales <= 3.0))
        with torch.no_grad():
            ala.gamma.fill_(-100)
        self.assertTrue(torch.all(ala.scales >= 0.5))

    def test_count_based_shrinkage_equation(self) -> None:
        module = CountBasedCovarianceShrinkage(tau=8.0)
        covariance = torch.zeros(2, 3, 3)
        counts = torch.tensor([0.0, 8.0])
        actual = module(covariance, counts)
        self.assertTrue(torch.allclose(actual[0], torch.eye(3)))
        self.assertTrue(torch.allclose(actual[1], torch.exp(torch.tensor(-1.0)) * torch.eye(3)))

    def test_loss_composition_and_shapes(self) -> None:
        criterion = AGSCLLoss(self.config.loss)
        raw_logits = torch.randn(4, 2, 2, requires_grad=True)
        projected = torch.nn.functional.normalize(torch.randn(2, 4, 2, 8), dim=-1)
        labels = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]])
        output = criterion(raw_logits, projected, labels)
        self.assertTrue(torch.allclose(output.total, output.classification + output.contrastive))
        output.total.backward()
        self.assertIsNotNone(raw_logits.grad)

    def test_moment_buffers_survive_state_dict(self) -> None:
        estimator = OnlineClassMomentEstimator(4, shrinkage_tau=8.0)
        estimator.update(torch.randn(4, 4), torch.tensor([0, 0, 1, 1]))
        buffer = io.BytesIO()
        torch.save(estimator.state_dict(), buffer)
        buffer.seek(0)
        restored = OnlineClassMomentEstimator(4, shrinkage_tau=8.0)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        self.assertTrue(torch.equal(estimator.count, restored.count))
        self.assertTrue(torch.equal(estimator.mean, restored.mean))
        self.assertTrue(torch.equal(estimator.covariance, restored.covariance))


if __name__ == "__main__":
    unittest.main()
