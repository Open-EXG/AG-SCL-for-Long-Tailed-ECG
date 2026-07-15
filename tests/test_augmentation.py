import unittest

import torch

from ag_scl.augmentation import TailAwareMultiViewAugmenter, fft_magnitude_phase, window_signal
from ag_scl.transforms import (
    BandConstrainedMagnitudeScaling,
    SignalNegation,
    positive_record_mask,
)
from tests.common import tiny_config


class AugmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = tiny_config()

    def test_fft_matches_reference_equation(self) -> None:
        ecg = torch.randn(3, 1, 32)
        windowed = window_signal(ecg, 4)
        actual = fft_magnitude_phase(windowed)
        spectrum = torch.fft.fft(windowed, dim=-1)
        expected = torch.cat((spectrum.abs(), spectrum.angle() / torch.pi), dim=1)
        self.assertTrue(torch.allclose(actual, expected))
        self.assertEqual(tuple(actual.shape), (3, 2, 4, 8))

    def test_any_positive_label_is_eligible(self) -> None:
        labels = torch.tensor([[0, 0], [1, 0], [0, 1]])
        self.assertEqual(positive_record_mask(labels).tolist(), [False, True, True])

    def test_all_zero_records_remain_unchanged(self) -> None:
        augmenter = TailAwareMultiViewAugmenter(self.config)
        ecg = torch.randn(4, 1, 32)
        labels = torch.zeros(4, 2, dtype=torch.long)
        views = augmenter(ecg, labels)
        self.assertEqual(tuple(views.shape), (3, 4, 2, 4, 8))
        self.assertTrue(torch.equal(views[0], views[1]))
        self.assertTrue(torch.equal(views[0], views[2]))

    def test_negation_is_sample_wise_and_tail_aware(self) -> None:
        transform = SignalNegation(probability=1.0)
        inputs = torch.ones(3, 1, 4, 8)
        labels = torch.tensor([[0, 0], [1, 0], [0, 1]])
        output = transform(inputs, labels)
        self.assertTrue(torch.equal(output[0], inputs[0]))
        self.assertTrue(torch.equal(output[1:], -inputs[1:]))

    def test_removed_low_frequency_band_is_unchanged(self) -> None:
        transform = BandConstrainedMagnitudeScaling(probability=1.0, sampling_rate=100.0)
        inputs = torch.ones(2, 2, 4, 20)
        labels = torch.ones(2, 1, dtype=torch.long)
        output = transform(inputs, labels)
        frequencies = torch.fft.fftfreq(20, d=0.01).abs()
        low = frequencies < 0.5
        self.assertTrue(torch.equal(output[:, 0, :, low], inputs[:, 0, :, low]))
        self.assertTrue(torch.equal(output[:, 1], inputs[:, 1]))


if __name__ == "__main__":
    unittest.main()
