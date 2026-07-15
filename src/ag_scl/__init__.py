"""Public API for Angular Gaussian Supervised Contrastive Learning."""

from .augmentation import TailAwareMultiViewAugmenter, fft_magnitude_phase, window_signal
from .config import AGSCLConfig
from .loss import AGSCLLoss, AdaptiveLogitAdjustment, AngularGaussianScorer
from .method import AGSCL, AGSCLTrainingOutput
from .model import SwinECGEncoder

__all__ = [
    "AGSCL",
    "AGSCLConfig",
    "AGSCLLoss",
    "AGSCLTrainingOutput",
    "AdaptiveLogitAdjustment",
    "AngularGaussianScorer",
    "SwinECGEncoder",
    "TailAwareMultiViewAugmenter",
    "fft_magnitude_phase",
    "window_signal",
]

__version__ = "0.1.0"

