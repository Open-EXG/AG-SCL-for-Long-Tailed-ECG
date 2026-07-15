"""Small configuration shared by CPU unit tests."""

from ag_scl.config import (
    AGSCLConfig,
    AugmentationConfig,
    DataConfig,
    LossConfig,
    ModelConfig,
    TransformConfig,
)


def tiny_config() -> AGSCLConfig:
    config = AGSCLConfig(
        data=DataConfig(
            num_labels=2,
            signal_length=32,
            num_windows=4,
            sampling_rate=8.0,
            input_channels=1,
        ),
        model=ModelConfig(
            patch_size=(1, 2),
            window_size=(2, 2),
            embed_dim=4,
            depths=(1, 1),
            num_heads=2,
            feature_dim=8,
            projection_dim=8,
            dropout=0.0,
            attention_dropout=0.0,
            drop_path_rate=0.0,
        ),
        loss=LossConfig(
            embedding_dim=8,
            contrastive_weight=1.0,
            contrastive_temperature=0.2,
            shrinkage_tau=8.0,
            label_state_priors=((0.9, 0.1), (0.99, 0.01)),
        ),
        augmentation=AugmentationConfig(
            transforms=TransformConfig(
                signal_negation=0.1,
                global_amplitude_scaling=0.5,
                additive_jitter=0.5,
                frequency_masking=0.5,
                band_constrained_phase_jitter=0.5,
                band_constrained_magnitude_scaling=0.5,
            )
        ),
    )
    config.validate()
    return config

