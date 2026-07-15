"""Run synthetic raw-logit inference with AG-SCL."""

from pathlib import Path

import torch

from ag_scl import AGSCL, AGSCLConfig


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = AGSCLConfig.from_yaml(root / "configs" / "ptbxl_ag_scl.yaml")
    method = AGSCL(config).eval()
    ecg = torch.randn(2, 1, config.data.signal_length)
    with torch.no_grad():
        logits = method(ecg)
    print(f"raw logits: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()

