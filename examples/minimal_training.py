"""Run one synthetic AG-SCL optimization step."""

from pathlib import Path

import torch

from ag_scl import AGSCL, AGSCLConfig


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = AGSCLConfig.from_yaml(root / "configs" / "ptbxl_ag_scl.yaml")
    method = AGSCL(config)
    method.train()
    method.begin_epoch()

    optimizer = torch.optim.AdamW(method.parameters(), lr=3e-4, weight_decay=0.02)
    ecg = torch.randn(2, 1, config.data.signal_length)
    labels = torch.tensor([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])

    output = method.training_step(ecg, labels)
    optimizer.zero_grad()
    output.loss.backward()
    optimizer.step()
    print(f"loss={output.loss.item():.6f}, logits={tuple(output.raw_logits.shape)}")


if __name__ == "__main__":
    main()

