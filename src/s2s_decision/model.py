"""Small three-branch network. Outputs activity logit and standardized pActivity."""

import torch
from torch import nn


def _branch(inputs, hidden, outputs):
    return nn.Sequential(
        nn.Linear(inputs, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, outputs), nn.ReLU()
    )


class Tiny(nn.Module):
    def __init__(self, fingerprint_bits=2048):
        super().__init__()
        if fingerprint_bits not in (1024, 2048):
            raise ValueError("fingerprint_bits must be 1024 or 2048")
        self.properties = _branch(40, 32, 16)
        self.fingerprint = _branch(fingerprint_bits, 128, 64)
        self.context = _branch(14, 32, 16)
        self.fusion = _branch(96, 64, 32)
        self.heads = nn.Linear(32, 2)

    def forward(self, properties, fingerprint, context):
        fused = torch.cat(
            (self.properties(properties), self.fingerprint(fingerprint), self.context(context)),
            dim=1,
        )
        return self.heads(self.fusion(fused))
