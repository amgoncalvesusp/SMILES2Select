"""Small three-branch network. Outputs activity logit and standardized pActivity."""

import torch
from torch import nn


def _branch(inputs, hidden, outputs):
    return nn.Sequential(
        nn.Linear(inputs, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, outputs), nn.ReLU()
    )


class Tiny(nn.Module):
    """Scale all internal widths; input features and two outputs remain unchanged."""

    def __init__(self, fingerprint_bits=2048, width_multiplier=1):
        super().__init__()
        if fingerprint_bits not in (1024, 2048):
            raise ValueError("fingerprint_bits must be 1024 or 2048")
        if type(width_multiplier) is not int or width_multiplier not in (1, 2):
            raise ValueError("width_multiplier must be integer 1 or 2")
        width = width_multiplier
        self.properties = _branch(40, 32 * width, 16 * width)
        self.fingerprint = _branch(fingerprint_bits, 128 * width, 64 * width)
        self.context = _branch(14, 32 * width, 16 * width)
        self.fusion = _branch(96 * width, 64 * width, 32 * width)
        self.heads = nn.Linear(32 * width, 2)

    def forward(self, properties, fingerprint, context):
        fused = torch.cat(
            (self.properties(properties), self.fingerprint(fingerprint), self.context(context)),
            dim=1,
        )
        return self.heads(self.fusion(fused))
