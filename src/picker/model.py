"""Regression CNN for P-wave arrival picking."""

import torch.nn as nn


class PickerCNN(nn.Module):
    """Simple 1D CNN that predicts the P-wave arrival as a single sample index.

    Input:  (batch, 1, 6000)
    Output: (batch,) — predicted arrival sample (raw scalar, no activation)

    Sequence lengths through the network:
        (1,  6000)
        (16, 1500)  after block 1  MaxPool1d(4)
        (32,  375)  after block 2  MaxPool1d(4)
        (64,   93)  after block 3  MaxPool1d(4)
        (64,    1)  after AdaptiveAvgPool1d(1)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,  16, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)