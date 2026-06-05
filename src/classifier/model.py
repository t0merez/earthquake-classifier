"""CNN classifier for earthquake vs. noise detection."""

import torch.nn as nn


class SpectrogramCNN(nn.Module):
    """2D CNN operating on n×n log-power spectrograms.

    Input:  (batch, 1, n, n)
    Output: (batch,)  — raw logit; apply sigmoid for probability
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: (1, 10, 10) → (32, 10, 10)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Block 2: (32, 10, 10) → (64, 5, 5)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: (64, 5, 5) → (128, 2, 2)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)