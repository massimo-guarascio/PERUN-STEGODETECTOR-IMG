import torch
import torch.nn as nn


class StegoDetectorExtendedTorch(nn.Module):
    """
    PyTorch version of the original Keras StegoDetectorExtended.

    Architecture:
        Linear -> ReLU -> BatchNorm -> Dropout
        Linear -> ReLU -> BatchNorm -> Dropout
        Linear -> ReLU -> BatchNorm -> Dropout
        Linear -> output logit

    Important:
        The final layer does NOT contain sigmoid.
        During training we will use BCEWithLogitsLoss,
        which applies sigmoid internally in a stable way.
    """

    def __init__(self, input_dim, dl_size=128, drop_pcg=0.025):
        super().__init__()

        self.block_0 = nn.Sequential(
            nn.Linear(input_dim, dl_size),
            nn.ReLU(),
            nn.BatchNorm1d(dl_size),
            nn.Dropout(drop_pcg),
        )

        self.block_1 = nn.Sequential(
            nn.Linear(dl_size, dl_size),
            nn.ReLU(),
            nn.BatchNorm1d(dl_size),
            nn.Dropout(drop_pcg),
        )

        self.block_2 = nn.Sequential(
            nn.Linear(dl_size, dl_size),
            nn.ReLU(),
            nn.BatchNorm1d(dl_size),
            nn.Dropout(drop_pcg),
        )

        self.output_layer = nn.Linear(dl_size, 1)

    def forward(self, x):
        x = self.block_0(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.output_layer(x)
        return x