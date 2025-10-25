import torch
import torch.nn as nn


class UNet1D(nn.Module):
    def __init__(self, channels=320, hidden=320, num_groups=8):
        super().__init__()

        # --- Encoder ---
        self.enc1 = nn.Sequential(
            nn.Conv1d(channels, hidden, 3, padding=1),
            nn.GroupNorm(num_groups, hidden),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(num_groups, hidden),
            nn.SiLU(inplace=True),
        )

        self.enc2 = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, stride=2, padding=1),  # Downsample
            nn.GroupNorm(num_groups, hidden),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(num_groups, hidden),
            nn.SiLU(inplace=True),
        )

        self.enc3 = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, stride=2, padding=1),  # Downsample again
            nn.GroupNorm(num_groups, hidden),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(num_groups, hidden),
            nn.SiLU(inplace=True),
        )

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(num_groups, hidden),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(num_groups, hidden),
            nn.SiLU(inplace=True),
        )

        # --- Decoder ---
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(hidden, hidden, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups, hidden),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(num_groups, hidden),
            nn.SiLU(inplace=True),
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(hidden, hidden, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups, hidden),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(num_groups, hidden),
            nn.SiLU(inplace=True),
        )

        # --- Output ---
        self.outc = nn.Conv1d(hidden, channels, 3, padding=1)

    def forward(self, x):
        """
        x: [B, N, C] — input sequence with masked (zeroed) channels.
        returns: [B, N, C]
        """
        x = x.permute(0, 2, 1)  # [B, C, N]

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Bottleneck
        b = self.bottleneck(e3)

        # Decoder with skip connections
        d2 = self.dec2(b) + e2
        d1 = self.dec1(d2) + e1

        # Output
        out = self.outc(d1)
        return out.permute(0, 2, 1)  # [B, N, C]
