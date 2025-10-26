import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two Conv1d + GroupNorm + SiLU blocks."""

    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """Downsampling block: halve temporal length, double channels."""

    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsampling block: double temporal length, halve channels."""

    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet1D(nn.Module):
    """
    1D UNet with dynamic depth, doubling channels per downsample.
    Example channel flow for depth=4, hidden=64:
      64 → 128 → 256 → 512 → bottleneck(512) → 256 → 128 → 64
    """

    def __init__(
        self, channels=320, hidden=320, num_groups=8, depth=3, use_sigmoid=True
    ):
        super().__init__()
        assert depth >= 1, "depth must be >= 1"
        self.use_sigmoid = use_sigmoid
        self.depth = depth

        # Encoder
        self.first_enc = ConvBlock(channels, hidden, num_groups)

        enc_layers = []
        in_ch = hidden
        for _ in range(depth):
            out_ch = in_ch * 2  # double channels each time
            enc_layers.append(DownBlock(in_ch, out_ch, num_groups))
            in_ch = out_ch
        self.encoders = nn.ModuleList(enc_layers)

        # Bottleneck
        self.bottleneck = ConvBlock(in_ch, in_ch, num_groups)
        bottleneck_ch = in_ch

        # Decoder (mirror of encoder)
        dec_layers = []
        for _ in range(depth):
            out_ch = bottleneck_ch // 2
            dec_layers.append(UpBlock(bottleneck_ch, out_ch, num_groups))
            bottleneck_ch = out_ch
        self.decoders = nn.ModuleList(dec_layers)

        # Output
        self.outc = nn.Conv1d(hidden, channels, 3, padding=1)
        if use_sigmoid:
            self.activation = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, C, N]
        skips = []

        # Initial conv
        x = self.first_enc(x)

        # Encoder
        for enc in self.encoders:
            skips.append(x)
            x = enc(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder (mirror)
        for dec in self.decoders:
            skip = skips.pop(-1)
            x = dec(x)
            # align temporal length before skip connection
            # if x.size(-1) != skip.size(-1):
            #     x = F.interpolate(
            #         x, size=skip.size(-1), mode="linear", align_corners=False
            #     )
            x = x + skip  # skip connection

        # Output layer
        x = self.outc(x)
        if self.use_sigmoid:
            x = self.activation(x)
        return x.permute(0, 2, 1)  # [B, N, C]
