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
        # Pad to even length to avoid size mismatches
        if x.size(-1) % 2 != 0:
            x = F.pad(x, (0, 1))
        return self.block(x)


class UpBlock(nn.Module):
    """Upsampling block: double temporal length, reduce channels."""

    def __init__(self, in_ch, out_ch, num_groups=8, use_concat=True):
        super().__init__()
        self.use_concat = use_concat

        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(inplace=True),
        )

        # If using concatenation, conv input will be out_ch + out_ch (from skip)
        # If using addition, conv input will be just out_ch
        conv_in_ch = out_ch * 2 if use_concat else out_ch

        self.conv = nn.Sequential(
            nn.Conv1d(conv_in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)

        if skip is not None:
            # Align temporal dimensions
            if x.size(-1) != skip.size(-1):
                x = F.interpolate(
                    x, size=skip.size(-1), mode="linear", align_corners=False
                )

            if self.use_concat:
                x = torch.cat([x, skip], dim=1)
            else:
                x = x + skip

        x = self.conv(x)
        return x


class UNet1D(nn.Module):
    """
    1D UNet for channel-wise noise prediction.

    Improvements:
    - Concatenation skip connections (better info preservation)
    - Zero-initialized output layer (starts predicting no noise)
    - Stronger bottleneck (double conv blocks)
    - Proper padding in downsampling
    - Validation of group normalization compatibility

    Args:
        channels: Number of input/output channels
        hidden: Initial hidden dimension
        num_groups: Number of groups for GroupNorm (must divide hidden evenly)
        depth: Number of downsampling layers
        factor: Channel multiplication factor per layer
        use_concat_skip: Use concatenation instead of addition for skip connections

    Example:
        model = UNet1D(channels=320, hidden=64, depth=3)
        noisy = torch.randn(4, 1000, 320)  # [B, T, C]
        noise_pred = model(noisy)  # [B, T, C]
    """

    def __init__(
        self,
        channels=320,
        hidden=64,
        num_groups=8,
        depth=3,
        factor=2,
        use_concat_skip=True,
    ):
        super().__init__()
        assert depth >= 1, "depth must be >= 1"
        assert (
            hidden % num_groups == 0
        ), f"hidden={hidden} must be divisible by num_groups={num_groups}"

        self.depth = depth
        self.use_concat_skip = use_concat_skip

        # Encoder
        self.first_enc = ConvBlock(channels, hidden, num_groups)

        enc_layers = []
        in_ch = hidden
        for _ in range(depth):
            out_ch = in_ch * factor
            assert (
                out_ch % num_groups == 0
            ), f"Channel count {out_ch} must be divisible by num_groups={num_groups}"
            enc_layers.append(DownBlock(in_ch, out_ch, num_groups))
            in_ch = out_ch
        self.encoders = nn.ModuleList(enc_layers)

        # Bottleneck - stronger for noise prediction
        self.bottleneck = nn.Sequential(
            ConvBlock(in_ch, in_ch, num_groups),
            ConvBlock(in_ch, in_ch, num_groups),
        )
        bottleneck_ch = in_ch

        # Decoder
        dec_layers = []
        for _ in range(depth):
            out_ch = bottleneck_ch // factor
            assert (
                out_ch % num_groups == 0
            ), f"Channel count {out_ch} must be divisible by num_groups={num_groups}"
            dec_layers.append(
                UpBlock(bottleneck_ch, out_ch, num_groups, use_concat_skip)
            )
            bottleneck_ch = out_ch
        self.decoders = nn.ModuleList(dec_layers)

        # Output layer - zero initialized for stable training
        self.outc = nn.Conv1d(hidden, channels, 3, padding=1)
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, time, channels]
        Returns:
            Predicted noise [batch, time, channels]
        """
        x = x.permute(0, 2, 1)  # [B, C, T]
        skips = []

        # Initial conv
        x = self.first_enc(x)

        # Encoder path
        for enc in self.encoders:
            skips.append(x)
            x = enc(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path with skip connections
        for dec in self.decoders:
            skip = skips.pop(-1)
            x = dec(x, skip)

        # Output
        x = self.outc(x)
        return x.permute(0, 2, 1)  # [B, T, C]


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = UNet1D(
        channels=320,
        hidden=64,
        num_groups=8,
        depth=3,
        factor=2,
        use_concat_skip=True,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 4
    seq_len = 1000
    num_channels = 320

    x = torch.randn(batch_size, seq_len, num_channels)
    y = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    assert x.shape == y.shape, "Input and output shapes must match!"
    print("✓ Shape test passed")
