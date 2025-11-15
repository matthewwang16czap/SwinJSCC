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
    1D UNet that outputs both clean and noise features.
    Input:  (B, H, C)
    Output: clean (B, H, C), noise (B, H, C)
    """

    def __init__(
        self,
        channels=320,
        hidden=64,
        num_groups=8,
        depth=3,
        factor=2,
        use_sigmoid=False,
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
            out_ch = in_ch * factor
            enc_layers.append(DownBlock(in_ch, out_ch, num_groups))
            in_ch = out_ch
        self.encoders = nn.ModuleList(enc_layers)

        # Bottleneck
        self.bottleneck = ConvBlock(in_ch, in_ch, num_groups)
        bottleneck_ch = in_ch

        # Decoder (mirror)
        dec_layers = []
        for _ in range(depth):
            out_ch = bottleneck_ch // factor
            dec_layers.append(UpBlock(bottleneck_ch, out_ch, num_groups))
            bottleneck_ch = out_ch
        self.decoders = nn.ModuleList(dec_layers)

        # Project back to the same number of input channels (before output)
        self.final_conv = nn.Conv1d(bottleneck_ch, channels, 3, padding=1)

        # Output layer now predicts *2 × channels* for clean + noise
        self.outc = nn.Conv1d(channels, channels * 2, 3, padding=1)
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

        if use_sigmoid:
            self.activation = nn.Sigmoid()

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask
        x = x.permute(0, 2, 1)  # [B, C, N]
        skips = []

        # Encoder
        x = self.first_enc(x)
        for enc in self.encoders:
            skips.append(x)
            x = enc(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for dec in self.decoders:
            skip = skips.pop(-1)
            x = dec(x)
            if x.size(-1) != skip.size(-1):
                x = F.interpolate(
                    x, size=skip.size(-1), mode="linear", align_corners=False
                )
            x = x + skip

        # Final projection back to input channels
        x = self.final_conv(x)

        # Output: (B, 2C, N)
        x = self.outc(x)  # (B, 2C, N)
        clean, noise = x.chunk(2, dim=1)  # (B, C, N) each

        if self.use_sigmoid:
            clean = torch.sigmoid(clean)  # only clean is squashed to (0,1)
        # leave `noise` as-is (raw residual)

        clean = clean.permute(0, 2, 1)  # (B, N, C)
        noise = noise.permute(0, 2, 1)

        if mask is not None:
            clean = clean * mask
            noise = noise * mask

        return clean, noise


if __name__ == "__main__":
    # Instantiate model
    model = UNet1D(channels=320, hidden=64, depth=3, factor=2)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test input dimensions
    batch_size = 16
    seq_len = 256
    num_channels = 320

    noisy = torch.randn(batch_size, seq_len, num_channels)
    mask = torch.ones(batch_size, seq_len, num_channels)

    # Mask out some feature channels
    mask[:, :, :50] = 0
    noisy = noisy * mask

    # Forward pass
    with torch.no_grad():
        clean, noise = model(noisy)

    print(f"Input:  {noisy.shape}")
    print(f"Clean:  {clean.shape}")
    print(f"Noise:  {noise.shape}")

    # Mask verification — if masking applied outside model
    if torch.all(mask == 1):
        print("No mask applied.")
    else:
        # Here we expect masked regions to remain zero since we zeroed input
        assert torch.allclose(noisy[mask == 0], torch.zeros_like(noisy[mask == 0]))
        print("✓ Mask correctly applied to input!")

    # GPU memory profiling
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        noisy = noisy.to(device)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            clean, noise = model(noisy)

        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"✓ Peak GPU memory: {peak_mem:.2f} GB")

    print("✓ Model forward test completed successfully!")
