import torch
import torch.nn as nn
from sionna.phy.channel import AWGN, FlatFadingChannel


class Channel(nn.Module):
    """
    Physically complex baseband channel backed by Sionna 2.0 blocks.
    Input shape: (B, N, C), where C = 2 * num_complex_dims
    """

    def __init__(self, config):
        super().__init__()
        self.chan_type = config.channel_type
        self.device = config.device
        # Instantiate Sionna 2.0 channel blocks
        if self.chan_type in (1, "awgn"):
            self._sionna_awgn = AWGN(device=str(self.device))
        elif self.chan_type in (2, "rayleigh"):
            # FlatFadingChannel: num_tx_ant=1, num_rx_ant=1 for flat per-symbol fading.
            # We'll apply it symbol-by-symbol by reshaping (B*N, 1) → (B*N, 1).
            self._sionna_rayleigh = FlatFadingChannel(
                num_tx_ant=1,
                num_rx_ant=1,
                return_channel=False,
                device=str(self.device),
            )
        if config.logger:
            config.logger.info(
                f"【Channel】: Built {self.chan_type} channel (Sionna 2.0), SNR {config.snrs} dB."
            )

    @staticmethod
    def _to_complex(x: torch.Tensor) -> torch.Tensor:
        """(B, N, C) packed real/imag  →  (B, N, C//2) complex"""
        B, N, C = x.shape
        assert C % 2 == 0, "Channel input C must be even (real + imag)"
        C2 = C // 2
        return torch.complex(x[..., :C2], x[..., C2:])

    @staticmethod
    def _to_packed(xc: torch.Tensor) -> torch.Tensor:
        """(B, N, C//2) complex  →  (B, N, C) packed real/imag"""
        return torch.cat([xc.real, xc.imag], dim=-1)

    @staticmethod
    def _normalize_power(
        xc: torch.Tensor, target_power: float = 1.0, eps: float = 1e-12
    ):
        power = torch.mean(torch.abs(xc) ** 2)
        scale = torch.sqrt(torch.tensor(target_power, device=xc.device) / (power + eps))
        return xc * scale, power

    @staticmethod
    def _snr_db_to_no(snr_db) -> torch.Tensor:
        """Convert SNR in dB to Sionna noise variance `no` (per complex dim).

        Sionna AWGN adds noise with variance `no` per complex dimension,
        i.e. total noise power = no.  For unit-power signals:
            no = 1 / snr_linear
        """
        if not isinstance(snr_db, torch.Tensor):
            snr_db = torch.tensor(float(snr_db))
        snr_linear = torch.pow(torch.tensor(10.0, device=snr_db.device), snr_db / 10.0)
        return 1.0 / snr_linear

    def forward(self, x: torch.Tensor, snr_db) -> torch.Tensor:
        """
        x      : (B, N, C) packed real/imag
        snr_db : scalar tensor or float
        """
        B, N, C = x.shape
        C2 = C // 2
        # 1. Convert to complex
        xc = self._to_complex(x)  # (B, N, C2)
        # 2. Power normalisation
        xc, _ = self._normalize_power(xc, target_power=1.0)
        # 3. Compute Sionna noise variance from SNR
        no = self._snr_db_to_no(snr_db).to(xc.device)
        # 4. Apply channel
        if self.chan_type in (0, "none"):
            yc = xc
        elif self.chan_type in (1, "awgn"):
            # AWGN: operates on any complex tensor shape
            yc = self._sionna_awgn(xc, no)  # (B, N, C2)
        elif self.chan_type in (2, "rayleigh"):
            # FlatFadingChannel expects (batch, num_tx_ant).
            # Reshape (B*N, C2) so each complex "symbol" is a 1-antenna vector,
            # apply the flat-fading block, then reshape back.
            flat = xc.reshape(B * N * C2, 1)  # treat each symbol independently
            flat_out = self._sionna_rayleigh(
                flat, no
            )  # (B*N, C2)  [SISO: num_rx_ant=C2? No…]
            yc = flat_out.reshape(B, N, C2)
        else:
            raise ValueError(f"Unsupported channel type: {self.chan_type}")
        # 5. Pack back to real
        y = self._to_packed(yc)  # (B, N, C)
        return y
