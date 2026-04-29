from .decoder import create_decoder
from .encoder import create_encoder
from .modules import ModulatorLayer
from loss.image_losses import ImageLoss
from utils.model_utils import cbr_to_channel
from .channel import Channel
import torch
import torch.nn as nn


class SwinJSCC(nn.Module):
    def __init__(self, config):
        super(SwinJSCC, self).__init__()
        self.config = config
        encoder_kwargs = config.encoder_kwargs
        decoder_kwargs = config.decoder_kwargs
        self.encoder = create_encoder(**encoder_kwargs)
        self.decoder = create_decoder(**decoder_kwargs)
        self.encoder_snr_modulator = ModulatorLayer(
            self.encoder.embed_dims[len(self.encoder.embed_dims) - 1],
            int(self.encoder.embed_dims[len(self.encoder.embed_dims) - 1] * 1.5),
        )
        self.encoder_cbr_modulator = ModulatorLayer(
            self.encoder.embed_dims[len(self.encoder.embed_dims) - 1],
            int(self.encoder.embed_dims[len(self.encoder.embed_dims) - 1] * 1.5),
        )
        self.decoder_snr_modulator = ModulatorLayer(
            self.decoder.embed_dims[0],
            int(self.decoder.embed_dims[0] * 1.5),
        )
        if config.logger is not None:
            config.logger.info("Network config: ")
            config.logger.info("Encoder: ")
            config.logger.info(encoder_kwargs)
            config.logger.info("Decoder: ")
            config.logger.info(decoder_kwargs)
        self.channel = Channel(config)
        self.image_loss = ImageLoss(normalized=True, data_range=1.0)

    def encoder_side_processing(self, input_image, snr=None, cbr=None):
        # Encoding
        feature, feature_H, feature_W = self.encoder(input_image)
        feature = self.encoder_snr_modulator(feature, snr)
        feature = self.encoder_cbr_modulator(feature, cbr)
        # TopK channel pruning based on channel activation magnitude
        k = cbr_to_channel(cbr)
        channel_scores = feature.abs().mean(dim=1)  # (B, C) — average over spatial N
        topk_indices = channel_scores.topk(k, dim=-1).indices  # (B, k)
        mask = torch.zeros_like(feature)
        mask.scatter_(2, topk_indices.unsqueeze(1).expand(-1, feature.size(1), -1), 1.0)
        feature = feature * mask
        # Channel simulation
        noisy_feature = (
            self.channel.forward(feature.float(), snr)
            if self.config.pass_channel
            else feature
        )
        return noisy_feature, feature_H, feature_W

    def decoder_side_processing(
        self,
        input_image,
        noisy_feature,
        feature_H,
        feature_W,
        valid,
        snr=None,
    ):
        noisy_feature = self.decoder_snr_modulator(noisy_feature, snr)
        # Decoding
        recon_images = self.decoder(
            noisy_feature,
            feature_H,
            feature_W,
            valid,
        )
        loss, metrics = self.image_loss((recon_images,), input_image, valid)
        return recon_images, loss, metrics

    def forward(self, input_image, valid, snr=None, cbr=None):
        snr_tensor = (
            torch.tensor([[snr]], device=input_image.device)
            if snr is not None
            else None
        )
        cbr_tensor = (
            torch.tensor([[float(cbr)]], device=input_image.device)
            if cbr is not None
            else None
        )
        noisy_feature, feature_H, feature_W = self.encoder_side_processing(
            input_image, snr=snr_tensor, cbr=cbr_tensor
        )
        recon_images, loss, metrics = self.decoder_side_processing(
            input_image,
            noisy_feature,
            feature_H,
            feature_W,
            valid,
            snr=snr_tensor,
        )
        return (
            recon_images,
            metrics,
            loss,
        )
