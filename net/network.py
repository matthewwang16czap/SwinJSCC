from net.decoder import *
from net.encoder import *
from loss.distortion import Distortion
from net.channel import Channel
from random import choice
import torch
import torch.nn as nn
from net.denoiser import UNet1D
from loss.denoise import denoise_loss_fn


class SwinJSCC(nn.Module):
    def __init__(self, args, config):
        super(SwinJSCC, self).__init__()
        self.config = config
        encoder_kwargs = config.encoder_kwargs
        decoder_kwargs = config.decoder_kwargs
        self.encoder = create_encoder(**encoder_kwargs)
        self.decoder = create_decoder(**decoder_kwargs)
        if config.logger is not None:
            config.logger.info("Network config: ")
            config.logger.info("Encoder: ")
            config.logger.info(encoder_kwargs)
            config.logger.info("Decoder: ")
            config.logger.info(decoder_kwargs)
        self.distortion_loss = Distortion(args)
        self.denoise_loss = denoise_loss_fn
        self.channel = Channel(args, config)
        self.pass_channel = config.pass_channel
        self.squared_difference = torch.nn.MSELoss(reduction="none")
        self.H = self.W = 0
        self.multiple_snr = args.multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])
        self.channel_number = args.C.split(",")
        for i in range(len(self.channel_number)):
            self.channel_number[i] = int(self.channel_number[i])
        self.downsample = config.downsample
        self.model = args.model
        # feature_channels = encoder_kwargs["embed_dims"][-1]
        self.feature_denoiser = (
            UNet1D(
                channels=encoder_kwargs["embed_dims"][-1],
                hidden=encoder_kwargs["embed_dims"][-1],
                factor=1,
            )
            if args.denoise
            else None
        )

    def distortion_loss_wrapper(self, x_gen, x_real):
        distortion_loss = self.distortion_loss.forward(
            x_gen, x_real, normalization=self.config.norm
        )
        return distortion_loss

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def forward(self, input_image, given_SNR=None, given_rate=None):
        B, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(
                H // (2**self.downsample), W // (2**self.downsample)
            )
            self.H = H
            self.W = W

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        if given_rate is None:
            channel_number = choice(self.channel_number)
        else:
            channel_number = given_rate

        if self.model == "SwinJSCC_w/o_SAandRA" or self.model == "SwinJSCC_w/_SA":
            feature = self.encoder(input_image, chan_param, channel_number, self.model)
            CBR = feature.numel() / 2 / input_image.numel()
            if self.pass_channel:
                noisy_feature = self.feature_pass_channel(feature, chan_param)
            else:
                noisy_feature = feature

        elif self.model == "SwinJSCC_w/_RA" or self.model == "SwinJSCC_w/_SAandRA":
            feature, mask = self.encoder(
                input_image, chan_param, channel_number, self.model
            )
            CBR = channel_number / (2 * 3 * 2 ** (self.downsample * 2))
            avg_pwr = torch.sum(feature**2) / mask.sum()
            if self.pass_channel:
                noisy_feature = self.feature_pass_channel(feature, chan_param, avg_pwr)
            else:
                noisy_feature = feature
            noisy_feature = noisy_feature * mask

        # --- Pass noisy feature through feature_denoiser network ---
        restored_feature = (
            self.feature_denoiser(noisy_feature)
            if self.feature_denoiser
            else noisy_feature
        )

        recon_image = self.decoder(restored_feature, chan_param, self.model)
        mse = self.squared_difference(
            input_image * 255.0, recon_image.clamp(0.0, 1.0) * 255.0
        )
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0.0, 1.0))

        # # for training denoiser only
        # denoise_loss, _, _ = self.denoise_loss(
        #     restored_feature, feature, mask, alpha=0, beta=1.0
        # )

        # if self.config.denoise_training:
        #     return recon_image, CBR, chan_param, mse.mean(), denoise_loss
        # else:
        #     return recon_image, CBR, chan_param, mse.mean(), loss_G.mean()

        return (
            recon_image,
            restored_feature,
            noisy_feature,
            feature,
            mask,
            CBR,
            chan_param,
            mse.mean(),
            loss_G.mean(),
        )
