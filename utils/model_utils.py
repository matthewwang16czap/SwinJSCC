import torch
from functools import reduce
from operator import mul


def quantize_symmetric(x, bits=8):
    qmax = 2 ** (bits - 1) - 1
    max_val = x.abs().max()
    scale = max_val / qmax
    scale = scale + 1e-8
    x_q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int32)
    return x_q, scale


def dequantize_symmetric(x_q, scale):
    return x_q.float() * scale


def cbr_to_channel(cbr, image_dims=(3, 256, 256), patch_size=2, downsample_layers=4):
    image_numel = reduce(mul, image_dims)
    # feature size is equal to (H*W) / (patch_size*(2^downsample_layers))^2
    feature_size = (
        image_dims[1]
        * image_dims[2]
        // (patch_size * 2 ** (downsample_layers - 1)) ** 2
    )
    complex_ratio = 2  # one complex number can represent two real numbers
    feature_channel = int(cbr * complex_ratio * image_numel / feature_size)
    return feature_channel
