import torch
from functools import reduce
from operator import mul
import math
import json


def quantize_symmetric(x, bits=8):
    qmax = 2 ** (bits - 1) - 1
    max_val = x.abs().max()
    scale = max_val / qmax
    scale = scale + 1e-8
    x_q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int32)
    return x_q, scale


def dequantize_symmetric(x_q, scale):
    return x_q.float() * scale


def freeze_model(net, config):
    # lpips loss model freezing
    for name, param in net.image_loss.lpips.named_parameters():
        param.requires_grad = False


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


def load_ldpc_configs(path="./logs/thresholds.json"):
    with open(path) as f:
        configs = json.load(f)
    # sort ascending by threshold so we can select greedily
    return sorted(configs, key=lambda x: x["threshold_snr_db"])


def select_ldpc_config(snr_db, configs=None):
    if configs is None:
        configs = load_ldpc_configs()
    # sorted ascending by threshold, pick the last one below snr_db
    reliable = [c for c in configs if snr_db >= c["threshold_snr_db"]]
    if not reliable:
        return None
    return reliable[-1]  # highest threshold still below snr_db


def mask_cbr_overhead(
    cbr,
    snr_db,
    image_dims=(3, 256, 256),
    patch_size=2,
    downsample_layers=4,
):
    """
    Returns the additional CBR cost of transmitting the mask via LDPC.
    Returns float('inf') if no reliable LDPC config exists at this SNR.
    """
    image_numel = reduce(mul, image_dims)
    feature_channel = cbr_to_channel(cbr, image_dims, patch_size, downsample_layers)
    config = select_ldpc_config(snr_db)
    if config is None:
        return float("inf")  # system non-functional at this SNR
    mask_bits = feature_channel
    codewords_needed = math.ceil(mask_bits / config["k"])
    mask_coded_bits = codewords_needed * config["n"]
    mask_symbols = math.ceil(mask_coded_bits / config["m"])
    return mask_symbols / image_numel
