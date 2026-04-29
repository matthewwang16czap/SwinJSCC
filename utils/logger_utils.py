import os
import logging
from .ddp_utils import is_main_process


def logger_configuration(config, save_log=False, test_mode=False):
    logger = logging.getLogger("Deep joint source channel coder")
    # Avoid re-adding handlers in case of multiple imports / repeated calls
    if logger.hasHandlers():
        logger.handlers.clear()
    if test_mode:
        config.workdir += "_test"
    if save_log and is_main_process():
        os.makedirs(config.workdir, exist_ok=True)
        os.makedirs(config.models_dir, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s] %(message)s")
    # --- Only rank 0 prints to stdout ---
    if is_main_process():
        stdhandler = logging.StreamHandler()
        stdhandler.setLevel(logging.INFO)
        stdhandler.setFormatter(formatter)
        logger.addHandler(stdhandler)
    # --- All ranks can log to file if you want shared logs ---
    if save_log:
        filehandler = logging.FileHandler(config.log_dir)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    config.logger = logger
    return config.logger


def get_logger_dir(logger):
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return os.path.dirname(handler.baseFilename)
    return "."


basic_metrics_keys = {"snr", "cbr", "psnr"}
metric_order = ["loss", "snr", "cbr", "psnr", "mse", "lpips", "ssim", "msssim"]


# formatting for testing logging
def format_value(k, v):
    val = v[0] if isinstance(v, list) else v
    if k == "time":
        return f"{val:.3f}s"
    elif k in basic_metrics_keys:
        return f"{val:>10.3f}"
    else:
        return f"{val:>10.2e}"


# formatting for training logging
def format_metric(k, v):
    vals = v if isinstance(v, list) else [v]
    # Format each value based on whether it is a basic metric
    if k in basic_metrics_keys:
        val_strs = [f"{val:>8.3f}" for val in vals]
    else:
        val_strs = [f"{val:>10.2e}" for val in vals]
    # Returns "KEY: val1 val2 ..."
    return f"{k.upper()}: " + " ".join(val_strs)
