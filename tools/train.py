from utils.logger_utils import format_metric, metric_order
from utils.data_utils import get_batch_data
import torch
import time
from random import choice


def train_one_step(
    epoch,
    global_step,
    net,
    train_loader,
    optimizer,
    logger,
    config,
    scaler,
):
    is_ddp = hasattr(net, "module")
    net.train()
    optimizer.zero_grad(set_to_none=True)
    # train batch data
    for batch_idx, data in enumerate(train_loader):
        start_time = time.time()
        global_step += 1
        input, valid = get_batch_data(data, config)
        # SNR and CBR selection
        snr = choice(config.snrs)
        cbr = choice(config.cbrs)
        # Forward and backward pass
        with torch.autocast(
            device_type=input.device.type, enabled=(scaler is not None)
        ):
            (
                _,
                metrics,
                loss,
            ) = net(input, valid, snr, cbr)
            loss = loss / config.accum_steps
        if scaler is not None:
            if is_ddp and (batch_idx % config.accum_steps != config.accum_steps - 1):
                # During accumulation, avoid syncing DDP gradients
                with net.no_sync():
                    scaler.scale(loss).backward()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            if is_ddp and (batch_idx % config.accum_steps != config.accum_steps - 1):
                with net.no_sync():
                    loss.backward()
            else:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        # Update metrics
        metrics = {
            "time": time.time() - start_time,
            "snr": [snr],
            "cbr": [float(cbr)],
            "loss": loss.item(),
            **metrics,
        }
        # Logging
        if global_step % config.print_step == 0:
            process = (
                (global_step % train_loader.__len__())
                / (train_loader.__len__())
                * 100.0
            )
            step_info = [
                f"Epoch {epoch + 1}",
                f"Step [{batch_idx + 1}/{len(train_loader)}={process:.2f}%]",
            ]
            display_metrics = ["loss", "snr", "cbr", "psnr"]
            metrics_info = [
                format_metric(k, metrics[k]) for k in display_metrics if k in metrics
            ]
            logger.info(" | ".join(step_info))
            logger.info(" | ".join(metrics_info))
    return global_step
