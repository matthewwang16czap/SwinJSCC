import torchvision
import torch
import torch.distributed as dist
import json
from collections import defaultdict
from utils.data_utils import get_batch_data
from utils.logger_utils import get_logger_dir, format_value, metric_order
from utils.universal_utils import get_path


def test(net, test_loader, logger, config):
    net.eval()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    results = []
    for snr in config.snrs:
        for cbr in config.cbrs:
            metrics = defaultdict(float)
            counts = 0
            with torch.no_grad():
                for batch_idx, data in enumerate(test_loader):
                    input, valid = get_batch_data(data, config)
                    if rank == 0 and snr == config.snrs[0] and cbr == config.cbrs[0]:
                        save_path = get_path(".", "recons", f"origin_{batch_idx}.png")
                        torchvision.utils.save_image(input[0], save_path)
                    (
                        recon_images,
                        model_metrics,
                        loss,
                    ) = net(input, valid, snr, cbr)
                    # for visualization
                    if rank == 0:
                        save_path = get_path(
                            ".",
                            "recons",
                            f"recon{batch_idx}_{snr:.3f}_{float(cbr):.3f}.png",
                        )
                        torchvision.utils.save_image(recon_images[0], save_path)
                    # Update batch data to metrics
                    batch_size = input.size(0)
                    for k, v in model_metrics.items():
                        # metrics values are list of length 1
                        metrics[k] += v[0] * batch_size
                    counts += batch_size
            # DDP reduce
            for key in metrics:
                tensor = torch.tensor(metrics[key], device=config.device)
                if dist.is_initialized():
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                metrics[key] = (tensor / (counts * world_size)).item()
            # Store result
            results.append({"snr": snr, "cbr": float(cbr), **metrics})
    # logging
    if rank == 0 and logger is not None:
        logger.info("Start Test:")
        keys = [k for k in metric_order if k in results[0]]
        logger.info("".join(f"{k.upper():>10}" for k in keys))
        for r in results:
            logger.info("".join(format_value(k, r[k]) for k in keys))
        logger.info("Finish Test!")
        with open(get_path(get_logger_dir(logger), "test_results.json"), "w") as f:
            json.dump(results, f, indent=4)
    return results
