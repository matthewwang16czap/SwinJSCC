from utils import *
import torch
from loss.distortion import *
from loss.denoise import *
import time
from random import choice


def train_one_epoch(
    epoch, global_step, net, train_loader, optimizer, CalcuSSIM, logger, args, config
):
    net.train()
    # Initialize metrics
    metrics_names = ["elapsed", "losses", "psnrs", "msssims", "cbrs", "snrs"]
    metrics = {name: AverageMeter() for name in metrics_names}

    for batch_idx, data in enumerate(train_loader):
        start_time = time.time()
        global_step += 1

        # Handle different data formats
        input = data[0] if args.trainset == "CIFAR10" else data
        input = input.cuda()

        # Forward pass
        (
            recon_image,
            restored_feature,
            noisy_feature,
            feature,
            mask,
            CBR,
            SNR,
            mse,
            loss_G,
        ) = net(input)

        # Backward pass
        optimizer.zero_grad()
        loss_G.backward()
        optimizer.step()

        # Update metrics
        metrics["elapsed"].update(time.time() - start_time)
        metrics["losses"].update(loss_G.item())
        metrics["cbrs"].update(CBR)
        metrics["snrs"].update(SNR)

        if mse.item() > 0:
            psnr = 10 * (torch.log(255.0 * 255.0 / mse) / np.log(10))
            metrics["psnrs"].update(psnr.item())
            msssim = 1 - CalcuSSIM(input, recon_image.clamp(0.0, 1.0)).mean().item()
            metrics["msssims"].update(msssim)

        # Logging
        if global_step % config.print_step == 0:
            process = (
                (global_step % train_loader.__len__())
                / (train_loader.__len__())
                * 100.0
            )

            log_components = [
                f"Epoch {epoch}",
                f"Step [{batch_idx + 1}/{len(train_loader)}={process:.2f}%]",
                f"Time {metrics['elapsed'].val:.3f}",
                f"Loss {metrics['losses'].val:.3f} ({metrics['losses'].avg:.3f})",
                f"CBR {metrics['cbrs'].val:.4f} ({metrics['cbrs'].avg:.4f})",
                f"SNR {metrics['snrs'].val:.1f} ({metrics['snrs'].avg:.1f})",
                f"PSNR {metrics['psnrs'].val:.3f} ({metrics['psnrs'].avg:.3f})",
                f"MSSSIM {metrics['msssims'].val:.3f} ({metrics['msssims'].avg:.3f})",
                f"Lr {config.learning_rate}",
            ]

            logger.info(" | ".join(log_components))

            # Reset metrics after logging
            for metric in metrics.values():
                metric.clear()

    # Final reset of metrics
    for metric in metrics.values():
        metric.clear()

    return global_step


def test(net, test_loader, CalcuSSIM, logger, args, config):
    config.isTrain = False
    net.eval()
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]
    multiple_snr = args.multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])
    channel_number = args.C.split(",")
    for i in range(len(channel_number)):
        channel_number[i] = int(channel_number[i])
    results_snr = np.zeros((len(multiple_snr), len(channel_number)))
    results_cbr = np.zeros((len(multiple_snr), len(channel_number)))
    results_psnr = np.zeros((len(multiple_snr), len(channel_number)))
    results_msssim = np.zeros((len(multiple_snr), len(channel_number)))
    for i, SNR in enumerate(multiple_snr):
        for j, rate in enumerate(channel_number):
            with torch.no_grad():
                for batch_idx, data in enumerate(test_loader):
                    start_time = time.time()
                    # Handle different data formats
                    input = data[0]
                    input = input.cuda()
                    (
                        recon_image,
                        restored_feature,
                        noisy_feature,
                        feature,
                        mask,
                        CBR,
                        SNR,
                        mse,
                        loss_G,
                    ) = net(input, SNR, rate)
                    # torchvision.utils.save_image(recon_image, os.path.join("/home/matthewwang16czap/projects/SwinJSCC/data/", f"recon/{names[0]}"))

                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255.0 * 255.0 / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = (
                            1
                            - CalcuSSIM(input, recon_image.clamp(0.0, 1.0))
                            .mean()
                            .item()
                        )
                        msssims.update(msssim)

                    log = " | ".join(
                        [
                            f"Time {elapsed.val:.3f}",
                            f"CBR {cbrs.val:.4f} ({cbrs.avg:.4f})",
                            f"SNR {snrs.val:.1f}",
                            f"PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})",
                            f"MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})",
                            f"Lr {config.learning_rate}",
                        ]
                    )
                    logger.info(log)
            results_snr[i, j] = snrs.avg
            results_cbr[i, j] = cbrs.avg
            results_psnr[i, j] = psnrs.avg
            results_msssim[i, j] = msssims.avg
            for t in metrics:
                t.clear()

    print("SNR: {}".format(results_snr.tolist()))
    print("CBR: {}".format(results_cbr.tolist()))
    print("PSNR: {}".format(results_psnr.tolist()))
    print("MS-SSIM: {}".format(results_msssim.tolist()))
    print("Finish Test!")


def train_one_epoch_denoiser(
    epoch, global_step, net, train_loader, optimizer, logger, args, config
):
    net.train()
    elapsed, losses = [AverageMeter() for _ in range(2)]

    # --- Freeze encoder/channel ---
    for name, param in net.named_parameters():
        if "feature_denoiser" in name or "decoder" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    for batch_idx, data in enumerate(train_loader):
        start_time = time.time()
        global_step += 1

        # Handle different data formats
        input = data[0] if args.trainset == "CIFAR10" else data
        input = input.cuda()

        # Forward pass
        (
            recon_image,
            restored_feature,
            noisy_feature,
            feature,
            mask,
            CBR,
            SNR,
            mse,
            loss_G,
        ) = net(input)
        feature = feature.detach()
        noisy_feature = noisy_feature.detach()
        mask = mask.detach()
        noise = noisy_feature - feature
        pred_noise = noisy_feature - restored_feature

        # ---------------------- Loss Components ---------------------- #
        # (1) Orthogonal loss: encourage pred_noise ⟂ restored_feature
        orth_loss = masked_orthogonal_loss(restored_feature, feature, mask)

        # (2) MSE between restored_feature and ground-truth feature
        mse_loss = masked_mse_loss(restored_feature, feature, mask)

        # (3) Noise mean regularization (only for AWGN)
        if args.channel_type == "awgn":
            noise_mean_real = (
                pred_noise.real.mean()
                if torch.is_complex(pred_noise)
                else pred_noise.mean()
            )
            noise_mean_reg = noise_mean_real**2
        else:
            noise_mean_reg = torch.tensor(0.0, device=input.device)

        # (4) Self-consistency: D(feature + pred_noise) ≈ feature
        restored_twice = net.feature_denoiser(feature + pred_noise)
        self_loss = masked_mse_loss(restored_twice, feature, mask)

        # (5) emphasize decoder's reconstruction quality
        no_noise_recon_image = net.decoder(feature, 60, net.model)
        no_noise_recon_loss = net.distortion_loss.forward(
            input, no_noise_recon_image.clamp(0.0, 1.0)
        )

        # ---------------------- Combine ---------------------- #
        a_1, a_2, a_3, a_4, a_5, a_6 = config.alpha_losses  # tuple of 6 weights
        total_loss = (
            a_1 * orth_loss
            + a_2 * mse_loss
            + a_3 * noise_mean_reg
            + a_4 * self_loss
            + a_5 * no_noise_recon_loss
            + a_6 * loss_G
        )

        # print(total_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # ---------------------- Logging ---------------------- #
        elapsed.update(time.time() - start_time)
        losses.update(total_loss.item())
        if (global_step % config.print_step) == 0:
            logger.info(
                f"[Epoch {epoch} | Step {global_step}] "
                f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                f"Orth {orth_loss.item():.4f} | MSE {mse_loss.item():.4f} | "
                f"MeanReg {noise_mean_reg.item():.6f} | Self {self_loss.item():.4f} |"
                f"NoNoiseRecon {no_noise_recon_loss.item():.4f} | Recon {loss_G.item():.4f} |"
            )
            elapsed.clear()
            losses.clear()

    return global_step
