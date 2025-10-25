from utils import *
import torch
from loss.distortion import *
import time


def train_one_epoch(
    epoch, global_step, net, train_loader, optimizer, CalcuSSIM, logger, args, config
):
    net.train()
    elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]
    if args.trainset == "CIFAR10":
        for batch_idx, (input, label) in enumerate(train_loader):
            start_time = time.time()
            global_step += 1
            input = input.cuda()
            recon_image, CBR, SNR, mse, loss_G = net(input)
            loss = loss_G
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR)
            if mse.item() > 0:
                psnr = 10 * (torch.log(255.0 * 255.0 / mse) / np.log(10))
                psnrs.update(psnr.item())
                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0.0, 1.0)).mean().item()
                msssims.update(msssim)

            if (global_step % config.print_step) == 0:
                process = (
                    (global_step % train_loader.__len__())
                    / (train_loader.__len__())
                    * 100.0
                )
                log = " | ".join(
                    [
                        f"Epoch {epoch}",
                        f"Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]",
                        f"Time {elapsed.val:.3f}",
                        f"Loss {losses.val:.3f} ({losses.avg:.3f})",
                        f"CBR {cbrs.val:.4f} ({cbrs.avg:.4f})",
                        f"SNR {snrs.val:.1f} ({snrs.avg:.1f})",
                        f"PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})",
                        f"MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})",
                        f"Lr {config.learning_rate}",
                    ]
                )
                logger.info(log)
                for i in metrics:
                    i.clear()
    else:
        for batch_idx, input in enumerate(train_loader):
            start_time = time.time()
            global_step += 1
            input = input.cuda()
            recon_image, CBR, SNR, mse, loss_G = net(input)
            loss = loss_G
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR)
            if mse.item() > 0:
                psnr = 10 * (torch.log(255.0 * 255.0 / mse) / np.log(10))
                psnrs.update(psnr.item())
                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0.0, 1.0)).mean().item()
                msssims.update(msssim)

            if (global_step % config.print_step) == 0:
                process = (
                    (global_step % train_loader.__len__())
                    / (train_loader.__len__())
                    * 100.0
                )
                log = " | ".join(
                    [
                        f"Epoch {epoch}",
                        f"Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]",
                        f"Time {elapsed.val:.3f}",
                        f"Loss {losses.val:.3f} ({losses.avg:.3f})",
                        f"CBR {cbrs.val:.4f} ({cbrs.avg:.4f})",
                        f"SNR {snrs.val:.1f} ({snrs.avg:.1f})",
                        f"PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})",
                        f"MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})",
                        f"Lr {config.learning_rate}",
                    ]
                )
                logger.info(log)
                for i in metrics:
                    i.clear()
    for i in metrics:
        i.clear()

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
                if args.trainset == "CIFAR10":
                    for batch_idx, (input, label) in enumerate(test_loader):
                        start_time = time.time()
                        input = input.cuda()
                        recon_image, CBR, SNR, mse, loss_G = net(input, SNR, rate)

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
                else:
                    for batch_idx, batch in enumerate(test_loader):
                        input, names = batch
                        start_time = time.time()
                        input = input.cuda()
                        recon_image, CBR, SNR, mse, loss_G = net(input, SNR, rate)
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
                            MSSSIM = -10 * np.math.log10(1 - msssim)
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
