import argparse


def create_parser():
    parser = argparse.ArgumentParser(description="SwinJSCC")
    parser.add_argument("--training", action="store_true", help="training or testing")
    parser.add_argument(
        "--pass-channel", action="store_true", help="whether to pass channel"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=256,
        choices=[256, 512],
        help="training image size",
    )
    parser.add_argument(
        "--trainset",
        type=str,
        default="DIV2K",
        choices=["COCO", "DIV2K"],
        help="train dataset name",
    )
    parser.add_argument(
        "--testset",
        type=str,
        default="Kodak",
        choices=["Kodak", "CLIC21", "ffhq", "COCO"],
        help="specify the testset for HR models",
    )
    parser.add_argument(
        "--channel-type",
        type=str,
        default="awgn",
        choices=["awgn", "rayleigh"],
        help="wireless channel model, awgn or rayleigh",
    )
    parser.add_argument("--cbrs", type=str, default="0.125", help="multiple cbrs")
    parser.add_argument("--snrs", type=str, default="10", help="multiple snrs")
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        choices=["base"],
        help="SwinJSCC model size",
    )
    parser.add_argument(
        "--amp", action="store_true", help="enable torch.cuda.amp mixed precision"
    )
    return parser
