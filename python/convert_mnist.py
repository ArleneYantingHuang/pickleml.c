import torch
import argparse
from pathlib import Path


def main(args):
    model = torch.load("./python/mnist_model.pth")
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(model["net.0.weight"])
    model["net.0.weight"].cpu().numpy().tofile(
        args.output_dir / "mnist_model_0_weight.bin"
    )
    model["net.0.bias"].cpu().numpy().tofile(args.output_dir / "mnist_model_0_bias.bin")
    model["net.2.weight"].cpu().numpy().tofile(
        args.output_dir / "mnist_model_2_weight.bin"
    )
    model["net.2.bias"].cpu().numpy().tofile(args.output_dir / "mnist_model_2_bias.bin")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("output_dir", type=str, default=".")
    main(args=argparser.parse_args())
