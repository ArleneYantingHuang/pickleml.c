from models import *

import datasets
import torch
from torchvision.transforms import v2 as T
from torch.nn import functional as F
from tqdm import tqdm

import argparse


def transform_data(batch):
    image_transform = T.Compose(
        [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize((0.5,), (0.5,))]
    )
    image = torch.stack([image_transform(img) for img in batch["image"]])
    label = torch.tensor(batch["label"])
    return {"image": image, "label": label}


def main(args):
    dataset = datasets.load_dataset("mnist")

    train_dataset = (
        dataset["train"]
        .map(transform_data, batched=True, batch_size=32, num_proc=16)
        .with_format("torch")
    )
    test_dataset = (
        dataset["test"]
        .map(transform_data, batched=True, batch_size=32, num_proc=16)
        .with_format("torch")
    )

    model = SimpleFCN().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_scaler = torch.cuda.amp.GradScaler()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    for epoch in range(args.epochs):
        model.train()

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, batch in progress_bar:
            image = batch["image"].cuda()
            label = batch["label"].cuda()

            with torch.cuda.amp.autocast():
                pred = model(image)
                loss = F.cross_entropy(pred, label)

            optimizer.zero_grad()
            loss_scaler.scale(loss).backward()
            optimizer.step()

            progress_bar.set_postfix(train_loss=loss.item())

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(test_loader):
                image = batch["image"].cuda()
                label = batch["label"].cuda()

                pred_logits = model(image)
                pred_label = pred_logits.argmax(dim=1)
                correct += (pred_label == label).sum().item()
                total += len(label)

        print(f"Epoch {epoch}, Accuracy {correct/total}")

    torch.save(model.state_dict(), "mnist_model.pth")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--batch_size", type=int, default=64)
    argparser.add_argument("--epochs", type=int, default=10)
    argparser.add_argument("--lr", type=float, default=0.001)

    main(argparser.parse_args())
