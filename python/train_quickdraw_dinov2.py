import datasets
import multiprocessing as mp
import models
import argparse
import torch
from torch.nn import functional as F
from torchvision.transforms import v2 as T
import tqdm
import time
from transformers import AutoImageProcessor


def main(args):

    # Load the QuickDraw dataset and split it into training and testing sets
    # Running only on 1 shard (1% of data) for demonstration purposes
    test_size = 0.05
    seed = 42
    shard_size = 100
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    def transform_data(batch):
        image_transforms = [
            processor(images=img, return_tensors="pt") for img in batch["image"]
        ]
        inputs = torch.stack(
            [
                img_transform["pixel_values"].squeeze(0)
                for img_transform in image_transforms
            ]
        )
        label = torch.tensor(batch["label"])
        return {"image": inputs, "label": label}

    dataset = datasets.load_dataset("quickdraw", num_proc=mp.cpu_count())
    dataset = dataset["train"].train_test_split(test_size=test_size, seed=seed)
    train_dataset = dataset["train"].shuffle(seed).shard(num_shards=shard_size, index=0)
    train_dataset.set_transform(transform_data)
    test_dataset = dataset["test"].shuffle(seed).shard(num_shards=shard_size, index=0)
    test_dataset.set_transform(transform_data)

    torch.autograd.set_detect_anomaly(True)

    model = models.DinoV2(
        hidden_size=768,  # Dino V2 hidden size
        num_classes=345,  # number of classes in the dataset
        freeze=False,  # freeze Dino gradient
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    # loss_scaler = torch.cuda.amp.GradScaler()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=mp.cpu_count() // 2,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=mp.cpu_count() // 2,
        pin_memory=True,
    )

    for epoch in range(args.epochs):
        model.train()

        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for i, batch in progress_bar:
            image = batch["image"].cuda()
            label = batch["label"].cuda()

            # with torch.cuda.amp.autocast():  # half precision
            prediction, logits = model(image)
            loss = F.cross_entropy(logits, label)

            optimizer.zero_grad()
            # loss_scaler.scale(loss).backward()  # backpropagation with half precision
            # loss_scaler.step(optimizer)
            # loss_scaler.update()

            loss.backward()
            optimizer.step()

            progress_bar.set_description(f"Loss: {loss.item():.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                image = batch["image"].cuda()
                label = batch["label"].cuda()

                prediction, logits = model(image)
                correct += (prediction.argmax(1) == label).sum().item()
                total += label.size(0)

        print(f"Epoch {epoch}: Test Accuracy: {correct / total:.4f}")

    torch.save(model.state_dict(), f"quickdraw_{time.time()}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    args = parser.parse_args()
    main(args)
