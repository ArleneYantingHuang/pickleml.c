import torch


class SimpleFCN(torch.nn.Module):
    def __init__(self, input_size=(28, 28), num_classes=10):
        super().__init__()
        num_pixels = input_size[0] * input_size[1]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(num_pixels, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, num_classes),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = x.flatten(1)
        x = self.net(x)
        return x
