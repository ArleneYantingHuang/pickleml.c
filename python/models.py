import torch
import torch.nn.functional as F
import einops
from transformers import AutoImageProcessor, AutoModel


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


class SimpleFCN2(torch.nn.Module):
    def __init__(self, input_size=(28, 28), num_classes=10):
        super().__init__()
        num_pixels = input_size[0] * input_size[1]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(num_pixels, 1024),
            torch.nn.ELU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ELU(),
            torch.nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = x.flatten(1)
        logits = self.net(x)
        x = F.softmax(logits, dim=1)
        return x, logits


class TransformerFeedForward(torch.nn.Module):
    def __init__(self, D, D_ff):  # D: embedding size, D_ff: feed-forward size
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(D, D_ff), torch.nn.GELU(), torch.nn.Linear(D_ff, D)
        )

    def forward(self, x):
        return self.net(x)


class TransformerLayer(torch.nn.Module):
    def __init__(
        self, D, H, D_ff
    ):  # D: embedding size, H: number of heads, D_ff: feed-forward size
        super().__init__()
        self.norm_attention = torch.nn.LayerNorm(D)
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=D, num_heads=H, batch_first=True
        )
        self.norm_ff = torch.nn.LayerNorm(D)
        self.ff = TransformerFeedForward(D, D_ff)

    def forward(self, x):
        y = self.norm_attention(x)
        y = self.attention(y, y, y)[0]
        x = x + y
        y = self.norm_ff(x)
        y = self.ff(x)
        x = x + y
        return x


class TransformerEmbedding(torch.nn.Module):
    def __init__(
        self, P, D, H, W
    ):  # P: patch size, D: embedding size, H: height, W: width
        super().__init__()
        self.P = P
        self.embedding = torch.nn.Linear(P * P, D)
        self.N = H * W // P // P
        self.positional_embedding = torch.nn.Parameter(torch.randn(self.N, D))
        self.class_token = torch.nn.Parameter(torch.randn(1, D))

    def forward(self, x):
        # (h ph) = H, (w pw) = W, (h w) = N; B, C, H, W -> B, C, N, D
        x = einops.rearrange(
            x, "B C (h ph) (w pw)  -> B (h w) (ph pw C)", ph=self.P, pw=self.P
        )
        # B, N, D
        x = self.embedding(x)
        # B, N, D + 1, N, D with broadcasting
        x = x + self.positional_embedding.unsqueeze(0)
        # 1, D -> 1, 1, D -> B, 1, D
        class_token = self.class_token.unsqueeze(0).repeat(x.size(0), 1, 1)
        # B, N + 1, D
        x = torch.cat([class_token, x], dim=1)
        return x


class TransformerClassifier(torch.nn.Module):
    def __init__(self, D, num_classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(D),
            torch.nn.Linear(D, 2 * D),
            torch.nn.GELU(),
            torch.nn.Linear(2 * D, num_classes),
        )

    def forward(self, x):
        logits = self.net(x)  # B, num_classes
        x = F.softmax(logits, dim=1)  # B, num_classes
        return x, logits


class TransformerClassifierComplex(torch.nn.Module):
    def __init__(self, D, num_classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(D),
            torch.nn.Linear(D, 2 * D),
            torch.nn.GELU(),
            torch.nn.Linear(2 * D, D),
            torch.nn.GELU(),
            torch.nn.Linear(D, D // 2),
            torch.nn.GELU(),
            torch.nn.Linear(D // 2, num_classes)
        )

    def forward(self, x):
        logits = self.net(x)  # B, num_classes
        x = F.softmax(logits, dim=1)  # B, num_classes
        return x, logits



class VisionTransformer(torch.nn.Module):
    def __init__(self, H, W, P, D, D_ff, num_heads, num_layers, num_classes):
        super().__init__()
        self.embedding = TransformerEmbedding(P, D, H, W)
        self.transformer = torch.nn.Sequential(
            *[TransformerLayer(D, num_heads, D_ff) for _ in range(num_layers)]
        )
        self.classifier = TransformerClassifier(D, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        encoder_hiden_states = self.transformer(x)  # B, N + 1, D
        class_token = encoder_hiden_states[:, 0, :]  # B, D
        score, logits = self.classifier(class_token)  # B, num_classes
        return score, logits


class DinoV2(torch.nn.Module):
    def __init__(self, hidden_size, num_classes, freeze = False):
        super().__init__()
        self.dinov2 = AutoModel.from_pretrained("facebook/dinov2-base")
        self.classifier = TransformerClassifier(hidden_size, num_classes)
        if freeze:
            for param in self.dinov2.parameters():
                param.requires_grad = False


    def forward(self, x):
        last_hidden_state = self.dinov2(pixel_values=x)[0]
        class_token = last_hidden_state[:, 0, :]
        score, logits = self.classifier(class_token)
        return score, logits
