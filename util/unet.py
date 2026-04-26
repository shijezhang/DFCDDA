import torch
import torch.nn as nn
import math


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class CondEmbedder(nn.Module):
    def __init__(self, cond_dim, hidden_dim):
        super(CondEmbedder, self).__init__()
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, cond):
        return self.cond_embed(cond)


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, time_features, cond_features, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.emb_time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_features, out_features)
        )
        self.emb_cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_features, out_features)
        )
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, emb_t, emb_y):
        h = self.fc(x)
        h = h + self.emb_time(emb_t) + self.emb_cond(emb_y)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)

        return h


class Linear_UNet(nn.Module):
    def __init__(self, input_dim=128, cond_dim=64, hidden_dims=[128, 128, 64, 64], dropout=0):
        super(Linear_UNet, self).__init__()
        self.hidden_dims = hidden_dims

        self.time_embedding = TimestepEmbedder(hidden_dims[0])
        self.cond_embedding = CondEmbedder(cond_dim, hidden_dims[0])

        self.layers = nn.ModuleList()

        self.layers.append(ResidualBlock(input_dim, hidden_dims[0], hidden_dims[0], hidden_dims[0], dropout))

        for i in range(len(hidden_dims) - 1):
            self.layers.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i + 1], hidden_dims[0], hidden_dims[0], dropout))

        self.reverse_layers = nn.ModuleList()
        for i in reversed(range(len(hidden_dims) - 1)):
            self.reverse_layers.append(
                ResidualBlock(hidden_dims[i + 1], hidden_dims[i], hidden_dims[0], hidden_dims[0], dropout))

        self.out = nn.Sequential(
            nn.Linear(hidden_dims[0], input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.SiLU(),
            nn.Linear(input_dim * 2, input_dim)
        )

    def forward(self, x, t, y):
        emb_t = self.time_embedding(t)
        emb_y = self.cond_embedding(y)

        history = []
        for layer in self.layers:
            x = layer(x, emb_t, emb_y)
            history.append(x)

        history.pop()

        for layer in self.reverse_layers:
            x = layer(x, emb_t, emb_y)
            x = x + history.pop()

        x = self.out(x)
        return x
