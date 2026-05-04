"""Cell-type-conditioned decoder: latent z + type -> native expression simplex."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CellTypeDecoder(nn.Module):
    def __init__(self, z_dim, n_types, n_genes,
                 type_embed_dim = 32, hidden = (256, 256)):
        super().__init__()
        self.type_embed = nn.Embedding(n_types, type_embed_dim)
        dims = [z_dim + type_embed_dim, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.Softplus()]
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(hidden[-1], n_genes)
        self.register_buffer("type_logit_shift", None)
        self.use_logit_shift = False
        with torch.no_grad():
            self.out.weight.mul_(0.1)
            self.out.bias.zero_()

    def forward(self, z, cell_type):
        e = self.type_embed(cell_type)
        h = self.trunk(torch.cat([z, e], dim=-1))
        logits = self.out(h)
        if self.use_logit_shift and self.type_logit_shift is not None:
            logits = logits + self.type_logit_shift[cell_type]
        return F.softmax(logits, dim=-1)
