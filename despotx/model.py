"""DeSpotX model: GAT encoder + cell-type decoder + latent diffusion prior."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import CellTypeDecoder
from .encoder import GATEncoder


def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) *
                      torch.arange(half, device=t.device) / (half - 1))
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class LatentScoreNet(nn.Module):
    """Score network for the latent diffusion. Conditioned on the encoder's
    spatial-context embedding h and the cell-type embedding."""

    def __init__(self, z_dim, h_dim, n_types,
                 t_emb_dim = 64, type_embed_dim = 32,
                 hidden = 256):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.type_embed = nn.Embedding(n_types, type_embed_dim)
        in_dim = z_dim + t_emb_dim + h_dim + type_embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, z_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_t, t, h, cell_type):
        t_emb = sinusoidal_embedding(t, self.t_emb_dim)
        ct_emb = self.type_embed(cell_type)
        return self.net(torch.cat([z_t, t_emb, h, ct_emb], dim=-1))


class DeSpotX(nn.Module):
    """DeSpotX: cell-type-aware, manifold-regularised decontamination."""

    def __init__(self, n_genes, n_types, z_dim = 16,
                 hidden = 256, heads = 4, T = 100,
                 anchor_mask=None, anchor_weight = 50.0,
                 anchor_warmup_epochs = 3,
                 diff_loss_weight = 1.0, refine_t_frac = 0.2,
                 eps_prior_weight = 1.0,
                 chi_mode = "other_type_hard", chi_soft_leak = 0.1,
                 edge_feat = "kernel",
                 diff_detach = True,
                 mu_eps_strategy = "clamp",
                 mu_eps_hyperprior_weight = 1.0,
                 n_gat_layers = 1,
                 eps_head_init = 0.20):
        super().__init__()
        self.n_genes = n_genes
        self.n_types = n_types
        self.z_dim = z_dim
        self.T = T
        self.diff_weight = diff_loss_weight
        self.refine_t_frac = refine_t_frac
        self.eps_prior_weight = eps_prior_weight
        self.diff_detach = diff_detach
        if mu_eps_strategy not in ("free", "clamp", "hyperprior", "fixed"):
            raise ValueError(
                f"mu_eps_strategy must be one of 'free', 'clamp', "
                f"'hyperprior', 'fixed'; got {mu_eps_strategy!r}")
        self.mu_eps_strategy = mu_eps_strategy
        self.mu_eps_hyperprior_weight = mu_eps_hyperprior_weight

        # Beta(prior_a, prior_b) on ε with concentration ν=20; μ_ε learnable
        # (clamped to [0.05, 0.95] in training_step to prevent runaway, 2026-04-28).
        self.prior_concentration = 20.0
        init_logit = float(np.log(0.20 / 0.80))
        if mu_eps_strategy == "fixed":
            self.register_buffer(
                "eps_prior_mean_logit", torch.tensor(init_logit))
        else:
            self.eps_prior_mean_logit = nn.Parameter(torch.tensor(init_logit))

        self.encoder = GATEncoder(
            n_genes=n_genes, n_types=n_types, z_dim=z_dim,
            hidden=hidden, heads=heads,
            chi_mode=chi_mode, chi_soft_leak=chi_soft_leak,
            edge_feat=edge_feat,
            n_gat_layers=n_gat_layers)
        self.decoder = CellTypeDecoder(z_dim, n_types, n_genes)
        self.score_net = LatentScoreNet(
            z_dim=z_dim, h_dim=hidden, n_types=n_types)
        self.eps_head = nn.Linear(hidden, 1)
        # eps_head_init seeds the architectural ε prior (default 0.20).
        eps_head_init = float(np.clip(eps_head_init, 1e-3, 1 - 1e-3))
        nn.init.constant_(self.eps_head.bias,
                          math.log(eps_head_init / (1 - eps_head_init)))
        self.log_theta = nn.Parameter(torch.zeros(n_genes))

        self.register_buffer(
            "alphas_cumprod",
            torch.cumprod(1.0 - torch.linspace(1e-4, 0.02, T), dim=0))

        if anchor_mask is not None:
            self.register_buffer(
                "anchor_mask",
                torch.from_numpy(anchor_mask.astype(np.float32)))
        else:
            self.anchor_mask = None
        self.anchor_weight_target = anchor_weight
        self.anchor_warmup_epochs = anchor_warmup_epochs
        self._current_anchor_weight = 0.0

    # ------------------------------------------------------------------
    def set_epoch(self, epoch):
        # Paper §B.5: lambda_A linearly increases from 0 at epoch 0 to the
        # target by epoch == anchor_warmup_epochs, then stays at the target.
        if self.anchor_warmup_epochs > 0:
            self._current_anchor_weight = self.anchor_weight_target * min(
                1.0, epoch / max(self.anchor_warmup_epochs, 1))
        else:
            self._current_anchor_weight = self.anchor_weight_target

    def warm_start_decoder(self, adata, cell_type_col,
                            set_decoder_shift = True):
        """Set decoder per-type logit shift from empirical log-mean expression,
        and register the encoder's frequency-balanced global-ambient fallback."""
        import scipy.sparse as sp

        X = adata.X
        ct = np.asarray(adata.obs[cell_type_col].values)
        _, ct_int = np.unique(ct, return_inverse=True)
        means = np.zeros((self.n_types, self.n_genes), dtype=np.float32)
        for t in range(self.n_types):
            rows = np.where(ct_int == t)[0]
            Xt = X[rows]
            total = (np.asarray(Xt.sum(axis=0)).ravel() if sp.issparse(Xt)
                     else Xt.sum(axis=0))
            means[t] = (total + 1e-3) / (total + 1e-3).sum()
        log_means = np.log(means + 1e-8)
        log_means -= log_means.mean(axis=1, keepdims=True)
        shift = torch.from_numpy(log_means.astype(np.float32))
        self.register_buffer("type_logit_shift", shift)
        if set_decoder_shift:
            self.decoder.type_logit_shift = self.type_logit_shift
            self.decoder.use_logit_shift = True

        # Frequency-balanced fallback: equal weight per cell type.
        global_ambient = means.mean(axis=0)
        global_ambient = global_ambient / global_ambient.sum()
        self.encoder.set_global_ambient(global_ambient.astype(np.float32))

    # ------------------------------------------------------------------
    def _encode(self, batch):
        h = self.encoder._encode_graph(batch)
        z = self.encoder.head_z_loc(h)
        return h, z

    def training_step(self, batch):
        """One gradient step of the joint objective (Eq. 6 of the paper)."""
        x = batch["counts"]
        ct = batch["cell_type"]
        B = x.size(0)
        device = x.device

        h, z = self._encode(batch)
        phi = self.decoder(z, ct)
        chi = self.encoder.chi_from_neighbors(batch)
        eps = torch.sigmoid(self.eps_head(h)).squeeze(-1)

        d = x.sum(dim=-1, keepdim=True).clamp(min=1.0)
        theta = self.log_theta.exp().clamp(1e-3, 1e4)
        mu = d * ((1 - eps.unsqueeze(-1)) * phi + eps.unsqueeze(-1) * chi)
        mu = mu.clamp(min=1e-8)
        probs = mu / (mu + theta)
        nb = torch.distributions.NegativeBinomial(total_count=theta, probs=probs)
        loss_recon = -nb.log_prob(x).sum(dim=-1).mean()

        loss_anchor = torch.tensor(0.0, device=device)
        if self.anchor_mask is not None and self._current_anchor_weight > 0:
            anchor_per_cell = self.anchor_mask[ct]
            native_contrib = (1 - eps.unsqueeze(-1)) * phi * d
            loss_anchor = self._current_anchor_weight * (
                native_contrib * anchor_per_cell).sum(dim=-1).mean()

        t = torch.randint(0, self.T, (B,), device=device)
        noise = torch.randn_like(z)
        abar = self.alphas_cumprod[t].unsqueeze(-1)
        # diff_detach (paper §B.4): stop-gradient between L_diff and encoder.
        z_in = z.detach() if self.diff_detach else z
        h_in = h.detach() if self.diff_detach else h
        z_t = torch.sqrt(abar) * z_in + torch.sqrt(1 - abar) * noise
        pred_noise = self.score_net(z_t, t, h_in, ct)
        loss_diff = F.mse_loss(pred_noise, noise)

        prior_mean = torch.sigmoid(self.eps_prior_mean_logit)
        if self.mu_eps_strategy == "clamp":
            prior_mean = prior_mean.clamp(0.05, 0.95)
        conc = self.prior_concentration
        prior_a = prior_mean * conc + 1e-3
        prior_b = (1 - prior_mean) * conc + 1e-3
        neg_log_prior = -((prior_a - 1) * torch.log(eps.clamp(min=1e-6))
                          + (prior_b - 1) * torch.log((1 - eps).clamp(min=1e-6)))
        loss_eps_prior = self.eps_prior_weight * neg_log_prior.mean()

        # Optional Beta(2,8) hyperprior on μ_ε.
        loss_mu_eps_hyper = torch.tensor(0.0, device=device)
        if self.mu_eps_strategy == "hyperprior":
            pm_safe = prior_mean.clamp(min=1e-6, max=1 - 1e-6)
            loss_mu_eps_hyper = self.mu_eps_hyperprior_weight * -(
                1.0 * torch.log(pm_safe) + 7.0 * torch.log(1 - pm_safe))

        total = (loss_recon + loss_anchor
                 + self.diff_weight * loss_diff + loss_eps_prior
                 + loss_mu_eps_hyper)
        return total, {
            "recon": loss_recon.item(),
            "anchor": loss_anchor.item(),
            "diff": loss_diff.item(),
            "eps_prior": loss_eps_prior.item(),
            "mu_eps_hyper": loss_mu_eps_hyper.detach().item(),
        }

    @torch.no_grad()
    def adjust_counts(self, batch, n_refine_steps = 10):
        """DDIM-refine z, decode phi, return x * (1-eps)*phi / [(1-eps)*phi + eps*chi]."""
        x = batch["counts"]
        ct = batch["cell_type"]
        B = x.size(0)
        device = x.device

        h, z_enc = self._encode(batch)
        eps = torch.sigmoid(self.eps_head(h)).squeeze(-1)
        d = x.sum(dim=-1, keepdim=True).clamp(min=1.0)
        chi = self.encoder.chi_from_neighbors(batch)

        # diff_weight=0: skip DDIM (untrained score net would corrupt z).
        if self.diff_weight == 0:
            z = z_enc
        else:
            t_max = max(1, int(self.T * self.refine_t_frac))
            steps = torch.linspace(0, t_max - 1, n_refine_steps,
                                   device=device).long()
            t_start = steps[-1]
            abar_start = self.alphas_cumprod[t_start]
            noise = torch.randn_like(z_enc)
            z = torch.sqrt(abar_start) * z_enc + torch.sqrt(1 - abar_start) * noise

            for i in reversed(range(len(steps))):
                t = steps[i]
                t_b = t.expand(B)
                abar_t = self.alphas_cumprod[t]
                pred_noise = self.score_net(z, t_b, h, ct)
                pred_z0 = (z - torch.sqrt(1 - abar_t) * pred_noise) / \
                    torch.sqrt(abar_t).clamp(min=1e-3)
                if i == 0:
                    z = pred_z0
                else:
                    # DDIM eta=0 (deterministic).
                    abar_prev = self.alphas_cumprod[steps[i - 1]]
                    z = (torch.sqrt(abar_prev) * pred_z0
                         + torch.sqrt(1 - abar_prev) * pred_noise)

        phi = self.decoder(z, ct)
        native = (1 - eps.unsqueeze(-1)) * phi
        ambient = eps.unsqueeze(-1) * chi
        p_native = native / (native + ambient + 1e-12)

        anchor_per_cell = (self.anchor_mask[ct]
                           if self.anchor_mask is not None
                           else torch.zeros_like(x))
        adjusted = x * p_native * (1.0 - anchor_per_cell)
        return adjusted, eps
