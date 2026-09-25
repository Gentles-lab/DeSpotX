"""DeSpotX model: GAT encoder + cell-type decoder."""

import math
import numpy as np
import torch
import torch.nn as nn

from .decoder import CellTypeDecoder
from .encoder import GATEncoder


class DeSpotX(nn.Module):
    """DeSpotX: cell-type-aware decontamination."""

    def __init__(self, n_genes, n_types, z_dim = 16,
                 hidden = 256, heads = 4,
                 anchor_mask=None, anchor_weight = 50.0,
                 anchor_warmup_epochs = 3,
                 apply_output_mask = True,
                 eps_prior_weight = 1.0,
                 chi_mode = "other_type_hard", chi_soft_leak = 0.1,
                 edge_feat = "kernel",
                 mu_eps_strategy = "clamp",
                 mu_eps_hyperprior_weight = 1.0,
                 n_gat_layers = 1,
                 eps_head_init = 0.20):
        super().__init__()
        self.n_genes = n_genes
        self.n_types = n_types
        self.z_dim = z_dim
        self.eps_prior_weight = eps_prior_weight
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
        self.eps_head = nn.Linear(hidden, 1)
        # eps_head_init seeds the architectural ε prior (default 0.20).
        eps_head_init = float(np.clip(eps_head_init, 1e-3, 1 - 1e-3))
        nn.init.constant_(self.eps_head.bias,
                          math.log(eps_head_init / (1 - eps_head_init)))
        self.log_theta = nn.Parameter(torch.zeros(n_genes))

        if anchor_mask is not None:
            self.register_buffer(
                "anchor_mask",
                torch.from_numpy(anchor_mask.astype(np.float32)))
        else:
            self.anchor_mask = None
        self.anchor_weight_target = anchor_weight
        self.anchor_warmup_epochs = anchor_warmup_epochs
        # Whether inference applies the hard output mask (1 - anchor_mask[ct]) to
        # the adjusted counts. False drops only that hard zeroing, independent of
        # the anchor training loss.
        self.apply_output_mask = apply_output_mask
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
        if set_decoder_shift:
            # Register on the decoder so .to(device) moves it correctly.
            self.decoder.register_buffer("type_logit_shift", shift)
            self.decoder.use_logit_shift = True
        else:
            self.register_buffer("type_logit_shift", shift)

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

        total = (loss_recon + loss_anchor + loss_eps_prior
                 + loss_mu_eps_hyper)
        return total, {
            "recon": loss_recon.item(),
            "anchor": loss_anchor.item(),
            "eps_prior": loss_eps_prior.item(),
            "mu_eps_hyper": loss_mu_eps_hyper.detach().item(),
        }

    @torch.no_grad()
    def adjust_counts(self, batch):
        """Decode phi, return x * (1-eps)*phi / [(1-eps)*phi + eps*chi] * (1 - anchor_mask[ct])."""
        x = batch["counts"]
        ct = batch["cell_type"]

        h, z = self._encode(batch)
        eps = torch.sigmoid(self.eps_head(h)).squeeze(-1)
        chi = self.encoder.chi_from_neighbors(batch)

        phi = self.decoder(z, ct)
        native = (1 - eps.unsqueeze(-1)) * phi
        ambient = eps.unsqueeze(-1) * chi
        p_native = native / (native + ambient + 1e-12)

        if self.apply_output_mask and self.anchor_mask is not None:
            adjusted = x * p_native * (1.0 - self.anchor_mask[ct])
        else:
            adjusted = x * p_native
        return adjusted, eps
