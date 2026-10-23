"""Failure-conditioned reverse attention with a local evidence residual.

Rows of A are later steps; columns are earlier steps receiving evidence.
PyTorch is an optional dependency used only by this experiment.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F

VERSION = "attentionmap_v1"
VARIANTS = ("linear", "mlp", "fixed_graph", "dense_unconditioned", "dense", "dense_no_prior")


class AttentionMapRanker(nn.Module):
    def __init__(self, input_dim: int, failure_dim: int = 256,
                 hidden: int = 32, variant: str = "dense") -> None:
        super().__init__()
        if variant not in VARIANTS or input_dim < 256 or hidden < 1:
            raise ValueError("Invalid attention model configuration")
        self.variant = variant
        self.hidden = hidden
        self.linear = nn.Linear(input_dim, 1)
        self.encode = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh())
        self.failure_encode = nn.Sequential(nn.Linear(failure_dim, hidden), nn.Tanh())
        self.local = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        # Q/K use only decision/action and observation text, before external
        # failure interactions or cached q. Thus the A-only condition ablation
        # removes external failure information from A, while retaining it in u/p.
        self.relation_encode = nn.Sequential(nn.Linear(256, hidden), nn.Tanh())
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.failure_query = nn.Linear(hidden, hidden, bias=False)
        self.failure_key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.seed = nn.Linear(3 * hidden, 1)
        self.relation_score = nn.Sequential(nn.Linear(2 * hidden + 1, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.graph_bias = nn.Parameter(torch.tensor(1.0))
        self.distance_penalty = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: Tensor, failure: Tensor, prior: Tensor,
                *, shuffle_seed: int | None = None) -> dict[str, Tensor]:
        n = len(x)
        if x.ndim != 2 or n == 0 or prior.shape != (n, n) or failure.ndim != 1:
            raise ValueError("Invalid attention input dimensions")
        if not all(torch.isfinite(t).all() for t in (x, failure, prior)):
            raise ValueError("Nonfinite attention inputs")
        if (prior < 0).any() or (prior > 1).any() or torch.triu(prior, diagonal=0).any():
            raise ValueError("Prior must use [target, source] with source < target")
        if self.variant == "linear":
            scores = self.linear(x).squeeze(-1)
            return {"scores": scores, "local": scores, "relation": torch.zeros_like(scores)}
        h = self.encode(x)
        f = self.failure_encode(failure).expand(n, -1)
        local_features = torch.cat([h, f, h * f], dim=-1)
        local = self.local(local_features).squeeze(-1)
        if self.variant == "mlp":
            return {"scores": local, "local": local, "relation": torch.zeros_like(local)}
        relation_h = self.relation_encode(x[:, :256])
        query, key = self.query(relation_h), self.key(relation_h)
        if self.variant != "dense_unconditioned":
            query = query + self.failure_query(f)
            key = key + self.failure_key(f)
        logits = query @ key.T / math.sqrt(self.hidden)
        if self.variant != "dense_no_prior":
            logits = logits + self.graph_bias * prior
        index = torch.arange(n, device=x.device)
        distance = (index[:, None] - index[None, :]).clamp(min=0) / max(n - 1, 1)
        logits = logits - F.softplus(self.distance_penalty) * distance
        allowed = torch.ones((n, n), dtype=torch.bool, device=x.device).tril()
        if self.variant == "fixed_graph":
            allowed &= (prior > 0) | torch.eye(n, dtype=torch.bool, device=x.device)
        attention = torch.softmax(logits.masked_fill(~allowed, -torch.inf), dim=-1)
        if shuffle_seed is not None:
            # Permute each row over its allowed positions; preserve row mass,
            # temporal mask and value distribution while breaking source identity.
            generator = torch.Generator(device=x.device).manual_seed(shuffle_seed)
            shuffled = attention.clone()
            for j in range(n):
                cols = torch.where(allowed[j])[0]
                shuffled[j, cols] = attention[j, cols[torch.randperm(len(cols), generator=generator, device=x.device)]]
            attention = shuffled
        seeds = torch.softmax(self.seed(local_features).squeeze(-1), dim=0)
        mass_weights = seeds[:, None] * attention
        mass = mass_weights.sum(dim=0)
        context = mass_weights.T @ self.value(relation_h) / mass[:, None].clamp_min(1e-8)
        features = torch.cat([context, h * context, torch.log1p(n * mass)[:, None]], dim=-1)
        relation = self.relation_score(features).squeeze(-1)
        return {"scores": local + relation, "local": local, "relation": relation,
                "attention": attention, "seeds": seeds, "mass": mass}


def listwise_loss(scores: Tensor, root_ordinals: list[int]) -> Tensor:
    if not root_ordinals or len(set(root_ordinals)) != len(root_ordinals):
        raise ValueError("Nonempty unique training labels required")
    if any(i < 0 or i >= len(scores) for i in root_ordinals):
        raise ValueError("Training label outside trajectory")
    return -torch.logsumexp(torch.log_softmax(scores, dim=0)[root_ordinals], dim=0)
