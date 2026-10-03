"""Classification losses and batch mixing."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def build_criterion(kind="ce", **kw):
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        if kw.get("weight") is None:
            raise ValueError("ce_weighted requires weight")
        return nn.CrossEntropyLoss(weight=kw["weight"])
    raise ValueError(kind)


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(label_smoothing=smoothing)
    def forward(self, logits, target):
        return self.loss(logits, target)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be nonnegative")
        self.gamma = gamma
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))
    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        loss = -(1 - logp.exp()).pow(self.gamma) * logp
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def class_weights(counts, beta=0.0):
    n = torch.as_tensor(counts, dtype=torch.float64)
    if (n <= 0).any() or beta < 0 or beta >= 1:
        raise ValueError("counts must be positive, beta in [0,1)")
    w = 1 / n if beta == 0 else (1-beta) / (1-torch.pow(torch.tensor(beta, dtype=n.dtype), n))
    return (w / w.mean()).float()


def mix_batch(x, y, alpha=1.0, mode="cutmix"):
    if alpha <= 0 or mode not in ("cutmix", "mixup"):
        raise ValueError("alpha > 0 and mode cutmix/mixup required")
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        return lam*x + (1-lam)*x[perm], (y, y[perm], lam)
    _, _, h, w = x.shape
    rw, rh = int(w*np.sqrt(1-lam)), int(h*np.sqrt(1-lam))
    cx, cy = np.random.randint(w), np.random.randint(h)
    x1, x2 = max(0, cx-rw//2), min(w, cx+(rw+1)//2)
    y1, y2 = max(0, cy-rh//2), min(h, cy+(rh+1)//2)
    out = x.clone()
    out[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    lam = 1 - (x2-x1)*(y2-y1)/(w*h)
    return out, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    a, b, lam = targets
    return lam*criterion(logits, a)+(1-lam)*criterion(logits, b)
