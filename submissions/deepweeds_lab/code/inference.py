"""Validation-selected inference methods. Never fit any setting on test."""
from __future__ import annotations
import copy
import numpy as np


def predict_logits(model, loader, device, view=None):
    import torch
    model.eval()
    names, ys, zs = [], [], []
    with torch.inference_mode():
        for images, labels, batch_names in loader:
            images = images.to(device, non_blocking=True)
            transformed = view(images) if view else images
            if isinstance(transformed, (tuple, list)):
                logits = torch.stack([model(x) for x in transformed]).mean(0)
            else:
                logits = model(transformed)
            names.extend(batch_names)
            ys.append(labels.numpy())
            zs.append(logits.float().cpu().numpy())
    return names, np.concatenate(ys), np.concatenate(zs)


def view_identity(x):
    return x


def view_hflip(x):
    import torch
    return torch.flip(x, dims=(-1,))


def views_multicrop(x, crop):
    h, w = x.shape[-2:]
    if crop <= 0 or crop > min(h,w):
        raise ValueError("invalid crop")
    return [x[..., top:top+crop, left:left+crop] for top,left in ((0,0),(0,w-crop),(h-crop,0),(h-crop,w-crop),((h-crop)//2,(w-crop)//2))]


def views_multiscale(x, sizes):
    from torch.nn import functional as F
    return [F.interpolate(x,size=(int(s),int(s)),mode="bilinear",align_corners=False) for s in sizes]


def _softmax(z):
    z=np.asarray(z,dtype=np.float64)
    z=z-z.max(axis=-1,keepdims=True)
    a=np.exp(z)
    return a/a.sum(axis=-1,keepdims=True)


def aggregate_views(logits_per_view, space="prob"):
    arrays=[np.asarray(z) for z in logits_per_view]
    if not arrays or any(z.shape != arrays[0].shape for z in arrays):
        raise ValueError("views must be nonempty and have equal shape")
    if space == "prob":
        return np.mean([_softmax(z) for z in arrays],axis=0)
    if space == "logit":
        return _softmax(np.mean(arrays,axis=0))
    raise ValueError(space)


def ensemble_probs(list_of_probs):
    arrays=[np.asarray(p,dtype=np.float64) for p in list_of_probs]
    if not arrays or any(p.shape != arrays[0].shape for p in arrays):
        raise ValueError("probabilities must be nonempty and aligned")
    if any(not np.allclose(p.sum(1),1,atol=1e-5) for p in arrays):
        raise ValueError("input must contain probabilities")
    return np.mean(arrays,axis=0)


def fit_temperature(val_logits, val_labels):
    z=np.asarray(val_logits,dtype=np.float64)
    y=np.asarray(val_labels,dtype=np.int64)
    if z.ndim != 2 or len(z) != len(y) or len(y)==0:
        raise ValueError("invalid validation logits/labels")
    # Bounded scalar optimization keeps T positive and avoids an extreme solution.
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp
    def nll(log_t):
        scaled=z/np.exp(log_t)
        return float(np.mean(logsumexp(scaled,axis=1)-scaled[np.arange(len(y)),y]))
    result=minimize_scalar(nll,bounds=(np.log(.05),np.log(20.0)),method="bounded")
    if not result.success:
        raise RuntimeError("temperature optimization failed")
    return float(np.exp(result.x))


def apply_temperature(logits,T):
    if not np.isfinite(T) or T <= 0:
        raise ValueError("T must be positive")
    return _softmax(np.asarray(logits)/T)


def predict_variant(model, loader, device, method="I00", temperature=None):
    """One ordered pass over a split; TTA views stay inside the same pass.

    Returns filenames, labels, probabilities, and base one-view logits. The
    latter let I04 save calibrated and uncalibrated predictions without a
    second test pass.
    """
    import torch
    model = fuse_conv_bn(model) if method == "I05" else model
    model.eval()
    names, labels, probs, base = [], [], [], []
    if method not in ("I00", "I01", "I02", "I03", "I04", "I05"):
        raise ValueError(method)
    with torch.inference_mode():
        for x, y, batch_names in loader:
            x=x.to(device,non_blocking=True)
            z=model(x).float().cpu().numpy()
            if method == "I01":
                z2=model(view_hflip(x)).float().cpu().numpy()
                p=aggregate_views([z,z2],"prob")
            elif method == "I02":
                zs=[model(v).float().cpu().numpy() for v in views_multicrop(x,192)]
                p=aggregate_views([z,*zs],"logit")
            elif method == "I03":
                # Use the base 224 view plus a 256 view; only CNNs with
                # flexible spatial dimensions should use this method.
                z2=model(views_multiscale(x,[256])[0]).float().cpu().numpy()
                p=aggregate_views([z,z2],"prob")
            elif method == "I04":
                if temperature is None:
                    raise ValueError("I04 requires validation-fitted temperature")
                p=apply_temperature(z,temperature)
            else:
                p=_softmax(z)
            names.extend(batch_names)
            labels.append(y.numpy())
            probs.append(p)
            base.append(z)
    return names,np.concatenate(labels),np.concatenate(probs),np.concatenate(base)


def fuse_conv_bn(model):
    import torch
    from torch import nn
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    fused=copy.deepcopy(model).eval()
    def walk(parent):
        children=list(parent.named_children())
        for (_, a), (b_name,b) in zip(children,children[1:]):
            if isinstance(a,nn.Conv2d) and isinstance(b,nn.BatchNorm2d):
                a_name=next(name for name,obj in children if obj is a)
                setattr(parent,a_name,fuse_conv_bn_eval(a,b))
                setattr(parent,b_name,nn.Identity())
        for child in parent.children():
            walk(child)
    walk(fused)
    return fused
