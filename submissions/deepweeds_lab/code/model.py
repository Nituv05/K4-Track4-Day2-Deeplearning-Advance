"""Backbones, parameter groups and computational cost."""
from __future__ import annotations
import copy
SUGGESTED_BACKBONES = {"resnet50":"resnet50", "resnext50":"resnext50_32x4d", "convnext_tiny":"convnext_tiny", "deit_small":"deit_small_patch16_224", "swin_tiny":"swin_tiny_patch4_window7_224", "efficientnet_b0":"efficientnet_b0", "mobilenetv3":"mobilenetv3_large_100"}


def _head_ids(model):
    head = model.get_classifier()
    return {id(p) for p in head.parameters()}


def build_model(name, pretrained=True, num_classes=9, drop_rate=0.0, init="finetune"):
    import timm
    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(init)
    model = timm.create_model(name, pretrained=bool(pretrained and init != "scratch"), num_classes=num_classes, drop_rate=drop_rate)
    pretrained_cfg=getattr(model,"pretrained_cfg",{}) or {}
    model.weight_tag = str(pretrained_cfg.get("hf_hub_id") or f"{name}.{pretrained_cfg.get('tag','default')}") if pretrained and init != "scratch" else "scratch"
    if init == "frozen":
        freeze_backbone(model)
    return model


def freeze_backbone(model):
    head = _head_ids(model)
    if not head:
        raise ValueError("classifier has no parameters")
    for p in model.parameters():
        p.requires_grad_(id(p) in head)
    model.backbone_frozen = True


def keep_frozen_backbone_eval(model):
    if getattr(model, "backbone_frozen", False):
        model.eval()
        model.get_classifier().train()


def param_groups(model, lr_backbone, lr_head, weight_decay):
    head = _head_ids(model)
    groups = {"backbone_decay":[], "backbone_no_decay":[], "head_decay":[], "head_no_decay":[]}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        target = "head" if id(p) in head else "backbone"
        groups[target + ("_no_decay" if p.ndim <= 1 else "_decay")].append(p)
    return [{"params": params, "lr": lr_head if key.startswith("head") else lr_backbone, "weight_decay": 0.0 if key.endswith("no_decay") else weight_decay} for key, params in groups.items() if params]


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size=224):
    import torch
    from thop import profile
    m = copy.deepcopy(model).cpu().eval()
    with torch.inference_mode():
        macs, _ = profile(m, inputs=(torch.zeros(1, 3, img_size, img_size),), verbose=False)
    return float(macs / 1e9) if macs > 0 else None
