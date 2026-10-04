"""Single training entry point for backbone, recipe, and final experiments."""
from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import get_type_hints
import argparse
import json
import math
import random
import sys
import time
import numpy as np

# Keep the supplied evaluator as the single source of metric definitions.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 32
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    norm_mean: tuple[float, float, float] | None = None
    norm_std: tuple[float, float, float] | None = None
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"
    save_test_predictions: bool = False
    final_temperature: bool = False
    final_inference: str = "I00"


def run_dir(cfg):
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg, split):
    if split not in ("val", "test"):
        raise ValueError(split)
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg):
    import torch
    from model import param_groups
    return torch.optim.AdamW(param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))


def build_scheduler(optimizer, cfg, steps_per_epoch):
    import torch
    total = max(1, cfg.epochs * steps_per_epoch)
    warm = max(0, round(cfg.warmup_epochs * steps_per_epoch))
    def factor(step):
        if warm and step < warm:
            return max(1e-6, (step+1)/warm)
        return .5*(1+math.cos(math.pi * min(1, (step-warm)/max(1,total-warm))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    def __init__(self, model, decay):
        import copy
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be in (0,1)")
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.model.parameters():
            p.requires_grad_(False)
    def update(self, model):
        import torch
        with torch.no_grad():
            for a, b in zip(self.model.parameters(), model.parameters()):
                if a.is_floating_point():
                    a.lerp_(b.detach(), 1-self.decay)
                else:
                    a.copy_(b)
            for a, b in zip(self.model.buffers(), model.buffers()):
                a.copy_(b)  # BN running statistics are copied, not averaged
    def copy_to(self, model):
        model.load_state_dict(self.model.state_dict())


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg, device, ema=None):
    import torch
    from losses import mix_batch, mixed_loss
    from model import keep_frozen_backbone_eval
    model.train()
    keep_frozen_backbone_eval(model)
    total_loss, seen = 0.0, 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = None
        if cfg.mix:
            images, targets = mix_batch(images, labels, cfg.mix_alpha, cfg.mix)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=bool(cfg.amp and device.type == "cuda")):
            logits = model(images)
            loss = mixed_loss(criterion, logits, targets) if targets is not None else criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema:
            ema.update(model)
        total_loss += float(loss.detach())*len(labels)
        seen += len(labels)
    return {"train_loss": total_loss/seen, "lr": optimizer.param_groups[0]["lr"]}


def evaluate(model, loader, criterion, device):
    import torch
    model.eval()
    names, ys, zs = [], [], []
    loss_sum = 0.0
    with torch.inference_mode():
        for images, labels, batch_names in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss_sum += float(criterion(logits, labels))*len(labels)
            names.extend(batch_names)
            ys.append(labels.cpu().numpy())
            zs.append(logits.float().cpu().numpy())
    return names, np.concatenate(ys), np.concatenate(zs), loss_sum/len(names)


def plot_curves(history, path, title):
    import matplotlib.pyplot as plt
    import pandas as pd
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    axes[0].plot(h.epoch, h.train_loss, label="train loss")
    axes[0].plot(h.epoch, h.val_loss, label="val loss")
    axes[1].plot(h.epoch, h.val_macro_f1, label="val macro-F1")
    for a in axes:
        a.set_xlabel("Epoch")
        a.grid(alpha=.3)
        a.legend()
    axes[0].set_ylabel("Cross-entropy / configured loss")
    axes[1].set_ylabel("Macro-F1")
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(cfg):
    import torch
    import pandas as pd
    import timm
    from dataset import load_split, check_split, build_transforms, make_loader
    from model import build_model, count_params, count_gmacs
    from losses import build_criterion, class_weights
    from eval import compute_metrics, save_predictions
    if cfg.epochs < 1 or cfg.batch_size < 2:
        raise ValueError("epochs >= 1 and batch_size >= 2 required")
    if cfg.save_test_predictions and cfg.exp_id not in ("T00", "F01"):
        raise ValueError("test is reserved for final F01 and T00 runs")
    if cfg.final_temperature and (cfg.exp_id != "F01" or not cfg.save_test_predictions):
        raise ValueError("final_temperature requires a final F01 test run")
    if cfg.final_inference != "I00" and (cfg.exp_id != "F01" or not cfg.save_test_predictions):
        raise ValueError("final_inference requires a final F01 test run")
    rd = run_dir(cfg)
    if rd.exists() and (rd / "summary.json").exists():
        raise FileExistsError(f"run already complete: {rd}; use a new exp_id or seed")
    if cfg.save_test_predictions and pred_path(cfg, "test").exists():
        raise FileExistsError(f"test already evaluated: {pred_path(cfg, 'test')}")
    if cfg.save_test_predictions and (rd / "test_started.flag").exists():
        raise FileExistsError(f"test pass already started: {rd}; inspect that run manually")
    set_seed(cfg.seed)
    train_df, val_df, test_df = load_split(cfg.labels_dir, cfg.fold)
    split_stats = check_split(train_df, val_df, test_df, cfg.images_dir)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "split_stats.json").write_text(json.dumps(split_stats, indent=2))
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    net = build_model(cfg.backbone, cfg.init != "scratch", 9, cfg.drop_rate, cfg.init).to(device)
    data_cfg=timm.data.resolve_data_config(net.pretrained_cfg)
    cfg.norm_mean=tuple(cfg.norm_mean or data_cfg["mean"])
    cfg.norm_std=tuple(cfg.norm_std or data_cfg["std"])
    train_loader = make_loader(train_df, cfg.images_dir, build_transforms(True,cfg.img_size,cfg.aug,cfg.norm_mean,cfg.norm_std),cfg.batch_size,True,cfg.sampler,cfg.num_workers)
    val_loader = make_loader(val_df, cfg.images_dir, build_transforms(False,cfg.img_size,mean=cfg.norm_mean,std=cfg.norm_std),cfg.batch_size,False,None,cfg.num_workers)
    weights = None
    if cfg.loss == "ce_weighted":
        counts = train_df.Label.value_counts().reindex(range(9), fill_value=0).to_numpy()
        weights = class_weights(counts, cfg.class_weight_beta or 0).to(device)
    criterion = build_criterion(cfg.loss, smoothing=cfg.label_smoothing, gamma=cfg.focal_gamma, weight=weights)
    if hasattr(criterion, "to"):
        criterion = criterion.to(device)
    opt = build_optimizer(net,cfg)
    sched = build_scheduler(opt,cfg,len(train_loader))
    scaler = torch.amp.GradScaler("cuda",enabled=bool(cfg.amp and device.type=="cuda"))
    ema = EMA(net,cfg.ema_decay) if cfg.ema_decay else None
    versions = {"torch":torch.__version__, "timm":timm.__version__,"pandas":pd.__version__}
    config_path=rd/"config.json"
    recorded=json.loads(json.dumps({**asdict(cfg),"device":str(device),"weight_tag":net.weight_tag,"versions":versions}))
    if config_path.exists():
        old=json.loads(config_path.read_text())
        if any(old.get(k)!=v for k,v in recorded.items() if k!="versions"):
            raise ValueError(f"Existing run has a different configuration: {rd}")
    else:
        config_path.write_text(json.dumps(recorded,indent=2))
    hist=[]
    best=-1.0
    best_epoch=None
    start_epoch=1
    last_path=rd/"last.pt"
    if last_path.exists():
        state=torch.load(last_path,map_location=device,weights_only=False)
        net.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        if ema:
            ema.model.load_state_dict(state["ema"])
        hist=state["history"]
        best=state["best"]
        best_epoch=state["best_epoch"]
        start_epoch=state["epoch"]+1
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type=="cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"Resuming {cfg.exp_id} seed {cfg.seed} from epoch {start_epoch}",flush=True)
    for epoch in range(start_epoch,cfg.epochs+1):
        t0=time.perf_counter()
        tr=train_one_epoch(net,train_loader,criterion,opt,sched,scaler,cfg,device,ema)
        chosen=ema.model if ema else net
        names,y,z,vl=evaluate(chosen,val_loader,criterion,device)
        probs=torch.softmax(torch.from_numpy(z),dim=1).numpy()
        met=compute_metrics(y,probs.argmax(1),probs)
        row={"epoch":epoch,**tr,"val_loss":vl,"val_macro_f1":met["macro_f1"],"val_top1":met["top1"],"val_ece":met["ece"],"epoch_seconds":time.perf_counter()-t0}
        hist.append(row)
        pd.DataFrame(hist).to_csv(rd/"history.csv",index=False)
        if met["macro_f1"]>best:
            best=met["macro_f1"]
            best_epoch=epoch
            torch.save({"state_dict":chosen.state_dict(),"epoch":epoch,"val_macro_f1":best},rd/"best.pt")
        state={"epoch":epoch,"model":net.state_dict(),"optimizer":opt.state_dict(),
               "scheduler":sched.state_dict(),"scaler":scaler.state_dict(),
               "ema":ema.model.state_dict() if ema else None,"history":hist,
               "best":best,"best_epoch":best_epoch,"python_rng":random.getstate(),
               "numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),
               "cuda_rng":torch.cuda.get_rng_state_all() if device.type=="cuda" else None}
        temporary=rd/"last.pt.tmp"
        torch.save(state,temporary)
        temporary.replace(last_path)
        print(cfg.exp_id,cfg.seed,row,flush=True)
    net.load_state_dict(torch.load(rd/"best.pt",map_location=device,weights_only=True)["state_dict"])
    names,y,z,_=evaluate(net,val_loader,criterion,device)
    np.savez_compressed(rd/"val_logits.npz",filenames=np.array(names),y_true=y,logits=z)
    probs=torch.softmax(torch.from_numpy(z),dim=1).numpy()
    temperature=None
    method="I04" if cfg.final_temperature else cfg.final_inference
    if method == "I04":
        from inference import fit_temperature
        temperature=fit_temperature(z,y)
    if method != "I00":
        from inference import predict_variant
        names,y,probs,_=predict_variant(net,val_loader,device,method,temperature)
    save_predictions(pred_path(cfg,"val"),names,y,probs)
    test_metrics=None
    if cfg.save_test_predictions:
        (rd / "test_started.flag").write_text("Test evaluation was started once for this seed.\n")
        test_loader=make_loader(test_df,cfg.images_dir,build_transforms(False,cfg.img_size,mean=cfg.norm_mean,std=cfg.norm_std),cfg.batch_size,False,None,cfg.num_workers)
        from inference import predict_variant, apply_temperature
        names,y,probs,z=predict_variant(net,test_loader,device,method,temperature)
        np.savez_compressed(rd/"test_logits.npz",filenames=np.array(names),y_true=y,logits=z)
        if temperature is not None:
            save_predictions(Path(cfg.pred_dir)/f"F01uncal_seed{cfg.seed}_test.csv",names,y,apply_temperature(z,1.0))
        save_predictions(pred_path(cfg,"test"),names,y,probs)
        met=compute_metrics(y,probs.argmax(1),probs)
        test_metrics={k:float(met[k]) for k in ("top1","macro_f1","balanced_acc","ece","nll")}
    plot_curves(hist,Path(cfg.curves_dir)/f"{cfg.exp_id}_seed{cfg.seed}.png",f"{cfg.exp_id} seed {cfg.seed} {cfg.backbone}")
    try:
        gmacs=count_gmacs(net,cfg.img_size)
    except Exception as exc:
        gmacs=None
        print(f"GMAC unavailable: {exc}")
    summary={"exp_id":cfg.exp_id,"seed":cfg.seed,"best_epoch":best_epoch,"val_macro_f1":best,"val_top1":hist[best_epoch-1]["val_top1"],"params_m":count_params(net),"gmacs_thop":gmacs,"mean_epoch_seconds":float(np.mean([r["epoch_seconds"] for r in hist])),"weight_tag":net.weight_tag,"temperature":temperature,"final_inference":method,"test_metrics":test_metrics}
    (rd/"summary.json").write_text(json.dumps(summary,indent=2))
    last_path.unlink(missing_ok=True)
    return summary


def parse_overrides(pairs):
    types=get_type_hints(Config)
    out={}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"expected KEY=VALUE: {pair}")
        key,val=pair.split("=",1)
        if key not in types:
            raise ValueError(f"unknown Config key: {key}")
        if val.lower() in ("none","null"):
            if "None" not in str(types[key]):
                raise ValueError(f"{key} cannot be None")
            out[key]=None
        elif types[key] is bool:
            if val.lower() not in ("true","false","1","0"):
                raise ValueError(f"invalid bool: {val}")
            out[key]=val.lower() in ("true","1")
        elif types[key] is int:
            out[key]=int(val)
        elif key in ("norm_mean","norm_std"):
            values=tuple(float(x) for x in val.split(","))
            if len(values)!=3:
                raise ValueError(f"{key} needs three comma-separated floats")
            out[key]=values
        elif types[key] is float or "float" in str(types[key]):
            out[key]=float(val)
        else:
            out[key]=val
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--set",nargs="*",default=[])
    args=parser.parse_args()
    print(json.dumps(run(Config(**parse_overrides(args.set))),indent=2))

if __name__=="__main__":
    main()
