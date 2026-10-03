"""Reproducible DeepWeeds experiment stages. Run from repository root.

Examples:
  python submissions/deepweeds_lab/code/experiments.py prepare
  python submissions/deepweeds_lab/code/experiments.py eda
  python submissions/deepweeds_lab/code/experiments.py sanity
  python submissions/deepweeds_lab/code/experiments.py backbones
  python submissions/deepweeds_lab/code/experiments.py training
  python submissions/deepweeds_lab/code/experiments.py inference --source T03
  python submissions/deepweeds_lab/code/experiments.py final --source T03 --method I01
  python submissions/deepweeds_lab/code/experiments.py export
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
SUB=Path(__file__).resolve().parent.parent
DATA=ROOT/"data"
LABELS=DATA/"labels"
CURVES=SUB/"curves"
PREDS=SUB/"predictions"
RUNS=SUB/"runs"
EXPECTED_MD5="b7b30f96d466fba86016aa5a26606e0f"
BASE_URL="https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels"


def base_cfg(exp_id="B01",seed=0,**changes):
    from train import Config
    return replace(Config(exp_id=exp_id,seed=seed,images_dir=str(DATA/"images"),labels_dir=str(LABELS),out_dir=str(RUNS),pred_dir=str(PREDS),curves_dir=str(CURVES)),**changes)


def _completed(exp_id,seed):
    return (RUNS/exp_id/f"seed{seed}"/"summary.json").is_file()


def _run(cfg):
    from train import run
    if _completed(cfg.exp_id,cfg.seed):
        if cfg.save_test_predictions and not (PREDS/f"{cfg.exp_id}_seed{cfg.seed}_test.csv").exists():
            raise FileNotFoundError("Completed final run lacks test predictions; inspect before continuing")
        print("Already complete:",cfg.exp_id,cfg.seed)
        return json.loads((RUNS/cfg.exp_id/f"seed{cfg.seed}"/"summary.json").read_text())
    return run(cfg)


def prepare():
    import csv
    import shutil
    LABELS.mkdir(parents=True,exist_ok=True)
    # A Kaggle Input named DeepWeeds contains images/ plus the original labels/.
    # Kaggle mounts it read-only, so link images and copy only the small CSVs.
    if Path("/kaggle/input").is_dir():
        for input_dir in Path("/kaggle/input").iterdir():
            source_labels=input_dir/"labels"
            source_images=input_dir/"images"
            names=("labels","train_subset0","val_subset0","test_subset0")
            if source_images.is_dir() and all((source_labels/f"{name}.csv").is_file() for name in names):
                with (source_labels/"labels.csv").open(newline="") as f:
                    sample=next(csv.DictReader(f))["Filename"]
                if not (source_images/sample).is_file():
                    continue
                for name in names:
                    target=LABELS/f"{name}.csv"
                    if not target.exists():
                        shutil.copy2(source_labels/f"{name}.csv",target)
                image_target=DATA/"images"
                if not image_target.exists():
                    image_target.symlink_to(source_images,target_is_directory=True)
                print("Using attached Kaggle Input:",input_dir)
                return
    for name in ("labels","train_subset0","val_subset0","test_subset0"):
        target=LABELS/f"{name}.csv"
        if not target.exists():
            subprocess.run(["curl","-fsSL","--retry","3","-o",str(target),f"{BASE_URL}/{name}.csv"],check=True)
    archive=DATA/"images.zip"
    h=hashlib.md5()
    if not archive.exists() or archive.stat().st_size<490_000_000:
        subprocess.run(["curl","-fL","--retry","3","-C","-","-o",str(archive),"https://zenodo.org/records/7939060/files/images.zip?download=1"],check=True)
    with archive.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    if h.hexdigest()!=EXPECTED_MD5:
        raise ValueError(f"images.zip MD5 mismatch: {h.hexdigest()}")
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if ".." in Path(info.filename).parts or Path(info.filename).is_absolute():
                raise ValueError(f"unsafe archive member: {info.filename}")
        z.extractall(DATA)
    if not (DATA/"images").exists():
        raise FileNotFoundError("Set DATA/images to the extracted JPG directory")
    print("Verified images.zip MD5 and extracted to",DATA)


def eda():
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image
    from dataset import CLASS_NAMES,load_split,check_split
    train,val,test=load_split(LABELS)
    stats=check_split(train,val,test,DATA/"images")
    all_labels=pd.read_csv(LABELS/"labels.csv")
    original=dict(zip(all_labels.Filename,all_labels.Label))
    fold=dict(zip(pd.concat([train,val,test]).Filename,pd.concat([train,val,test]).Label))
    if len(all_labels)!=17509 or set(original)!=set(fold):
        raise ValueError("fold 0 filenames do not match original labels.csv")
    label_disagreements=[{"Filename":name,"labels_csv":int(original[name]),"fold0":int(fold[name])} for name in original if original[name]!=fold[name]]
    if label_disagreements:
        print("Original label discrepancies (fold CSV remains authoritative):",label_disagreements)
    expected=[1125,1064,1031,1022,1062,1009,1074,1016,9106]
    observed=all_labels.Label.value_counts().reindex(range(9),fill_value=0).tolist()
    if observed!=expected:
        raise ValueError(f"class counts differ from paper: {observed}")
    (SUB/"eda.json").write_text(json.dumps({"split":stats,"all_classes":observed,"label_disagreements":label_disagreements},indent=2))
    CURVES.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(11,4))
    ax.bar(CLASS_NAMES,observed)
    ax.set_ylabel("Images")
    ax.tick_params(axis="x",rotation=35)
    fig.tight_layout();fig.savefig(CURVES/"class_distribution.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(9,3,figsize=(9,24))
    for label in range(9):
        samples=all_labels[all_labels.Label==label].head(3)
        for j,name in enumerate(samples.Filename):
            with Image.open(DATA/"images"/name) as im:
                axes[label,j].imshow(im.convert("RGB"))
            axes[label,j].set_title(f"{CLASS_NAMES[label]}: {name}",fontsize=7)
            axes[label,j].axis("off")
    fig.tight_layout();fig.savefig(CURVES/"class_samples.png",dpi=140);plt.close(fig)
    print(json.dumps(stats,indent=2))


def sanity():
    import torch
    import pandas as pd
    import matplotlib.pyplot as plt
    from dataset import load_split,build_transforms,make_loader
    from model import build_model
    from losses import FocalLoss,mix_batch
    train,_,_=load_split(LABELS)
    CURVES.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)
    x,y,_=next(iter(make_loader(train.head(8),DATA/"images",build_transforms(True),8,True,num_workers=0)))
    x,y=x.to(device),y.to(device)
    m=build_model("resnet50",True,9,init="frozen").to(device)
    m.eval();m.get_classifier().train()
    opt=torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=.01)
    with torch.inference_mode():
        initial_loss=float(torch.nn.functional.cross_entropy(m(x),y))
    history=[]
    for i in range(150):
        opt.zero_grad()
        logits=m(x)
        loss=torch.nn.functional.cross_entropy(logits,y)
        loss.backward();opt.step()
        history.append(float(loss))
    test_logits=torch.randn(16,9,device=device)
    test_y=torch.randint(0,9,(16,),device=device)
    focal_error=float((FocalLoss(0)(test_logits,test_y)-torch.nn.functional.cross_entropy(test_logits,test_y)).abs())
    if focal_error>=1e-6:
        raise AssertionError(f"focal gamma=0 differs from CE: {focal_error}")
    mixed,(_,_,lam)=mix_batch(x,y,1.0,"cutmix")
    print("uniform head CE theoretical",float(np.log(9)),"initial head loss",initial_loss,"overfit final loss",history[-1],"focal gamma=0 error",focal_error,"CutMix lambda",lam)
    fig,ax=plt.subplots();ax.plot(history);ax.set(xlabel="Update",ylabel="CE",title="Overfit one batch");fig.savefig(CURVES/"sanity_overfit.png",dpi=160);plt.close(fig)
    # De-normalize a mixed image to verify spatial placement visually.
    mean=torch.tensor((.485,.456,.406),device=device)[:,None,None]
    std=torch.tensor((.229,.224,.225),device=device)[:,None,None]
    pic=(mixed[0]*std+mean).clamp(0,1).permute(1,2,0).cpu().numpy()
    plt.imsave(CURVES/"sanity_cutmix.png",pic)
    fig,axes=plt.subplots(2,4,figsize=(10,5))
    for image,label,ax in zip(x,y,axes.flat):
        ax.imshow((image*std+mean).clamp(0,1).permute(1,2,0).cpu().numpy())
        ax.set_title(f"Label {int(label)}")
        ax.axis("off")
    fig.tight_layout();fig.savefig(CURVES/"sanity_augmented.png",dpi=160);plt.close(fig)


BACKBONES=[("B01","resnet50"),("B02","resnext50_32x4d"),("B03","convnext_tiny"),("B04","deit_small_patch16_224"),("B05","efficientnet_b0")]
TRAINING=[("T01",{"init":"scratch"}),("T02",{"init":"frozen"}),("T03",{"aug":"color"}),("T04",{"aug":"randaug"}),("T05",{"loss":"ls","label_smoothing":.1}),("T06",{"loss":"focal","focal_gamma":2.0}),("T07",{"loss":"ce_weighted"}),("T08",{"sampler":"balanced"}),("T09",{"mix":"mixup"}),("T10",{"mix":"cutmix"}),("T11",{"ema_decay":.999})]


def backbones():
    from benchmark import latency_report
    recorded={}
    file=SUB/"backbone_latency.json"
    if file.exists(): recorded=json.loads(file.read_text())
    for exp,name in BACKBONES:
        _run(base_cfg(exp,backbone=name))
        if exp not in recorded:
            model,_,device=_load_source(exp)
            recorded[exp]=latency_report(model,1,224,device=str(device))
            file.write_text(json.dumps(recorded,indent=2))


def training():
    if not _completed("B01",0):
        raise FileNotFoundError("Run backbones first; B01 is the T00 recipe on ResNet50")
    for exp,changes in TRAINING:
        _run(base_cfg(exp,**changes))
    def score(exp):
        return json.loads((RUNS/exp/"seed0"/"summary.json").read_text())["val_macro_f1"]
    aug_exp=max(("T03","T04"),key=score)
    loss_exp=max(("T05","T06","T07"),key=score)
    variants=dict(TRAINING)
    combined={**variants[aug_exp],**variants[loss_exp]}
    (SUB/"T12_recipe.json").write_text(json.dumps({"augmentation_source":aug_exp,"loss_source":loss_exp,"changes":combined},indent=2))
    _run(base_cfg("T12",**combined))


def _load_source(exp):
    import torch
    from train import Config
    from model import build_model
    rd=RUNS/exp/"seed0"
    cfg=Config(**{k:v for k,v in json.loads((rd/"config.json").read_text()).items() if k in Config.__dataclass_fields__})
    device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model=build_model(cfg.backbone,False,9,cfg.drop_rate,init="scratch").to(device)
    model.load_state_dict(torch.load(rd/"best.pt",map_location=device,weights_only=True)["state_dict"])
    model.eval()
    return model,cfg,device


def inference(source):
    import torch
    from dataset import load_split,build_transforms,make_loader
    from inference import predict_variant,fit_temperature,views_multicrop,view_hflip,views_multiscale,fuse_conv_bn
    from benchmark import bench
    from eval import compute_metrics,save_predictions
    model,cfg,device=_load_source(source)
    _,val,_=load_split(LABELS)
    loader=make_loader(val,DATA/"images",build_transforms(False,cfg.img_size,mean=cfg.norm_mean,std=cfg.norm_std),cfg.batch_size,False,num_workers=cfg.num_workers)
    archive=np.load(RUNS/source/"seed0"/"val_logits.npz")
    T=fit_temperature(archive["logits"],archive["y_true"])
    methods=["I00","I01","I02","I03","I04","I05"]
    if cfg.backbone.startswith(("deit","vit","swin")):
        methods=["I00","I01","I04","I05"]
    records=[]
    x=torch.randn(1,3,cfg.img_size,cfg.img_size,device=device)
    for method in methods:
        names,y,p,_=predict_variant(model,loader,device,method,T if method=="I04" else None)
        save_predictions(PREDS/f"{method}_seed0_val.csv",names,y,p)
        m=compute_metrics(y,p.argmax(1),p)
        bench_model=fuse_conv_bn(model) if method=="I05" else model
        def forward():
            with torch.inference_mode():
                z=bench_model(x)
                if method=="I01":
                    bench_model(view_hflip(x))
                elif method=="I02":
                    for v in views_multicrop(x,192): bench_model(v)
                elif method=="I03":
                    bench_model(views_multiscale(x,[256])[0])
                elif method=="I04":
                    torch.softmax(z/T,dim=1)
        sync=(lambda:torch.cuda.synchronize(device)) if device.type=="cuda" else None
        latency=bench(forward,10,100,sync)
        record={"exp_id":method,"source":source,"method":method,"k":6 if method=="I02" else 2 if method in ("I01","I03") else 1,"val_macro_f1":m["macro_f1"],"val_top1":m["top1"],"val_ece":m["ece"],"temperature":T if method=="I04" else None,"device":str(device),"batch":1,"dtype":"fp32","img_size":cfg.img_size,"latency":latency}
        records.append(record)
        print(record,flush=True)
    (SUB/"inference_results.json").write_text(json.dumps(records,indent=2))
    print("Select --method for final using validation macro-F1, ECE, and p95 latency only.")


def final(source,method):
    from train import Config
    if method not in ("I00","I01","I02","I03","I04","I05"):
        raise ValueError(method)
    if method in ("I02","I03") and source=="B04":
        raise ValueError("multi-crop/scale requires a spatially flexible CNN")
    selection_file=SUB/"inference_results.json"
    if not selection_file.exists() or not any(r["source"]==source and r["method"]==method for r in json.loads(selection_file.read_text())):
        raise ValueError("Run inference on the selected source and choose a method measured on val first")
    rd=RUNS/source/"seed0"
    cfg=Config(**{k:v for k,v in json.loads((rd/"config.json").read_text()).items() if k in Config.__dataclass_fields__})
    # Selection is explicit and must be based only on inference_results.json (val).
    for seed in (0,1,2):
        _run(base_cfg("T00",seed=seed,save_test_predictions=True))
        _run(replace(cfg,exp_id="F01",seed=seed,save_test_predictions=True,final_inference=method,final_temperature=False))


def export():
    import pandas as pd
    from eval import load_group,read_pred,compute_metrics,CLASS_NAMES
    rows={"Backbones":[],"Training":[],"Inference":[],"Final":[],"PerClass":[],"Latency":[],"Summary":[]}
    b01=None
    backbone_latency=json.loads((SUB/"backbone_latency.json").read_text()) if (SUB/"backbone_latency.json").exists() else {}
    path=RUNS/"B01"/"seed0"/"summary.json"
    if path.exists(): b01=json.loads(path.read_text())["val_macro_f1"]
    for exp,_ in BACKBONES:
        rd=RUNS/exp/"seed0"; p=rd/"summary.json"
        if not p.exists(): continue
        s=json.loads(p.read_text()); c=json.loads((rd/"config.json").read_text())
        lat=backbone_latency.get(exp,{})
        rows["Backbones"].append({"exp_id":exp,"backbone":c["backbone"],"weight_tag":s["weight_tag"],"params_M":s["params_m"],"GMAC_thop":s["gmacs_thop"],"img_size":c["img_size"],"epochs":c["epochs"],"seed":0,"macro_F1_val":s["val_macro_f1"],"top1_val":s["val_top1"],"seconds_per_epoch":s["mean_epoch_seconds"],"latency_p95_ms_batch1":lat.get("p95")})
        rows["Summary"].append({"exp_id":exp,"macro_F1_val":s["val_macro_f1"],"top1_val":s["val_top1"],"p95_ms_batch1":lat.get("p95")})
        if lat:
            rows["Latency"].append({"configuration":exp,"device":lat["gpu"],"dtype":lat["dtype"],"batch":1,"fused_BN":False,"p50_ms":lat["p50"],"p95_ms":lat["p95"],"p99_ms":lat["p99"],"images_per_s":lat["images_per_s"]})
    combo_file=SUB/"T12_recipe.json"
    all_training=TRAINING+([("T12",json.loads(combo_file.read_text())["changes"])] if combo_file.exists() else [])
    for exp,change in all_training:
        rd=RUNS/exp/"seed0";p=rd/"summary.json"
        if not p.exists():continue
        s=json.loads(p.read_text());c=json.loads((rd/"config.json").read_text())
        vp=read_pred(str(PREDS/f"{exp}_seed0_val.csv"))
        vm=compute_metrics(vp.y_true,vp.y_pred,vp.probs)
        axis="A:init" if exp in ("T01","T02") else "B:aug" if exp in ("T03","T04") else "C:loss" if exp in ("T05","T06","T07") else "D:sampler" if exp=="T08" else "B:mix" if exp in ("T09","T10") else "G:EMA" if exp=="T11" else "Combination"
        rows["Training"].append({"exp_id":exp,"backbone":c["backbone"],"axis":axis,"changed_from_T00":str(change),"seed":0,"macro_F1_val":s["val_macro_f1"],"top1_val":s["val_top1"],"delta_F1_vs_B01_T00":s["val_macro_f1"]-b01 if b01 is not None else None,"Chinee_Apple_F1_val":float(vm["f1"][0]),"Snake_Weed_F1_val":float(vm["f1"][7])})
        rows["Summary"].append({"exp_id":exp,"macro_F1_val":s["val_macro_f1"],"top1_val":s["val_top1"],"p95_ms_batch1":None})
    if (SUB/"inference_results.json").exists():
        infer_records=json.loads((SUB/"inference_results.json").read_text())
        base_p50=next((r["latency"]["p50"] for r in infer_records if r["method"]=="I00"),None)
        for r in infer_records:
            lat=r["latency"]
            rows["Inference"].append({"exp_id":r["exp_id"],"method":r["method"],"source":r["source"],"K":r["k"],"macro_F1_val":r["val_macro_f1"],"top1_val":r["val_top1"],"ECE_val":r["val_ece"],"p50_ms":lat["p50"],"p95_ms":lat["p95"],"p99_ms":lat["p99"],"images_per_s":1000/lat["p50"],"cost_vs_I00":lat["p50"]/base_p50 if base_p50 else None})
            rows["Latency"].append({"configuration":r["exp_id"],"device":r["device"],"dtype":r["dtype"],"batch":1,"fused_BN":r["method"]=="I05","p50_ms":lat["p50"],"p95_ms":lat["p95"],"p99_ms":lat["p99"],"images_per_s":1000/lat["p50"]})
            rows["Summary"].append({"exp_id":r["exp_id"],"macro_F1_val":r["val_macro_f1"],"top1_val":r["val_top1"],"p95_ms_batch1":lat["p95"]})
    for exp in ("T00","F01"):
        files=sorted(PREDS.glob(f"{exp}_seed*_test.csv"))
        if not files:continue
        group=load_group(str(PREDS/f"{exp}_seed*_test.csv"),str(LABELS/"test_subset0.csv"))
        val_f1=[]
        for pred,metric in zip(group.preds,group.metrics):
            vp=read_pred(str(PREDS/f"{exp}_seed{pred.seed}_val.csv"))
            vm=compute_metrics(vp.y_true,vp.y_pred,vp.probs)
            val_f1.append(vm["macro_f1"])
            config=json.loads((RUNS/exp/f"seed{pred.seed}"/"config.json").read_text())
            recipe=f"{config['backbone']} + {config['aug']} + {config['loss']} + {config.get('final_inference','I00')}"
            rows["Final"].append({"exp_id":exp,"recipe":recipe,"seed":pred.seed,"macro_F1_val":vm["macro_f1"],"macro_F1_test":metric["macro_f1"],"top1_test":metric["top1"],"ECE_test":metric["ece"]})
        rows["Final"].append({"exp_id":exp,"seed":"mean ± std","macro_F1_val":f"{np.mean(val_f1):.4f} ± {np.std(val_f1,ddof=1):.4f}","macro_F1_test":str(group.summary["macro_f1"]),"top1_test":str(group.summary["top1"]),"ECE_test":str(group.summary["ece"])})
        for i,name in enumerate(CLASS_NAMES):
            rows["PerClass"].append({"exp_id":exp,"class":name,"n_test":int(group.metrics[0]["support"][i]),"precision_mean":float(group.summary["precision"][0][i]),"recall_mean":float(group.summary["recall"][0][i]),"F1_mean":float(group.summary["f1"][0][i]),"F1_std":float(group.summary["f1"][1][i])})
    rows["Summary"].sort(key=lambda r:r["macro_F1_val"],reverse=True)
    rows["Summary"]=rows["Summary"][:10]
    with pd.ExcelWriter(SUB/"results.xlsx",engine="openpyxl") as writer:
        for sheet,records in rows.items():
            pd.DataFrame(records).to_excel(writer,sheet_name=sheet,index=False)
            ws=writer.sheets[sheet];ws.freeze_panes="A2";ws.auto_filter.ref=ws.dimensions
            for col in ws.columns:
                col_letter=col[0].column_letter
                ws.column_dimensions[col_letter].width=min(50,max(13,max(len(str(cell.value or "")) for cell in col)+2))
    print("Wrote",SUB/"results.xlsx")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("prepare","eda","sanity","backbones","training","inference","final","export"))
    p.add_argument("--source",default="T03",help="val-selected training experiment for inference/final")
    p.add_argument("--method",default="I04",help="val-selected inference method for final")
    args=p.parse_args()
    if args.stage=="prepare":prepare()
    elif args.stage=="eda":eda()
    elif args.stage=="sanity":sanity()
    elif args.stage=="backbones":backbones()
    elif args.stage=="training":training()
    elif args.stage=="inference":inference(args.source)
    elif args.stage=="final":final(args.source,args.method)
    else:export()

if __name__=="__main__":main()
