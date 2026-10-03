"""Synchronized inference latency measurements (forward only)."""
from __future__ import annotations
import time
import numpy as np


def bench(fn,warmup=10,iters=100,sync=None):
    if warmup<10 or iters<50:
        raise ValueError("warmup >= 10 and iters >= 50 required")
    for _ in range(warmup):
        fn()
    times=[]
    for _ in range(iters):
        if sync: sync()
        t0=time.perf_counter()
        fn()
        if sync: sync()
        times.append((time.perf_counter()-t0)*1000)
    p50,p95,p99=np.percentile(times,[50,95,99])
    return {"p50":float(p50),"p95":float(p95),"p99":float(p99),"mean":float(np.mean(times)),"n":iters}


def latency_report(model,batch_size,img_size,dtype="fp32",device="cuda",warmup=10,iters=100):
    import torch
    if dtype not in ("fp32","amp","fp16"):
        raise ValueError(dtype)
    dev=torch.device(device)
    model=model.to(dev).eval()
    if dtype=="fp16":
        model=model.half()
    x=torch.randn(batch_size,3,img_size,img_size,device=dev,dtype=torch.float16 if dtype=="fp16" else torch.float32)
    sync=(lambda: torch.cuda.synchronize(dev)) if dev.type=="cuda" else None
    def fn():
        with torch.inference_mode(), torch.autocast(device_type=dev.type,enabled=(dtype=="amp" and dev.type=="cuda")):
            model(x)
    result=bench(fn,warmup,iters,sync)
    return {**result,"gpu":torch.cuda.get_device_name(dev) if dev.type=="cuda" else str(dev),"dtype":dtype,"batch":batch_size,"img_size":img_size,"images_per_s":batch_size/(result["p50"]/1000),"torch":torch.__version__,"scope":"forward only; synthetic input"}


def tta_latency(model,k_views,**kw):
    import torch
    if k_views<1:
        raise ValueError("k_views >= 1")
    device=kw.get("device","cuda")
    dtype=kw.get("dtype","fp32")
    batch_size=kw.get("batch_size",1)
    img_size=kw.get("img_size",224)
    warmup=kw.get("warmup",10)
    iters=kw.get("iters",100)
    dev=torch.device(device)
    model=model.to(dev).eval()
    if dtype=="fp16": model=model.half()
    x=torch.randn(batch_size,3,img_size,img_size,device=dev,dtype=torch.float16 if dtype=="fp16" else torch.float32)
    sync=(lambda:torch.cuda.synchronize(dev)) if dev.type=="cuda" else None
    def fn():
        with torch.inference_mode(), torch.autocast(device_type=dev.type,enabled=(dtype=="amp" and dev.type=="cuda")):
            for i in range(k_views):
                model(torch.flip(x,(-1,)) if i%2 else x)
    result=bench(fn,warmup,iters,sync)
    return {**result,"k_views":k_views,"gpu":torch.cuda.get_device_name(dev) if dev.type=="cuda" else str(dev),"dtype":dtype,"batch":batch_size,"img_size":img_size,"images_per_s":batch_size/(result["p50"]/1000),"scope":"forward + horizontal flip only"}
