"""CPU checks for math and data contracts before a GPU run."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

from losses import FocalLoss, mix_batch, class_weights
from inference import aggregate_views, fit_temperature, apply_temperature, fuse_conv_bn
from dataset import DeepWeedsDataset, make_loader, build_transforms
from benchmark import bench


class TestMath(unittest.TestCase):
    def test_focal_gamma_zero_equals_ce(self):
        torch.manual_seed(7)
        z=torch.randn(16,9)
        y=torch.randint(0,9,(16,))
        self.assertLess(abs(FocalLoss(gamma=0)(z,y).item()-F.cross_entropy(z,y).item()),1e-6)

    def test_cutmix_lambda_uses_clipped_area(self):
        x=torch.stack((torch.zeros(1,4,4),torch.ones(1,4,4)))
        y=torch.tensor([0,1])
        with patch("torch.randperm",return_value=torch.tensor([1,0])), patch("numpy.random.beta",return_value=0.5), patch("numpy.random.randint",side_effect=[0,0]):
            mixed,(_,yb,lam)=mix_batch(x,y,mode="cutmix")
        self.assertEqual(yb.tolist(),[1,0])
        changed=(mixed[0]!=x[0]).sum().item()
        self.assertAlmostEqual(lam,1-changed/16)

    def test_weights_and_temperature(self):
        w=class_weights([1,2,4])
        self.assertAlmostEqual(float(w.mean()),1,places=6)
        self.assertGreater(w[0],w[2])
        z=np.array([[6.,0.],[0.,6.],[6.,0.],[0.,6.]])
        y=np.array([0,1,1,0])
        T=fit_temperature(z,y)
        self.assertGreater(T,0)
        p=apply_temperature(z,T)
        self.assertTrue(np.allclose(p.sum(1),1))
        self.assertTrue(np.allclose(aggregate_views([z,z],"prob"),apply_temperature(z,1)))

    def test_bn_fusion_preserves_output(self):
        torch.manual_seed(3)
        m=nn.Sequential(nn.Conv2d(3,5,3,padding=1,bias=False),nn.BatchNorm2d(5),nn.ReLU()).eval()
        x=torch.randn(2,3,8,8)
        fused=fuse_conv_bn(m)
        self.assertIsInstance(fused[1],nn.Identity)
        self.assertLess((m(x)-fused(x)).abs().max().item(),1e-5)


class TestData(unittest.TestCase):
    def test_eval_loader_keeps_csv_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name,color in (("b.jpg",(255,0,0)),("a.jpg",(0,255,0))):
                Image.new("RGB",(16,16),color).save(Path(tmp)/name)
            df=pd.DataFrame({"Filename":["b.jpg","a.jpg"],"Label":[2,1]})
            loader=make_loader(df,tmp,build_transforms(False,8),1,False,num_workers=0)
            got=[name[0] for _,_,name in loader]
            self.assertEqual(got,["b.jpg","a.jpg"])

    def test_benchmark_percentiles(self):
        r=bench(lambda:None,warmup=10,iters=50)
        self.assertEqual(r["n"],50)
        self.assertLessEqual(r["p50"],r["p95"])
        self.assertLessEqual(r["p95"],r["p99"])

if __name__=="__main__": unittest.main()
