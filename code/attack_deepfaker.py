import torch
import torch
from torch import nn
from dataset import get_dataloaders
from tqdm import tqdm
from torchvision.models import resnet34
from torchvision.transforms import Normalize
from math import sqrt
import torchvision.utils as vutils
import torchvision.transforms as T
from aggan import get_aggan_data, AGGAN
from hisd import HISD, get_hisd_data
from attgan import AttGAN, get_attgan_data
from time import sleep
device = torch.device("cuda:2")
model = AttGAN().to(device)
test_loader = get_attgan_data()
patch_side_x = 64
patch_side_y = 16
attack_neg = torch.load("attacknegeyeatt.pt").to(device)
attack_pos = torch.load("attackposeyeatt.pt").to(device)
epsilon = 0.1
lr = 0.001

loss_fn = nn.BCEWithLogitsLoss()
print(test_loader)
for epoch in range(1):

        
    actv = 0
    base = 0
    
    tot_samps = 0
    for batch, imgs in enumerate(test_loader):
        
        X, labels, c = imgs
        X = X.to(device)
        labels = labels.to(device)
       
        X_a = X.clone()
        
        for i in range(len(X)):
            cx, cy = ((c[0][i]+c[2][i])/2 * 128/178).long(), ((c[1][i]+c[3][i])/2 * 128/218).long()
            
            X_a[i, :, (cy-patch_side_y//2):(cy+patch_side_y//2), (cx-patch_side_x//2):(cx+patch_side_x//2)] += attack_neg if labels[i, 6] ==0 else attack_pos
        labels[:, 6] = torch.abs(labels[:, 6]-1)
        pred_a = model(X_a, labels)
        pred_b = model(X, labels)
        actv += float(torch.norm(pred_a-X_a))
        base += float(torch.norm(pred_b-X))
        if not batch%100: print(f"active norm diff: {actv}, base norm diff: {base}")

        if not batch%1000: 
            vutils.save_image(pred_a*0.5+0.5, "images/actvout.png")
            vutils.save_image(pred_b*0.5+0.5, "images/baseout.png")
            vutils.save_image(X_a*0.5+0.5, "images/actvin.png")
            vutils.save_image(X*0.5 + 0.5, "images/basein.png")
    # tr = Normalize(0.5, 0.5, 0.5)
    # vutils.save_image(tr(attack_pos)*0.5+0.5, "images/attackpos.png")
    # vutils.save_image(tr(attack_neg)*0.5+0.5, "images/attackneg.png")

