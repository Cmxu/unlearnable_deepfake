import torch
from torch import nn
from torch.nn import functional as F 
from dataset import CelebASeg
from tqdm import tqdm
from torchvision.models import resnet50, resnet34, resnet18
from torch.utils.data import DataLoader
from math import sqrt
import torchvision.transforms as T
import argparse
import torchvision.utils as vutils
from unet3.models.UNet_3Plus import UNet_3Plus
from unet3.loss.bceLoss import BCE_loss
from unet3.loss.msssimLoss import MSSSIM
from unet3.loss.iouLoss import IOU_loss
w1, w2 = 1, 1
transforms = T.Compose([
            
            T.Resize(128),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]) # attgan
seg_transforms = T.Compose([
           
            T.Resize(128),
            T.ToTensor(),
            
        ]) 
train_loader = DataLoader(CelebASeg(transforms=transforms, seg_transforms=seg_transforms, seg_attr=[8,15]), shuffle = True, batch_size=64, num_workers=3, pin_memory = True)
test_loader = DataLoader(CelebASeg(split="test", transforms=transforms, seg_transforms=seg_transforms, seg_attr=[8,15]), shuffle = True, batch_size=64, num_workers=3, pin_memory = True)

device = torch.device("cuda:1")
# model = resnet34(pretrained = True)
# model = nn.Sequential(*list(model.children())[:-3])
# num_classes = 3
# model.add_module('final_conv', nn.Sequential(nn.Conv2d(256, 256, kernel_size=1),nn.Sigmoid()))
# model.add_module('transpose_conv', nn.Sequential(nn.ConvTranspose2d(256, 128,
#                                     kernel_size=64, padding=16, stride=1), nn.Sigmoid(),
#                                     nn.ConvTranspose2d(128, num_classes,
#                                     kernel_size=64, padding=8, stride=1), nn.Sigmoid(),
#                                     nn.ConvTranspose2d(num_classes, num_classes,
#                                     kernel_size=59, padding=8, stride=1)))
                                    
# model = model.to(device)
num_classes = 1
model = UNet_3Plus(in_channels = 3, n_classes = num_classes).to(device)
model.load_state_dict(torch.load("seg2.pt"))
patch_side_x = 32
patch_side_y = 32
attack = torch.normal(0, 0.01, size = (3, patch_side_y, patch_side_x), requires_grad = True, device = device)
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
def label2image(pred):
    colormap = torch.tensor(VOC_COLORMAP, device=device)
    X = pred.long()
    return colormap[X, :]
epsilon = 0.05
lr = 0.001
ssimloss = MSSSIM()
def criterion(inputs, targets):
    b = BCE_loss(inputs, targets) 
    i = IOU_loss(inputs, targets) 
    s = ssimloss(inputs, targets)
    return b + w1*i + w2*s, b, i, s

for epoch in range(20):
    tot_correct = 0
    tot_samps = 0
    for batch, (X, _, Y) in enumerate(train_loader):
        X, Y = X.to(device), Y.to(device)
        
        X_a = X.detach().clone()
        attack_c = attack.detach().clone().requires_grad_(True)
        
        
        for i in range(len(X)):
            # for smiles: cx, cy = ((c[0][i]+c[2][i])/2 * 256/178).long(), ((c[1][i]+c[3][i])/2 * 256/218).long()
            cx, cy = 90, 30
            
            X_a[i, :, (cy-patch_side_y//2):(cy+patch_side_y//2), (cx-patch_side_x//2):(cx+patch_side_x//2)] += attack_c
            #X_a[i] += attack_c
        
        Y_hat = model(X_a)
        
        
        loss, bce, iou, ssim = criterion(Y_hat, Y.unsqueeze(1))
        Y_hat = Y_hat.permute(0,2,3,1)
        preds = torch.where(Y_hat>0.5, 1, 0).squeeze()
        tot_correct += torch.sum(preds.reshape(Y.shape)==Y)
        tot_samps += preds.numel()
        loss.backward()

        attack = attack_c + lr*torch.sign(attack_c.grad)
        
        
        attack = torch.where(attack > epsilon, epsilon, attack)
        attack = torch.where(attack < -epsilon, -epsilon, attack)
        if not batch%100: 
            print(f"Epoch: {epoch}, batch: {batch}, Loss: {loss}, Accuracy: {tot_correct/tot_samps}")
       
        
    

            vutils.save_image(X_a*0.5+0.5, "images/smile.png")
            vutils.save_image(X*0.5 + 0.5, "images/notsmile.png")
            tr = T.Normalize([0.5, 0.5, 0.5], [0.5,0.5,0.5])
            vutils.save_image((l:=label2image(preds)).permute(0,3,1,2)/Y.max(), "preds.png")
            vutils.save_image((l:=label2image(Y)).permute(0,3,1,2)/Y.max(), "truth.png")
        
            vutils.save_image(tr(attack)*0.5+0.5, "images/attackneg.png")
    
    torch.save(attack, "attackseg.pt")

