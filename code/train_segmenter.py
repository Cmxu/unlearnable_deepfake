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
def init_weights(m):
    
    if type(m) == nn.Linear:
        m.weight.normal_(0, 0.01)
    if type(m) == nn.Conv2d:
        m.weight.normal_(0, 0.01)




'''
transforms= T.Compose([T.ToTensor(),
                            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                            T.Resize((256, 256))]) - aggan transforms



'''
''' transform_list = [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform_list = [T.RandomCrop((128, 128))] + transform_list
transform_list = [T.Resize(128)] + transform_list
transform_list = [T.RandomHorizontalFlip()] + transform_list 
transform_list = [T.ColorJitter(0.1, 0.1, 0.1, 0.1)] + transform_list 
transforms = T.Compose(transform_list) # HiSD transforms
'''
transforms = T.Compose([
            
            T.Resize(128),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]) # attgan
seg_transforms = T.Compose([
           
            T.Resize(128),
            T.ToTensor(),
            
        ]) 
train_loader = DataLoader(CelebASeg(transforms=transforms, seg_transforms=seg_transforms, seg_attr=[3]), shuffle = True, batch_size=64, num_workers=3, pin_memory = True)
test_loader = DataLoader(CelebASeg(split="test", transforms=transforms, seg_transforms=seg_transforms, seg_attr=[3]), shuffle = True, batch_size=64, num_workers=3, pin_memory = True)

device = torch.device("cuda:2")
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
#model.load_state_dict(torch.load("seg2.pt"))

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
print(model)
print(model(next(iter(train_loader))[0].to(device)).shape)
def label2image(pred):
    colormap = torch.tensor(VOC_COLORMAP, device=device)
    X = pred.long()
    return colormap[X, :]
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels, 
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight
with torch.no_grad(): model.apply(init_weights)
# model.transpose_conv[-1].weight.data.copy_(bilinear_kernel(num_classes, num_classes, 59))
# model.transpose_conv[-3].weight.data.copy_(bilinear_kernel(64, num_classes, 32))
#criterion = nn.CrossEntropyLoss()
opt = torch.optim.SGD(lr = 0.01, params=model.parameters())
schedule_func = lambda epoch: 1/sqrt(epoch+1)
scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda = schedule_func)
ssimloss = MSSSIM()
def criterion(inputs, targets):
    b = BCE_loss(inputs, targets) 
    i = IOU_loss(inputs, targets) 
    s = ssimloss(inputs, targets)
    return b + w1*i + w2*s, b, i, s
# coeff = 0.005
# def div_loss(Y_hat):
#     l = torch.Tensor([0]).to(device)
#     for i in Y_hat:
#         for j in Y_hat:
#             l += torch.max(-torch.linalg.norm(i - j)**2 + 5, torch.Tensor([0]).to(device))
#     return l
for ep in range(1000):
    
    for i, (X, _, Y) in enumerate(train_loader):
        
        X, Y = X.to(device), Y.to(device)
        
        Y_hat = model(X)
        Y_hat = Y_hat
        
        loss, bce, iou, ssim = criterion(Y_hat, Y.unsqueeze(1))
        # loss, bce, iou, ssim = BCE_loss(Y_hat, Y.unsqueeze(1)), 0, 0, 0

        #loss = criterion(Y_hat, Y) 
        # + div_loss(Y_hat)*coeff
       
        model.zero_grad()
        loss.backward()
        opt.step()
        Y_hat = Y_hat.permute(0,2,3,1)
        if not i%100: 
            print(f"Epoch: {ep}, batch: {i}, Loss: {float(loss), float(bce), float(iou), float(ssim)}")
            preds = torch.where(Y_hat>0.5, 1, 0).squeeze()
            
            
            vutils.save_image((l:=label2image(preds)).permute(0,3,1,2)/Y.max(), "p3.png")
            vutils.save_image((l:=label2image(Y)).permute(0,3,1,2)/Y.max(), "t2.png")
            vutils.save_image(X * 0.5 + 0.5, "x2.png")
            
    torch.save(model.state_dict(), "seghair.pt")
    # scheduler.step()
    # for X, _, Y in test_loader:
    #     X, Y = X.to(device), Y.to(device)        
    #     Y_hat = model(X)        
    #     loss = criterion(torch.flatten(Y_hat, start_dim=0, end_dim=2), Y.squeeze().flatten().long())
        
         
        
    #     tot_loss += float(loss)
        
    #     tot_batch += 1
    # print(f"Epoch: {ep}, Test Loss: {tot_loss/tot_batch}")
    
    



