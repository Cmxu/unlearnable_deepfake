import torch
from torch import nn
from dataset import get_dataloaders
from tqdm import tqdm
from torchvision.models import resnet34
from math import sqrt
import torchvision.utils as vutils

device = torch.device("cuda:2")
model = resnet34().to(device)
model.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1)).to(device)

model.load_state_dict(torch.load("smile.pt"))

train_loader, test_loader, val_loader = get_dataloaders(filename="/share/datasets/celeba", batch_size = 64, selected_attr = [31], landmarks=True)
 
attack = torch.normal(0, 0.01, size = (3, 32, 32), requires_grad = True, device = device)
epsilon = 0.05
lr = 0.01
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(10):
    tot_correct = 0
    tot_samps = 0
    for i, (X, Y, c) in enumerate(train_loader):
        X, Y = X.to(device), Y.to(device)
        cx, cy = 63, 89
        X_a = X.detach().clone()
        attack_c = attack.detach().clone().requires_grad_(True)
       
        X_a[:, :, cy-16:cy+16, cx-16:cx+16] += attack_c
        pred_a = model(X_a)
        loss = loss_fn(pred_a, Y) 
        preds = torch.where(pred_a<0.5, 0, 1)
        tot_correct += torch.sum(preds==Y)
        tot_samps += len(X)
        loss.backward()
        attack = attack_c + lr*torch.sign(attack_c.grad)
        attack = torch.where(attack < -epsilon, -epsilon, attack)
        attack = torch.where(attack > epsilon, epsilon, attack)
        if not i%100: print(f"Epoch: {epoch}, batch: {i}, Loss: {loss}, Accuracy: {tot_correct/tot_samps}")
    loss_agg = 0
    tot_correct = 0
    tot_samps = 0
    for X, Y, c in test_loader:
        X, Y = X.to(device), Y.to(device)
        cx, cy = 63, 89
        X_a = X.clone()
        X_a[:, :, cy-16:cy+16, cx-16:cx+16] += attack
        pred_a = model(X_a)
        preds = torch.where(pred_a<0.5, 0, 1)
        tot_correct += torch.sum(preds==Y)
        tot_samps += len(X)
        loss = loss_fn(pred_a, Y) 
        loss_agg += float(loss)
    print(f"test loss: {loss_agg}, test acc: {tot_correct/tot_samps}")

    vutils.save_image(X_a*0.5+0.5, "images/smile.png")
    vutils.save_image(attack*20, "images/attack.png")
    
        



