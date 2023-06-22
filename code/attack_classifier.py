import torch
from torch import nn
from dataset import get_dataloaders
from tqdm import tqdm
from torchvision.models import resnet34
from torchvision.transforms import Normalize
from math import sqrt
import torchvision.utils as vutils
import torchvision.transforms as T
device = torch.device("cuda:2")
model = resnet34().to(device)
model.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1)).to(device)

model.load_state_dict(torch.load("smile.pt"))

train_loader, test_loader, val_loader = get_dataloaders(filename="/share/datasets/celeba", batch_size = 64, transforms= T.Compose([T.ToTensor(),
                            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                            T.Resize((256, 256))]), selected_attr = [31], landmarks=True)
patch_side = 64
attack_neg = torch.normal(0, 0.01, size = (3, patch_side, patch_side), requires_grad = True, device = device)
attack_pos = torch.normal(0, 0.01, size = (3, patch_side, patch_side), requires_grad = True, device = device)
epsilon = 0.05
lr = 0.001

loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(3):
    tot_correct = 0
    tot_samps = 0
    for batch, (X, Y, c) in enumerate(train_loader):
        X, Y = X.to(device), Y.to(device)
        
        X_a = X.detach().clone()
        attack_c_neg = attack_neg.detach().clone().requires_grad_(True)
        attack_c_pos = attack_pos.detach().clone().requires_grad_(True)
       
        for i in range(len(X)):
            cx, cy = ((c[0][i]+c[2][i])/2 * 256/178).long(), ((c[1][i]+c[3][i])/2 * 256/218).long()
            
            X_a[i, :, (cy-patch_side//2):(cy+patch_side//2), (cx-patch_side//2):(cx+patch_side//2)] += attack_c_neg if Y[i] == 0 else attack_c_pos
        pred_a = model(X_a)
        loss = loss_fn(pred_a, Y) 
        preds = torch.where(pred_a<0.5, 0, 1)
        tot_correct += torch.sum(preds==Y)
        tot_samps += len(X)
        loss.backward()
        attack_neg = attack_c_neg + lr*torch.sign(attack_c_neg.grad)
        attack_pos = attack_c_pos + lr*torch.sign(attack_c_pos.grad)
        attack_pos = torch.where(attack_pos < -epsilon, -epsilon, attack_pos)
        attack_pos = torch.where(attack_pos > epsilon, epsilon, attack_pos)
        attack_neg = torch.where(attack_neg > epsilon, epsilon, attack_neg)
        attack_neg = torch.where(attack_neg < -epsilon, -epsilon, attack_neg)
        if not batch%100: print(f"Epoch: {epoch}, batch: {batch}, Loss: {loss}, Accuracy: {tot_correct/tot_samps}")
       
        
    loss_agg = 0
    tot_correct = 0
    tot_samps = 0
    for X, Y, c in test_loader:
        X, Y = X.to(device), Y.to(device)
       
        X_a = X.clone()
        for i in range(len(X)):
            cx, cy = ((c[0][i]+c[2][i])/2 * 256/178).long(), ((c[1][i]+c[3][i])/2 * 256/218).long()
            
            X_a[i, :, (cy-patch_side//2):(cy+patch_side//2), (cx-patch_side//2):(cx+patch_side//2)] += attack_neg if Y[i] == 0 else attack_pos
        pred_a = model(X_a)
        preds = torch.where(pred_a<0.5, 0, 1)
        tot_correct += torch.sum(preds==Y)
        tot_samps += len(X)
        loss = loss_fn(pred_a, Y) 
        loss_agg += float(loss)
    print(f"test loss: {loss_agg}, test acc: {tot_correct/tot_samps}")

    vutils.save_image(X_a*0.5+0.5, "images/smile.png")
    vutils.save_image(X*0.5 + 0.5, "images/notsmile.png")
    tr = Normalize([0.5, 0.5, 0.5], [0.5,0.5,0.5])
    print(attack_pos)
    vutils.save_image(tr(attack_pos)*0.5+0.5, "images/attackpos.png")
    vutils.save_image(tr(attack_neg)*0.5+0.5, "images/attackneg.png")
    print(attack_pos)
print(attack_neg, attack_pos, attack_neg.abs().max(), attack_pos.abs().max())
torch.save(attack_neg, "attackneg.pt")
torch.save(attack_pos, "attackpos.pt")
