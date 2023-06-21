import torch
from torch import nn
from dataset import get_dataloaders
from tqdm import tqdm
from torchvision.models import resnet34, resnet101, resnet152
from math import sqrt
import torchvision.utils as vutils
import torchvision.transforms as T

device = torch.device("cuda:2")
model = resnet34().to(device)
model.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1)).to(device)

model.load_state_dict(torch.load("smile.pt"))

adversary = resnet152()
adversary.fc = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(),
                                      nn.Linear(2048, 2048), nn.ReLU(), 
                                      nn.Linear(2048, 3*32*32))
adversary = adversary.to(device)
def init_weights(m):
    
    if type(m) == nn.Linear:
        m.weight.normal_(0, 0.01)
    if type(m) == nn.Conv2d:
        m.weight.normal_(0, 0.01)
with torch.no_grad(): adversary.apply(init_weights)
train_loader, test_loader, val_loader = get_dataloaders(filename="/share/datasets/celeba", batch_size = 64, transforms= T.Compose([T.ToTensor(),
                            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                            T.Resize((256, 256))]), selected_attr = [31], landmarks=True)

lr = 0.0001
loss_fn = nn.BCEWithLogitsLoss()
lambda_a = 0.001
margin = 100000000
epsilon = 0.06
opt = torch.optim.Adam(params=adversary.parameters(), lr = lr)
 
for epoch in range(10):
    tot_correct = 0
    tot_samps = 0
    tot_loss = 0
    tot_norm = 0
    for batch, (X, Y, c) in enumerate(train_loader):
        
        X, Y = X.to(device), Y.to(device)
        
        X_a = X.detach().clone()
        attacks_temp = adversary(X).reshape(-1, 3,32,32)
        attacks = torch.clamp(attacks_temp, -epsilon, epsilon)
       
        for i in range(len(X)):
            cx, cy = ((c[0][i]+c[2][i])/2 * 256/178).long(), ((c[1][i]+c[3][i])/2 * 256/218).long()
            
            X_a[i, :, cy-16:cy+16, cx-16:cx+16] += attacks[i]
        pred_a = model(X_a)
        
        adv_loss = loss_fn(pred_a, Y) 
        norm_loss = torch.linalg.norm(attacks)*lambda_a
        loss = -1*adv_loss + norm_loss + margin
        preds = torch.where(pred_a<0.5, 0, 1)
        tot_correct += torch.sum(preds==Y)
        tot_loss += float(adv_loss)
        tot_norm += float(norm_loss)
        tot_samps += len(X)
        adversary.zero_grad()
        loss.backward()
        opt.step()
        


        
        if not batch%100: print(f"Epoch: {epoch}, batch: {batch}, Loss adversarial: {tot_loss/tot_samps}, Loss norm: {torch.linalg.norm(attacks)}, Accuracy: {tot_correct/tot_samps}")
    loss_agg = 0
    tot_correct = 0
    tot_samps = 0
    for X, Y, c in test_loader:
        X, Y = X.to(device), Y.to(device)
        
        X_a = X.clone()
        attacks_temp = adversary(X).reshape(-1, 3,32,32)
        attacks = torch.clamp(attacks_temp, -epsilon, epsilon)     
        for i in range(len(X)):
            cx, cy = ((c[0][i]+c[2][i])/2 * 128/178).long(), ((c[1][i]+c[3][i])/2 * 128/218).long()
            
            X_a[i, :, cy-16:cy+16, cx-16:cx+16] += attacks[i]
        pred_a = model(X_a)
        preds = torch.where(pred_a<0.5, 0, 1)
        tot_correct += torch.sum(preds==Y)
        tot_samps += len(X)
        loss = loss_fn(pred_a, Y) 
        loss_agg += float(loss)
    print(f"test loss: {loss_agg}, test acc: {tot_correct/tot_samps}")
    vutils.save_image(X_a*0.5+0.5, "images/smile.png")
    vutils.save_image(X*0.5+0.5, "images/notsmile.png")
    
    
       
    
        



