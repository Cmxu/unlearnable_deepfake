import torch
from torch import nn
from dataset import get_dataloaders
from tqdm import tqdm
from torchvision.models import resnet34
from math import sqrt
import torchvision.transforms as T
def init_weights(m):
    
    if type(m) == nn.Linear:
        m.weight.normal_(0, 0.01)
    if type(m) == nn.Conv2d:
        m.weight.normal_(0, 0.01)
    
        
selected_attr = [15] 
# 31 is smile, 15 is eyeglasses
prev_acc = 0


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
            T.CenterCrop(170),
            T.Resize(128),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]) # attgan

train_loader, test_loader, val_loader = get_dataloaders(filename="/share/datasets/celeba", batch_size = 64, transforms= transforms, selected_attr = selected_attr, )
device = torch.device("cuda:2")
model = resnet34().to(device)
model.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1)).to(device)
with torch.no_grad(): model.apply(init_weights)
criterion = nn.BCEWithLogitsLoss()
opt = torch.optim.SGD(lr = 0.05, params=model.parameters())
schedule_func = lambda epoch: 1/sqrt(epoch+1)
scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda = schedule_func)

for ep in range(15):
    
    for i, (X, Y) in enumerate(train_loader):
        
        X, Y = X.to(device), Y.to(device)
        
        Y_hat = model(X)
        
        loss = criterion(Y_hat.flatten(), Y.flatten())
        model.zero_grad()
        loss.backward()
        opt.step()
        preds = torch.where(Y_hat<0.5, 0, 1)        
        tot_correct = torch.sum(preds==Y)
        if not i%100: print(f"Epoch: {ep}, batch: {i}, Loss: {loss}, Accuracy: {tot_correct/len(X)}")
    tot_loss = 0
    tot_samp = 0
    tot_correct = 0
    tot_batch = 0
    for X, Y in test_loader:
        X, Y = X.to(device), Y.to(device)        
        Y_hat = model(X)        
        loss = criterion(Y_hat, Y)
        preds = torch.where(Y_hat<0.5, 0, 1)
         
        tot_correct += torch.sum(preds==Y)
        tot_loss += float(loss)
        tot_samp += len(X)
        tot_batch += 1
    print(f"Epoch: {ep}, Test Accuracy: {tot_correct/tot_samp}, Test Loss: {tot_loss/tot_batch}")
    
    

torch.save(model.state_dict(), "eyeglassesattgan.pt")

