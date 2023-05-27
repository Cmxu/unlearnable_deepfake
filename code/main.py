import torch, os, torchvision
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json, argparse
from attgan import AttGAN
from stargan import StarGAN, get_stargan_data
from hisd import HiSD
from aggan import AGGAN, get_aggan_data
import torchvision.utils as vutils


def PGD(images, G, Dloss, eps, alpha, iters, verbose, init_noise, device, args, concat=False):
    nargs = []
    for a in args:
        if isinstance(a, torch.Tensor): nargs.append(a.to(device))
        else: nargs.append(a)
    args = nargs
    X_orig = images.clone()
    X_var = images.clone()
    X_orig, X_var = X_orig.to(device), X_var.to(device)
    if type(Dloss) != str:
        Dloss.to(device)

    if Dloss == 'Distorting':
        first_imgs = [G(X_orig.detach(), *args)]

    if init_noise:
        random = torch.rand_like(X_var).uniform_(-init_noise, init_noise)
        X_var += random

    pbar = tqdm(range(iters), ncols=70, desc='PGD') if verbose == 1 else range(eps)
    for __ in pbar:
        X = X_var.clone()
        X.requires_grad = True
        output = G(X, *args)
        if type(Dloss) == str:
            if Dloss == 'Nullifying':
                loss = -1 * ((output - X_orig)**2).sum()  
            elif Dloss == 'Distorting':
                loss = ((output - first_imgs[0])**2).sum()
        else:
            if concat:
                output = torch.cat([output, X], 1)
            loss = Dloss(output)

        loss = loss.mean()
        loss.backward()
        grad_value = X.grad.data

        X_var = X_var + alpha*grad_value
        X_var = torch.where(X_var < X_orig-eps, X_orig-eps, X_var)
        X_var = torch.where(X_var > X_orig+eps, X_orig+eps, X_var)
        X_var = X_var.clamp(-1, 1)

    return X_var

def LASGSA(x0, G, eps, alpha, iters, device, args, set_q = 1000, norm_est_num = 30, h=0.001):
    
  


    
    
    nargs = []
    for a in args:
        if isinstance(a, torch.Tensor): nargs.append(a.to(device))
        else: nargs.append(a)
    args = nargs
    x0 = x0.to(device)
    y0 = G(x0, *args)
    
    X_var = x0.clone()

    score = lambda x: 1-((G(x, *args) - x0)**2).sum()/((y0-x0)**2).sum()
    valt = lambda x: -((G(x, *args) - x0)**2).sum()

    def clipping(X_var):
        X_var = torch.where(X_var < x0 - eps, x0 - eps, X_var)
        X_var = torch.where(X_var > x0 + eps, x0 + eps, X_var)
        X_var = X_var.clamp(-1, 1)
        return X_var

    total_query_counter = 1
    firstpass = True
    ptq = 0

    D = x0.numel()
    pbar = tqdm(range(iters), ncols=70)

    for i in pbar:
        with torch.no_grad():

            standard = valt(X_var)
            total_query_counter += 1
            partial_F_partial_vec = lambda vec: (valt(X_var+vec*h)- standard)/h

            a = -1*(G(X_var, *args) - x0)
            total_query_counter += 1
            ahat = a/a.norm()
            partial_G_partial_dir = lambda drct: (G(X_var + drct*h, *args) - G(X_var, *args))/h

            # Self-Guided
            vector = partial_G_partial_dir(ahat)
            total_query_counter += 1
            vector = vector/vector.norm()

            ## estimate grad norm
            est_norm_sqare_accumu = 0
            for j in range(norm_est_num):
                r = torch.randn_like(X_var)
                r /= r.norm() 
                est_norm_sqare_accumu += partial_F_partial_vec(r) ** 2 * D
                total_query_counter += 1
        
            est_norm_sqare = est_norm_sqare_accumu/norm_est_num
            est_norm = torch.sqrt(est_norm_sqare).cpu().item()
            est_alpha = partial_F_partial_vec(vector)/est_norm
            total_query_counter += 1

            D_2q_2 = (D + 2*set_q - 2)
            if est_alpha**2 < 1/D_2q_2:
                lambda_star = torch.Tensor([0]).to(device)
            elif est_alpha**2 > (2*set_q - 1)/D_2q_2:
                lambda_star = 1
            else:
                denominator = (2*est_alpha**2*D*set_q - est_alpha**4*D*D_2q_2 - 1)
                lambda_star = (1-est_alpha**2)*(est_alpha**2*D_2q_2 - 1)/denominator

            if lambda_star == 1:
                g = vector
            else:

                ## Limit-Aware
                bound_clamp = lambda delta: clipping(X_var + delta) - X_var
                g = torch.zeros_like(X_var)
                for j in range(set_q):
                    upper_region = bound_clamp(torch.ones_like(X_var))
                    lower_region = bound_clamp(torch.ones_like(X_var)*(-1)) * (-1)
                    bound_region = torch.where(upper_region < lower_region, upper_region, lower_region)
                    bound_region += 1e-5

                    r = torch.randn_like(X_var)
                    r = r/r.norm()
                    r *= bound_region
                    last = r - vector.view(-1).dot(r.view(-1))*vector
                    last = last/last.norm()
                    u = torch.sqrt(lambda_star) * vector + torch.sqrt(1 - lambda_star)*last
                    g += partial_F_partial_vec(u) * u
                    total_query_counter += 1
                g = g/set_q

            prev = X_var.clone()
            X_var = X_var + alpha*g
            X_var = clipping(X_var)

            length = (alpha*g).norm()

            ## Gradient-Sliding
            for __ in range(100):
                nX_var = X_var + (X_var - prev) * (1- ((X_var - prev).norm()/length+1e-6))
                nX_var = clipping(nX_var)
                length -= (nX_var - X_var).norm()
                if length <= 0:
                    break
                prev = X_var.clone()
                X_var = nX_var.clone()
    return X_var
                


gen, loader = StarGAN(), get_stargan_data()
device = torch.device("cuda:2")
gen = gen.to(device)
X, c = next(iter(loader))
X = X.to(device)
c = c.to(device)
c[:, 4] = torch.abs(c[:, 4]-1)
new_X = LASGSA(X, gen, 0.15, 0.8, 20, device, (c,), set_q = 1000, norm_est_num = 30, h=0.001)
#new_X = PGD(X, gen, "Nullifying", 0.1, 0.01, 15, 1, True, device)
vutils.save_image(X*0.5+0.5, "images/orig.png")
vutils.save_image(new_X*0.5+0.5, "images/adv.png")
vutils.save_image(gen(new_X, c)*0.5+0.5, "images/advout.png")
vutils.save_image(gen(X, c)*0.5+0.5, "images/origout.png")




