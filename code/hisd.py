"""
"""
import math
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import copy
import torch.nn.init as init
import yaml
from torch.utils.data import DataLoader
from dataset import CelebA_HiSD
import torchvision.transforms as T
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name
def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.01)

    return init_fun

class Dis(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.tags = hyperparameters['tags']
        channels = hyperparameters['discriminators']['channels']

        self.conv = nn.Sequential(
            nn.Conv2d(hyperparameters['input_dim'], channels[0], 1, 1, 0),
            *[DownBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],
            nn.AdaptiveAvgPool2d(1),
        )

        self.fcs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(channels[-1] + 
            # ALI part which is not shown in the original submission but help disentangle the extracted style. 
            hyperparameters['style_dim'] +
            # Tag-irrelevant part. Sec.3.4
            self.tags[i]['tag_irrelevant_conditions_dim'],
            # One for translated, one for cycle. Eq.4
            len(self.tags[i]['attributes'] * 2), 1, 1, 0),
        ) for i in range(len(self.tags))])

    def forward(self, x, s, y, i):
        f = self.conv(x)
        fsy = torch.cat([f, tile_like(s, f), tile_like(y, f)], 1)
        return self.fcs[i](fsy).view(f.size(0), 2, -1)
        
    def calc_dis_loss_real(self, x, s, y, i, j):
        loss = 0
        x = x.requires_grad_()
        out = self.forward(x, s, y, i)[:, :, j]
        loss += F.relu(1 - out[:, 0]).mean()
        loss += F.relu(1 - out[:, 1]).mean()
        loss += self.compute_grad2(out[:, 0], x)
        loss += self.compute_grad2(out[:, 1], x)
        return loss
    
    def calc_dis_loss_fake_trg(self, x, s, y, i, j):
        out = self.forward(x, s, y, i)[:, :, j]
        loss = F.relu(1 + out[:, 0]).mean()
        return loss
    
    def calc_dis_loss_fake_cyc(self, x, s, y, i, j):
        out = self.forward(x, s, y, i)[:, :, j]
        loss = F.relu(1 + out[:, 1]).mean()
        return loss

    def calc_gen_loss_real(self, x, s, y, i, j):
        loss = 0
        out = self.forward(x, s, y, i)[:, :, j]
        loss += out[:, 0].mean()
        loss += out[:, 1].mean()
        return loss

    def calc_gen_loss_fake_trg(self, x, s, y, i, j):
        out = self.forward(x, s, y, i)[:, :, j]
        loss = - out[:, 0].mean()
        return loss

    def calc_gen_loss_fake_cyc(self, x, s, y, i, j):
        out = self.forward(x, s, y, i)[:, :, j]
        loss = - out[:, 1].mean()
        return loss

    def compute_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
         )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg.mean()

##################################################################################
# Generator
##################################################################################

class Gen(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.tags = hyperparameters['tags']

        self.style_dim = hyperparameters['style_dim']
        self.noise_dim = hyperparameters['noise_dim']

        channels = hyperparameters['encoder']['channels']
        self.encoder = nn.Sequential(
            nn.Conv2d(hyperparameters['input_dim'], channels[0], 1, 1, 0),
            *[DownBlockIN(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )    

        channels = hyperparameters['decoder']['channels']
        self.decoder = nn.Sequential(
            *[UpBlockIN(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],
            nn.Conv2d(channels[-1], hyperparameters['input_dim'], 1, 1, 0)
        )   

        self.extractors = Extractors(hyperparameters)

        self.translators = nn.ModuleList([Translator(hyperparameters)
            for i in range(len(self.tags))]
        )
        
        self.mappers =  nn.ModuleList([Mapper(hyperparameters, len(self.tags[i]['attributes']))
            for i in range(len(self.tags))]
        )

    def encode(self, x):
        e = self.encoder(x)
        return e

    def decode(self, e):
        x = self.decoder(e)
        return x

    def extract(self, x, i):
        return self.extractors(x, i)
    
    def map(self, z, i, j):
        return self.mappers[i](z, j)

    def translate(self, e, s, i):
        return self.translators[i](e, s)


##################################################################################
# Extractors, Translator and Mapper
##################################################################################

class Extractors(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.num_tags = len(hyperparameters['tags'])
        channels = hyperparameters['extractors']['channels']
        self.model = nn.Sequential(
            nn.Conv2d(hyperparameters['input_dim'], channels[0], 1, 1, 0),
            *[DownBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels[-1],  hyperparameters['style_dim'] * self.num_tags, 1, 1, 0),
        )

    def forward(self, x, i):
        s = self.model(x).view(x.size(0), self.num_tags, -1)
        return s[:, i]

class Translator(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        channels = hyperparameters['translators']['channels']
        self.model = nn.Sequential( 
            nn.Conv2d(hyperparameters['encoder']['channels'][-1], channels[0], 1, 1, 0),
            *[MiddleBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

        self.style_to_params = nn.Linear(hyperparameters['style_dim'], self.get_num_adain_params(self.model))
        
        self.features = nn.Sequential(
            nn.Conv2d(channels[-1], hyperparameters['decoder']['channels'][0], 1, 1, 0),
        ) 

        self.masks = nn.Sequential(
            nn.Conv2d(channels[-1], hyperparameters['decoder']['channels'][0], 1, 1, 0),
            nn.Sigmoid()
        ) 
    
    def forward(self, e, s):
        p = self.style_to_params(s)
        self.assign_adain_params(p, self.model)

        mid = self.model(e)
        f = self.features(mid)
        m = self.masks(mid) 

        return f * m + e * (1 - m)

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ in ["AdaptiveInstanceNorm2d"]:
                m.bias = adain_params[:, :m.num_features].contiguous().view(-1, m.num_features, 1)
                m.weight = adain_params[:, m.num_features:2 * m.num_features].contiguous().view(-1, m.num_features, 1) + 1
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ in ["AdaptiveInstanceNorm2d"]:
                num_adain_params += 2 * m.num_features
        return num_adain_params


class Mapper(nn.Module):
    def __init__(self, hyperparameters, num_attributes):
        super().__init__()
        channels = hyperparameters['mappers']['pre_channels']
        self.pre_model = nn.Sequential(
            nn.Linear(hyperparameters['noise_dim'], channels[0]),
            *[LinearBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

        channels = hyperparameters['mappers']['post_channels']
        self.post_models = nn.ModuleList([nn.Sequential(
            *[LinearBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],
            nn.Linear(channels[-1], hyperparameters['style_dim']), 
            ) for i in range(num_attributes)
        ])

    def forward(self, z, j):
        z = self.pre_model(z)
        return self.post_models[j](z)

##################################################################################
# Basic Blocks
##################################################################################

class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = F.avg_pool2d(self.sc(x), 2)
        out = self.conv2(self.activ(F.avg_pool2d(self.conv1(self.activ(x.clone())), 2)))
        out = residual + out
        return out / math.sqrt(2)

class DownBlockIN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)

        # use nn.InstanceNorm2d(in_dim, affine=True) if you want.
        self.in1 = InstanceNorm2d(in_dim)
        self.in2 = InstanceNorm2d(in_dim)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = F.avg_pool2d(self.sc(x), 2)
        out = self.conv2(self.activ(self.in2(F.avg_pool2d(self.conv1(self.activ(self.in1(x.clone()))), 2))))
        out = residual + out
        return out / math.sqrt(2)

class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = F.interpolate(self.sc(x), scale_factor=2, mode='nearest')
        out = self.conv2(self.activ(self.conv1(F.interpolate(self.activ(x.clone()), scale_factor=2, mode='nearest'))))
        out = residual + out
        return out / math.sqrt(2)

class UpBlockIN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.in1 = InstanceNorm2d(in_dim)
        self.in2 = InstanceNorm2d(out_dim)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = F.interpolate(self.sc(x), scale_factor=2, mode='nearest')
        out = self.conv2(self.activ(self.in2(self.conv1(F.interpolate(self.activ(self.in1(x.clone())), scale_factor=2, mode='nearest')))))
        out = residual + out
        return out / math.sqrt(2)

class MiddleBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.adain1 = AdaptiveInstanceNorm2d(in_dim)
        self.adain2 = AdaptiveInstanceNorm2d(out_dim)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = self.sc(x)
        out = self.conv2(self.activ(self.adain2(self.conv1(self.activ(self.adain1(x.clone()))))))
        out = residual + out
        return out / math.sqrt(2)

class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.linear(self.activ(x))

##################################################################################
# Basic Modules and Functions
##################################################################################

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # weight and bias are dynamically assigned
        self.bias = None
        self.weight = None

    def forward(self, x):
        assert self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        bias_in = x.mean(-1, keepdim=True)
        weight_in = x.std(-1, keepdim=True)

        out = (x - bias_in) / (weight_in + self.eps) * self.weight + self.bias
        return out.view(N, C, H, W)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class InstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # weight and bias are dynamically assigned
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        bias_in = x.mean(-1, keepdim=True)
        weight_in = x.std(-1, keepdim=True)

        out = (x - bias_in) / (weight_in + self.eps) * self.weight + self.bias
        return out.view(N, C, H, W)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

def tile_like(x, target):
    # make x is able to concat with target at dim 1.
    x = x.view(x.size(0), -1, 1, 1)
    x = x.repeat(1, 1, target.size(2), target.size(3))
    return x
def update_average(model_tgt, model_src, beta=0.99):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert (p_src is not p_tgt)
            p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

class HiSD(nn.Module):
    def __init__(self, hyperparameters):
        super(HiSD, self).__init__()
        self.gen = Gen(hyperparameters)
        self.dis = Dis(hyperparameters)

        self.noise_dim = hyperparameters['noise_dim']
        self.hyperparameters = hyperparameters

    def forward(self, args, mode):
        
        if mode == 'gen':
            return self.gen_losses(*args)
        elif mode == 'dis':
            return self.dis_losses(*args)
        else:
            pass

    def gen_losses(self, x, y, i, j, j_trg):
        batch = x.size(0)

        # non-translation path
        e = self.gen.encode(x)
        x_rec = self.gen.decode(e)

        # self-translation path
        s = self.gen.extract(x, i)
        e_slf = self.gen.translate(e, s, i)
        x_slf = self.gen.decode(e_slf)

        # cycle-translation path
        ## translate
        s_trg = self.gen.map(torch.randn(batch, self.noise_dim).cuda(), i, j_trg)
        e_trg = self.gen.translate(e, s_trg, i)
        x_trg = self.gen.decode(e_trg)
        ## cycle-back
        e_trg_rec = self.gen.encode(x_trg)
        s_trg_rec = self.gen.extract(x_trg, i) 
        e_cyc = self.gen.translate(e_trg_rec, s, i)
        x_cyc = self.gen.decode(e_cyc)

        # Added style in discriminator (ALI, Adversarially Learned Inference)
        # has not been added into the submission now, 
        # which helps disentangle the extracted style.
        # I will add this part in the next version.
        # To stable the training and avoid training crash, detaching is necessary.
        # Adding ALI will possibly make the metrics different from the paper,
        # but I do think this version would be better.
        loss_gen_adv = self.dis.calc_gen_loss_real(x, s, y, i, j) + \
                       self.dis.calc_gen_loss_fake_trg(x_trg, s_trg.detach(), y, i, j_trg) + \
                       self.dis.calc_gen_loss_fake_cyc(x_cyc, s.detach(), y, i, j) 

        loss_gen_sty = F.l1_loss(s_trg_rec, s_trg)

        loss_gen_rec = F.l1_loss(x_rec, x) + \
                       F.l1_loss(x_slf, x) + \
                       F.l1_loss(x_cyc, x)

        loss_gen_total = self.hyperparameters['adv_w'] * loss_gen_adv + \
                         self.hyperparameters['sty_w'] * loss_gen_sty + \
                         self.hyperparameters['rec_w'] * loss_gen_rec

        loss_gen_total.backward()

        return loss_gen_adv, loss_gen_sty, loss_gen_rec, \
        x_trg.detach(), x_cyc.detach(), s.detach(), s_trg.detach()

    def dis_losses(self, x, x_trg, x_cyc, s, s_trg, y, i, j, j_trg):

        loss_dis_adv = self.dis.calc_dis_loss_real(x, s, y, i, j) + \
                       self.dis.calc_dis_loss_fake_trg(x_trg, s_trg, y, i, j_trg) + \
                       self.dis.calc_dis_loss_fake_cyc(x_cyc, s, y, i, j) 
        loss_dis_adv.backward()

        return loss_dis_adv

class HiSD_Trainer(nn.Module):
    def __init__(self, hyperparameters, multi_gpus=False):
        super(HiSD_Trainer, self).__init__()
        # Initiate the networks
        self.multi_gpus = multi_gpus
        self.models = HiSD(hyperparameters)

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']

        self.dis_opt = torch.optim.Adam(self.models.dis.parameters(),
                                        lr=hyperparameters['lr_dis'], betas=(beta1, beta2),
                                        weight_decay=hyperparameters['weight_decay'])

        self.gen_opt = torch.optim.Adam([{'params': self.models.gen.encoder.parameters()},
                                         {'params': self.models.gen.translators.parameters()},
                                         {'params': self.models.gen.extractors.parameters()},
                                         {'params': self.models.gen.decoder.parameters()},
                                         # Different LR for mappers.
                                         {'params': self.models.gen.mappers.parameters(), 'lr': hyperparameters['lr_gen_mappers']},
                                        ],
                                        lr=hyperparameters['lr_gen_others'], betas=(beta1, beta2),
                                        weight_decay=hyperparameters['weight_decay'])

        
        self.apply(weights_init(hyperparameters['init']))

        # For historical average version of the generators
        self.models.gen_test = copy.deepcopy(self.models.gen)


    def update(self, x, y, i, j, j_trg):

        this_model = self.models.module if self.multi_gpus else self.models

        # gen 
        for p in this_model.dis.parameters():
            p.requires_grad = False
        for p in this_model.gen.parameters():
            p.requires_grad = True

        self.gen_opt.zero_grad()

        self.loss_gen_adv, self.loss_gen_sty, self.loss_gen_rec, \
        x_trg, x_cyc, s, s_trg = self.models((x, y, i, j, j_trg), mode='gen')

        self.loss_gen_adv = self.loss_gen_adv.mean()
        self.loss_gen_sty = self.loss_gen_sty.mean()
        self.loss_gen_rec = self.loss_gen_rec.mean()
        
        nn.utils.clip_grad_norm_(this_model.gen.parameters(), 100)
        self.gen_opt.step()

        # dis
        for p in this_model.dis.parameters():
            p.requires_grad = True
        for p in this_model.gen.parameters():
            p.requires_grad = False

        self.dis_opt.zero_grad()

        self.loss_dis_adv = self.models((x, x_trg, x_cyc, s, s_trg, y, i, j, j_trg), mode='dis')
        self.loss_dis_adv = self.loss_dis_adv.mean()

        nn.utils.clip_grad_norm_(this_model.dis.parameters(), 100)
        self.dis_opt.step()

        update_average(this_model.gen_test, this_model.gen)

        return self.loss_gen_adv.item(), \
               self.loss_gen_sty.item(), \
               self.loss_gen_rec.item(), \
               self.loss_dis_adv.item()


    def sample(self, x, x_trg, j, j_trg, i):
        this_model = self.models.module if self.multi_gpus else self.models
        if True:
            gen = this_model.gen_test
        else:
            gen = this_model.gen

        out = [x]
        with torch.no_grad():

            e = gen.encode(x)

            # Latent-guided 1 
            z = torch.randn(1, gen.noise_dim).cuda().repeat(x.size(0), 1)
            s_trg = gen.map(z, i, j_trg)
            x_trg_ = gen.decode(gen.translate(e, s_trg, i))
            out += [x_trg_]

            # Latent-guided 2
            z = torch.randn(1, gen.noise_dim).cuda().repeat(x.size(0), 1)
            s_trg = gen.map(z, i, j_trg)
            x_trg_ = gen.decode(gen.translate(e, s_trg, i))
            out += [x_trg_]

            s_trg = gen.extract(x_trg, i)
            # Reference-guided 1: use x_trg[0, 1, ..., n] as reference
            x_trg_ = gen.decode(gen.translate(e, s_trg, i))
            out += [x_trg, x_trg_]

            # Reference-guided 2: use x_trg[n, n-1, ..., 0] as reference
            x_trg_ = gen.decode(gen.translate(e, s_trg.flip([0]), i))
            out += [x_trg.flip([0]), x_trg_]

        return out

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.models.gen.load_state_dict(state_dict['gen'])
        self.models.gen_test.load_state_dict(state_dict['gen_test'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.models.dis.load_state_dict(state_dict['dis'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])

        for state in self.dis_opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        
        for state in self.gen_opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        print('Resume from iteration %d' % iterations)
        return iterations
    

    def save(self, snapshot_dir, iterations):
        this_model = self.models.module if self.multi_gpus else self.models
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'gen': this_model.gen.state_dict(), 'gen_test': this_model.gen_test.state_dict()}, gen_name)
        torch.save({'dis': this_model.dis.state_dict()}, dis_name)
        torch.save({'dis': self.dis_opt.state_dict(), 
                    'gen': self.gen_opt.state_dict()}, opt_name)

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)
def hisd():
    config = get_config("deepfakers/HiSD/outputs/celeba-hq/config.yaml")
    noise_dim = config['noise_dim']
    trainer = HiSD_Trainer(config)
    state_dict = torch.load("../models/gen_00200000.pt")
    trainer.models.gen.load_state_dict(state_dict['gen_test'])
    trainer.models.gen.cuda()
    return trainer.models.gen


def get_hisd_data(landmarks = False):
    
    transform_list = [T.ToTensor(),
                        T.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    transform_list = [T.RandomCrop((128, 128))] + transform_list
    transform_list = [T.Resize(128)] + transform_list
    transform_list = [T.RandomHorizontalFlip()] + transform_list 
    transform_list = [T.ColorJitter(0.1, 0.1, 0.1, 0.1)] + transform_list 
    transform = T.Compose(transform_list)

    attr = [5,31,15]
    loaders = [[DataLoader(
                dataset=CelebA_HiSD("/share/datasets/celeba", transform, attr[i], j, conditions = [20, 39], landmarks = landmarks),
                batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
                                    for j in range(2)] for i in range(3)]
    return loaders

class HISD(nn.Module):
    def __init__(self, device = torch.device("cuda:2")):
        super().__init__()
        config = get_config("deepfakers/HiSD/outputs/celeba-hq/config.yaml")
        trainer = HiSD_Trainer(config)
        state_dict = torch.load("../models/gen_00200000.pt")
        trainer.models.gen.load_state_dict(state_dict['gen_test'])
        self.gen = trainer.models.gen.to(device)
        self.device = device
        
    def forward(self, img, tags, i, j, j_trg):
        x = img
        y = tags
        
        
        batch = x.size(0)
        
                # non-translation path
        e = self.gen.encode(x)
        
        s_trg = self.gen.map(torch.randn(batch, 32).to(self.device), i, j_trg)
        e_trg = self.gen.translate(e, s_trg, i)
        x_trg = self.gen.decode(e_trg)
        return x_trg
    


