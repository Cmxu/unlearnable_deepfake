import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from dataset import CelebA_d2d
import torchvision.transforms as T
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, use_dropout)

    def build_conv_block(self, dim, norm_layer, use_dropout):
        conv_block = [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=4, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6):
        assert(n_blocks >= 0)
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, use_dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ReflectionPad2d(1),
                      nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                kernel_size=3, stride=1),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True),
                      nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2)*4,
                                kernel_size=1, stride=1),
                      nn.PixelShuffle(2),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True),
                     ]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        output = self.model(input)
        mask = F.sigmoid(output[:, :1])
        oimg = output[:, 1:]
        mask = mask.repeat(1, 3, 1, 1)
        result = oimg*mask + input*(1-mask)

        return result, mask, oimg
    


def get_aggan_data(landmarks = False):
    
    transforms_ = T.Compose([ T.RandomHorizontalFlip(), T.Resize((256, 256)),
                T.ToTensor(),
                
                # transforms.Lambda(lambda img: img[:, 1:-1, 1:-1]),
                T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]) ])
    dataloader = DataLoader(CelebA_d2d(filename = "/share/datasets/celeba", transforms=transforms_, attr_ind=31, landmarks = landmarks), batch_size=1, shuffle=True, num_workers=3)
    return dataloader
class AGGAN(nn.Module):
    def __init__(self):
        super().__init__()
        g = Generator()
        g.load_state_dict(torch.load("../models/netG_A2B.pth"))
    
        self.gen = g
    def forward(self, img):
        return self.gen(img)[0]
