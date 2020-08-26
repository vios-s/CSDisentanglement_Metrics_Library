import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import torch.nn.init as init

#Content
#Create content trainer
class Cont_Trainer(nn.Module):
    def __init__(self):
        super(Cont_Trainer, self).__init__()
        lr = 0.0001
        self.AE = AutoEncoder()
        self.print_network(self.AE, 'AutoEncoder')
        #Setup the optimizers
        beta1 = 0.5
        beta2 = 0.999
        AE_params = list(self.AE.parameters())
        self.AE_opt = torch.optim.Adam([p for p in AE_params if p.requires_grad], lr=lr, betas=(beta1, beta2))
        self.AE.apply(self.initialize_weights)

    def l2_criterion(self, inp, target):
        return torch.mean(torch.abs(inp - target) ** 2)

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
            init.xavier_uniform_(m.weight.data)

    def forward(self, train_input):
        self.eval()
        reconstruction = self.AE(train_input)
        self.train()
        return reconstruction

    def test(self, train_input, ori_images):
        self.eval()
        reconstruction = self.AE(train_input)
        mse = self.l2_criterion(reconstruction, ori_images)
        self.train()
        return mse

    def AE_update(self, train_input, ori_images):
        self.AE.zero_grad()
        reconstruction = self.AE(train_input)
        self.loss = self.l2_criterion(reconstruction, ori_images)
        self.loss.backward()
        self.AE_opt.step()
        return self.loss

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.model = []
        # 128x64x64 -> 64x32x32 -> 64x16x16
        # 32x32x32 -> 16x64x64 -> 8x128x128 -> 3x128x128
        dim = 128
        input_dim = 128
        output_dim = 3
        #Input layer
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3)]

        #Downsampling blocks
        for i in range(2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1)]

        #Upsampling blocks
        for i in range(3):
            self.model += [UpConv2dBlock(dim, dim // 2, 4, 2, 1)]
            dim //= 2

        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, relu=False)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, relu=True):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True

        #Initialize padding
        self.pad = nn.ZeroPad2d(padding)

        #Initialize normalization
        norm_dim = output_dim
        self.norm = nn.InstanceNorm2d(norm_dim, affine=True)

        #Initialize activation
        #Initialize activation
        if relu:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = nn.Tanh()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        x = self.norm(x)
        x = self.activation(x)
        return x

class UpConv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, relu=True):
        super(UpConv2dBlock, self).__init__()
        self.use_bias = True

        #Initialize normalization
        norm_dim = output_dim
        self.norm = nn.InstanceNorm2d(norm_dim, affine=True)

        #Initialize activation
        if relu:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = nn.Tanh()
        self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


#Style
#Create style trainer
class Sty_Trainer(nn.Module):
    def __init__(self):
        super(Sty_Trainer, self).__init__()
        lr = 0.0001
        self.DE = Decoder()
        self.print_network(self.DE, 'Decoder')
        #Setup the optimizers
        beta1 = 0.5
        beta2 = 0.999
        DE_params = list(self.DE.parameters())
        self.DE_opt = torch.optim.Adam([p for p in DE_params if p.requires_grad], lr=lr, betas=(beta1, beta2))
        self.DE.apply(self.initialize_weights)

    def l2_criterion(self, inp, target):
        return torch.mean(torch.abs(inp - target) ** 2)

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
            init.xavier_uniform_(m.weight.data)

    def forward(self, train_input):
        self.eval()
        reconstruction = self.DE(train_input)
        self.train()
        return reconstruction

    def test(self, train_input, ori_images):
        self.eval()
        reconstruction = self.DE(train_input)
        mse = self.l2_criterion(reconstruction, ori_images)
        self.train()
        return mse

    def DE_update(self, train_input, ori_images):
        self.DE.zero_grad()
        reconstruction = self.DE(train_input)
        self.loss = self.l2_criterion(reconstruction, ori_images)
        self.loss.backward()
        self.DE_opt.step()
        return self.loss

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = []
        self.liner = []
        self.upsample = "nearest"
        self.h = 128
        self.w = 128
        dim = 256
        input_dim = 8
        output_dim = 3
        #Linear layer layer
        self.liner = nn.Sequential(
            nn.Linear(8, 256),
            nn.Linear(256, 256*4*4),
            nn.Linear(256*4*4, 256*8*8),
        )
        #Upsampling blocks
        for i in range(4):
            self.model += [UpConv2dBlock(dim, dim // 2, 4, 2, 1)]
            dim //= 2

        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, relu=False)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        liner_feature = self.liner(x)
        liner_feature = liner_feature.view(liner_feature.size(0), 256, 8, 8)
        return self.model(liner_feature)
