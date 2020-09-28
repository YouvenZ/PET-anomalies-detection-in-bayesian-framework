import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from utils import init_weights,init_weights_orthogonal_normal
from torchsummary import summary
import torch.nn.functional as F



base_channels = 24
num_convs_per_block = 3
default_channels_per_block = (
     base_channels,
	2* base_channels,
	 4*base_channels,
     8*base_channels,
	 8*base_channels,
	 8*base_channels,
	 8*base_channels,
	  8*base_channels)
input_channels = tuple([1])+tuple([i for i in default_channels_per_block])
print(input_channels)

channels_per_block = default_channels_per_block
down_channels_per_block = tuple([i / 2 for i in default_channels_per_block])

class NDConvGenerator(object):
    """
    generic wrapper around conv-layers to avoid 2D vs. 3D distinguishing in code.
    """
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, c_in, c_out, ks, pad=0, stride=1, norm=None, relu='relu'):
        """
        :param c_in: number of in_channels.
        :param c_out: number of out_channels.
        :param ks: kernel size.
        :param pad: pad size.
        :param stride: kernel stride.
        :param norm: string specifying type of feature map normalization. If None, no normalization is applied.
        :param relu: string specifying type of nonlinearity. If None, no nonlinearity is applied.
        :return: convolved feature_map.
        """
        if self.dim == 2:
            conv = nn.Conv2d(c_in, c_out, kernel_size=ks, padding=pad, stride=stride)
            conv.apply(init_weights_orthogonal_normal)

            if norm is not None:
                if norm == 'instance_norm':
                    norm_layer = nn.InstanceNorm2d(c_out)
                elif norm == None:
                    norm_layer = nn.BatchNorm2d(c_out)
                else:
                    raise ValueError('norm type as specified in configs is not implemented...')
                conv = nn.Sequential(conv, norm_layer)
                conv.apply(init_weights_orthogonal_normal)

        else:
            conv = nn.Conv3d(c_in, c_out, kernel_size=ks, padding=pad, stride=stride)
            conv.apply(init_weights_orthogonal_normal)
            if norm is not None:
                if norm == 'instance_norm':
                    norm_layer = nn.InstanceNorm3d(c_out)
                elif norm == None:
                    norm_layer = nn.BatchNorm3d(c_out)
                else:
                    raise ValueError('norm type as specified in configs is not implemented... {}'.format(norm))
                conv = nn.Sequential(conv, norm_layer)
                conv.apply(init_weights_orthogonal_normal)
            

        if relu is not None:
            if relu == 'relu':
                relu_layer = nn.ReLU(inplace=True)
            elif relu == 'leaky_relu':
                relu_layer = nn.LeakyReLU(inplace=True)
            else:
                raise ValueError('relu type as specified in configs is not implemented...')
            conv = nn.Sequential(conv, relu_layer)
            conv.apply(init_weights_orthogonal_normal)

        return conv



class Residual_block(nn.Module):

    def __init__(self,dim,input_channels,n_channels_in=None,n_down_channels=None,conv_per_block=3, stride=1,norm=None, relu='relu'):
        super(Residual_block,self).__init__()
        self.dim=dim
        self.n_channels_in=n_channels_in
        self.n_down_channels =n_down_channels
        self.conv_per_block=conv_per_block
        self.input_channels = input_channels
        conv = NDConvGenerator(self.dim)
        self.pre_activation_relu = nn.ReLU(inplace=False) if relu == 'relu' else nn.LeakyReLU(inplace=False)

        if n_down_channels is None:
            n_down_channels = n_channels_in

        

        self.first_conv = conv(self.input_channels, self.n_down_channels, ks=3, stride=stride,pad=1, norm=norm, relu=None)
        self.convs= nn.ModuleList([conv(self.n_down_channels, self.n_down_channels, ks=3, stride=stride, pad=1, norm=norm, relu="relu") for _ in range(conv_per_block-1)])
    


        self.conv_skip = conv(self.input_channels, self.n_channels_in, ks=1, stride=stride, norm=norm, relu=None)
        self.conv_residual = conv(self.n_down_channels, self.n_channels_in, ks=1, stride=stride, norm=norm, relu=None)
        

    def forward(self,x):
        skip = x

        residual = self.pre_activation_relu(x)
        # print("shape: residual ",residual.shape)
        Fconv = self.first_conv(residual)
        # print("shape: Fconv ",Fconv.shape)

        for i,c in enumerate(self.convs):
            Fconv = c(Fconv)
            # print("shape: Fconv ",i,Fconv.shape)



            if i < self.conv_per_block - 2:
                Fconv = F.relu(Fconv)
  


        incomming_channels = skip.shape[1]


        if incomming_channels != self.n_channels_in:
            skip = self.conv_skip(skip)
            # print("skip shape:",skip.shape)

        if self.n_down_channels != self.n_channels_in:
            residual = self.conv_residual(Fconv)
            # print("skip shape:",skip.shape)


        out = skip + residual
        
        return out


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x



class Probabilistic_block(nn.Module):
    def __init__(self,dim,channels_per_block=None):

        super(Probabilistic_block,self).__init__()

        self.dim = dim
        conv = NDConvGenerator(self.dim)
        self._latent_dims = (1,1,1,1)
        self._channels_per_block = channels_per_block
        self.num_latent_levels = len(self._latent_dims)

    
        self.probabilistic_layers=[]
   

        ##### Probabilistic blocks
        self.probabilistic_layers.append(conv(self._channels_per_block[-1], 2*self._latent_dims[0], ks=1, stride=1, norm=None, relu=None))
        for level in range(1,self.num_latent_levels):
            latent_dims = self._latent_dims[level]
            self.probabilistic_layers.append(conv(2*latent_dims, 2*latent_dims, ks=1, stride=1, norm=None, relu=None))

        self.probabilistic_layers = nn.Sequential(*self.probabilistic_layers)
        
    def forward(self,x):
        out = self.probabilistic_layers(x)
        return out



# net=Probabilistic_block(dim=2,channels_per_block=channels_per_block)
# print(net)

# inter_net = Interpolate(scale_factor=2,mode="bilinear")
# tensor=inter_net.forward(torch.ones(1,16,16,16))
# print(tensor.shape)


# net=Residual_block(dim=3,
#                     input_channels=24,
#                     n_channels_in=int(channels_per_block[1]),
#                     n_down_channels=int(down_channels_per_block[1]),
#                     conv_per_block=3, 
#                     stride=1,
#                     norm=None,
#                     relu='relu')
# print(net)
# summary(net.cuda(),input_size=[(24,128,128,128)])

