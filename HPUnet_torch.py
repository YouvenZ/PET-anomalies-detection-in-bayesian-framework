import torch
import torch.nn as nn
from torch.distributions import Normal, Independent, kl
from torch.autograd import Variable
import numpy as np
from utils import init_weights,init_weights_orthogonal_normal,l2_regularisation
from torchsummary import summary
from unet_utils import *
from exp_utils import save_ckp,load_ckp
from data import *
import torch.nn.functional as F
from StitchingDeTr import *
import time
from model_utils import *
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torchvision.transforms as T




# class Joiner(nn.Sequential):
#     def __init__(self, backbone, position_embedding):
#         #super().__init__(backbone, position_embedding)
#         super(Joiner,self).__init__()
#         #super(Hierarchical_Core,self).__init__()

#     def forward(self, tensor_list: NestedTensor):
#         xs = self[0](tensor_list)
#         print(xs.shape,"xs shape")

#         out: List[NestedTensor] = []
#         pos = []
#         for name, x in xs.items():
#             out.append(x)
#             # position encoding
#             pos.append(self[1](x).to(x.tensors.dtype))

#         return out, pos

# ########################################################
# #                                                      #
# #              HierarchicalCore model                  #
# #                                                      #
# ########################################################

# args={'d_model':192,
#         'dropout':0.1,
#         'nhead':8,
#         'dim_feedforward':2048,
#         'num_encoder_layers':6,
#         'num_decoder_layers':6,
#         'normalize_before':False,
#         'return_intermediate_dec':True,}

# transformer=build_transformer(args)

class Hierarchical_Core(nn.Module):
    """
    A block of n resudual layers where each layer is followed by a non-linear activation function
    Between each block we add a pooling operation.
    """
    def __init__(self,dim,input_channels,channels_per_block=None,convs_per_block=None,blocks_per_level=None,down_channels_per_block=None,Posterior=False):

        super(Hierarchical_Core,self).__init__()

        self.dim = dim
        conv = NDConvGenerator(self.dim)
        self._latent_dims = (1,1,1,1)

        number_of_classes = 2
        self.input_channels=input_channels

        if Posterior:
            self.input_channels[0] = number_of_classes+1
        else:
            self.input_channels[0] = 1
        

        self._channels_per_block = channels_per_block
        self.num_levels = len(self._channels_per_block)
        self.num_latent_levels = len(self._latent_dims)
        self._convs_per_block = convs_per_block
        self._blocks_per_level = blocks_per_level
        if down_channels_per_block is None:
            self._down_channels_per_block = channels_per_block
        else:
            self._down_channels_per_block = down_channels_per_block
        self.residual_block = Residual_block
        self.probabilistic_block = Probabilistic_block(dim=self.dim,channels_per_block=self._channels_per_block)
        self.Pool_layers = nn.ModuleList()
        self.list_test=[]
        self.decoder_layers=nn.ModuleList()
        self.res_layers=nn.ModuleList()
        self.next_decoder_layers=nn.ModuleList()
        self.probabilistic_layers=[]
        if self.dim == 2:
            self.interpolate = Interpolate(2,mode="bilinear")
        else:
            self.interpolate = Interpolate(2,mode="trilinear")

        ####### decoder layers
        for probabilistic_level in range(self.num_latent_levels):
            d_layers = nn.ModuleList([Residual_block(dim=int(self.dim),
                            input_channels=int(2*self.input_channels[-1]+self._latent_dims[0]),
                            n_channels_in=int(self._channels_per_block[::-1][1]),
                            n_down_channels=int(self._down_channels_per_block[::-1][1]),
                            conv_per_block=3, 
                            stride=1,
                            norm=None,
                            relu='relu')]+[Residual_block(dim=int(self.dim),
                            input_channels=int(self.input_channels[-1]),
                            n_channels_in=int(self._channels_per_block[::-1][probabilistic_level + 1]),
                            n_down_channels=int(self._down_channels_per_block[::-1][probabilistic_level + 1]),
                            conv_per_block=3, 
                            stride=1,
                            norm=None,
                            relu='relu') for _ in range(self._blocks_per_level-1)])


            self.decoder_layers.append(nn.Sequential(*d_layers))    




        for level in range(self.num_levels):


            residual_layers = nn.ModuleList([Residual_block(dim=int(self.dim),
                    input_channels=int(self.input_channels[level]),
                    n_channels_in=int(self._channels_per_block[level]),
                    n_down_channels=int(self._down_channels_per_block[level]),
                    conv_per_block=3, 
                    stride=1,
                    norm=None,
                    relu='relu')]+[Residual_block(dim=int(self.dim),
                    input_channels=int(self.input_channels[level+1]),
                    n_channels_in=int(self._channels_per_block[level]),
                    n_down_channels=int(self._down_channels_per_block[level]),
                    conv_per_block=3, 
                    stride=1,
                    norm=None,
                    relu='relu') for _ in range(self._blocks_per_level-1)])


            self.res_layers.append(nn.Sequential(*residual_layers))    


            if level!= self.num_levels - 1:
                if self.dim == 2:
                    self.Pool_layers.append(nn.AvgPool2d(1, stride=2,ceil_mode=False))
                    #self.layers.append(nn.ModuleList([down,nn.AvgPool2d(1, stride=2,ceil_mode=False)]))
                if self.dim == 3:
                    self.Pool_layers.append(nn.AvgPool3d(1, stride=2,ceil_mode=False))
                    # self.layers.append(nn.ModuleList([down,nn.AvgPool3d(1, stride=2,ceil_mode=False)]))
        num_latents = len(self._latent_dims)    
        start_level = num_latents + 1
        num_levels = len(self._channels_per_block)

        for level in range(start_level,num_levels,1):
            #print(level)
            next_d_layers = nn.ModuleList([Residual_block(dim=int(self.dim),
                            input_channels=int(self.input_channels[::-1][level-1]+self.input_channels[::-1][level]),
                            n_channels_in=int(self._channels_per_block[::-1][level]),
                            n_down_channels=int(self._down_channels_per_block[::-1][level]),
                            conv_per_block=3, 
                            stride=1,
                            norm=None,
                            relu='relu')]+[Residual_block(dim=int(self.dim),
                            input_channels=int(self._channels_per_block[::-1][level]),
                            n_channels_in=int(self._channels_per_block[::-1][level]),
                            n_down_channels=int(self._down_channels_per_block[::-1][level]),
                            conv_per_block=3, 
                            stride=1,
                            norm=None,
                            relu='relu') for _ in range(self._blocks_per_level-1)])


            self.next_decoder_layers.append(nn.Sequential(*next_d_layers)) 

        
        self.decoder_layers = nn.Sequential(*self.decoder_layers)
        self.next_decoder_layers = nn.Sequential(*self.next_decoder_layers)

        #print("++++++++++++++++++++++")

        #print(self.decoder_layers)
        # print("++++++++++++++++++++++")
        # print(self.probabilistic_layers)
        # print("++++++++++++++++++++++")

        # print(self.next_decoder_layers)
        # print("++++++++++++++++++++++")

                
        # print(self.decoder_layers,len(self.decoder_layers))

    def forward(self,x,mean=False, z_q=None):
        blocks=[]
        used_latents=[]
        distributions=[]
        if isinstance(mean,bool):
            mean = [mean] * self.num_latent_levels

        features=x
        for i,block in enumerate(self.res_layers):
            #print("Block",i,block)
            features=block(features)
            blocks.append(features)
            if i!= self.num_levels-1:
                features=self.Pool_layers[i](features)

        decoder_features = blocks[-1]
        #print(decoder_features.shape,1)


        for proba_level in range(self.num_latent_levels):
            #print(proba_level)
            latent_dim = self._latent_dims[proba_level]
            mu_log_sigma = self.probabilistic_block(decoder_features)
            #print(mu_log_sigma.shape,"mu logsigma shape")

            # mu_log_sigma = torch.squeeze(mu_log_sigma,dim=1)
            # print(mu_log_sigma.shape,"mu logsigma shape squeeze")
            # print(mu_log_sigma[Ellipsis,:latent_dim].shape,"mu  shape Ellipsis")
            # print(mu_log_sigma[Ellipsis,latent_dim:].shape,"logsigma shape Ellipsis")
            mu = mu_log_sigma[:,:latent_dim]
            #print("mu shape:",mu.shape)
            log_sigma = mu_log_sigma[:,latent_dim:]
            #print("Logsigma shape:",log_sigma.shape)



            # mu = mu_log_sigma[:,:latent_dim,...]
            # print("mu shape:",mu.shape)
            # log_sigma = mu_log_sigma[:,latent_dim:,...]
            # print("Logsigma shape:",log_sigma.shape)
            dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
            #dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),0)

            distributions.append(dist)
        
            if z_q is not None:
                z = z_q[proba_level]
                #print(z.shape,"z_q")
            elif mean[proba_level]:
                z = dist.base_dist.loc
                #print(z.shape,"Proba level")
            else:
                z = dist.sample()
                #print(z.shape,"z shape")
            used_latents.append(z)
            # print(z.shape,"sample shape")
            decoder_output_lo = torch.cat([z, decoder_features], axis=1)
            # print(decoder_output_lo.shape,"decoder_lo")

            decoder_output_hi = self.interpolate(decoder_output_lo)
            # print(decoder_output_hi.shape,"decoder_hi")
            # print(blocks[::-1][proba_level + 1].shape,"block")
            decoder_features = torch.cat([decoder_output_hi, blocks[::-1][proba_level + 1]], axis=1)
            # print(decoder_features.shape)

            decoder_features = self.decoder_layers[proba_level](decoder_features)
        
        #print('decoder features {}'.format(decoder_features.shape))
            
        return {'decoder_features': decoder_features,
            'encoder_features': blocks,
            'distributions': distributions,
            'used_latents': used_latents}
        
        # return decoder_features,{
        #     'encoder_features': blocks,
        #     'distributions': distributions,
        #     'used_latents': used_latents}

########################################################
#                                                      #
#              StitchingDecoder model                  #
#                                                      #
########################################################



class StitchingDecoder(nn.Module):


    
    def __init__(self, dim,latent_dims,input_channels,channels_per_block, num_classes,
               down_channels_per_block=None, convs_per_block=3,
               blocks_per_level=3):
        super(StitchingDecoder, self).__init__()

        self.dim = dim
        conv = NDConvGenerator(self.dim)

        self._latent_dims = latent_dims
        self._channels_per_block = channels_per_block
        self.input_channels = input_channels
        self._num_classes = num_classes
        self._convs_per_block = convs_per_block
        self._blocks_per_level = blocks_per_level
        if down_channels_per_block is None:
            down_channels_per_block = channels_per_block
        self._down_channels_per_block = down_channels_per_block
        
        if self.dim == 2:
            self.interpolate = Interpolate(2,mode="bilinear")
        else:
            self.interpolate = Interpolate(2,mode="trilinear")
        self.residual_block = Residual_block
        self.decoder_feat = nn.ModuleList()

        
        num_latents = len(self._latent_dims)    
        start_level = num_latents + 1
        num_levels = len(self._channels_per_block)

        for level in range(start_level,num_levels,1):
            #print(level)
            next_d_layers = nn.ModuleList([Residual_block(dim=int(self.dim),
                            input_channels=int(self.input_channels[::-1][level-1]+self.input_channels[::-1][level]),
                            n_channels_in=int(self._channels_per_block[::-1][level]),
                            n_down_channels=int(self._down_channels_per_block[::-1][level]),
                            conv_per_block=3, 
                            stride=1,
                            norm=None,
                            relu='relu')]+[Residual_block(dim=int(self.dim),
                            input_channels=int(self._channels_per_block[::-1][level]),
                            n_channels_in=int(self._channels_per_block[::-1][level]),
                            n_down_channels=int(self._down_channels_per_block[::-1][level]),
                            conv_per_block=3, 
                            stride=1,
                            norm=None,
                            relu='relu') for _ in range(self._blocks_per_level-1)])


            self.decoder_feat.append(nn.Sequential(*next_d_layers)) 

        self.last_conv = conv(self.input_channels[1], num_classes, ks=1, stride=1, norm=None, relu=None)

    def forward(self,encoder_features=None,decoder_features=None):

        # Hcore = self.Hieararchical_core.forward(x,mean=False,z_q=None)
        # decoder_features = Hcore["decoder_features"]
        # blocks = Hcore["encoder_features"]
        num_latents = len(self._latent_dims)
        start_level = num_latents + 1
        num_levels = len(self._channels_per_block)   

        for idx,level in enumerate(range(start_level, num_levels, 1)):
            decoder_features = self.interpolate(decoder_features)
            # print(decoder_features.shape,"decoder_features shape",idx)
            # print(blocks[::-1][level].shape,"blocks[::-1][level] shape",idx)
            decoder_features = torch.cat([decoder_features,encoder_features[::-1][level]], axis=1)
            # print(decoder_features.shape,"decoder_features shape",idx)
            decoder_features = self.decoder_feat[idx](decoder_features)   
            # print(decoder_features.shape,"decoder_features shape",idx)

        logits = self.last_conv(decoder_features)
        
        return logits

########################################################
#                                                      #
#              HierarchicalProbUnet model              #
#                                                      #
########################################################

# class Backbone(HPUnetbackbone):
#      def __init__(self):
#         backbone = self.
#         super().__init__(backbone)








class HierarchicalProbUNet(nn.Module):


    def __init__(self,dim,
               latent_dims=(1, 1, 1, 1),
               input_channels=None,
               channels_per_block=None,
               num_classes=2,
               down_channels_per_block=None,
               convs_per_block=2,
               blocks_per_level=3):
        super(HierarchicalProbUNet, self).__init__()
        self.dim = dim
        conv = NDConvGenerator(self.dim)
        self._latent_dims = latent_dims
        self._channels_per_block = channels_per_block
        self.input_channels = input_channels
        self._num_classes = num_classes
        self._convs_per_block = convs_per_block
        self._blocks_per_level = blocks_per_level
        if down_channels_per_block is None:
            down_channels_per_block = channels_per_block
        self._down_channels_per_block = down_channels_per_block

        self._prior = Hierarchical_Core(dim=self.dim,input_channels=list(self.input_channels),channels_per_block=list(self._channels_per_block),
               down_channels_per_block=list(self._down_channels_per_block), convs_per_block=3,
               blocks_per_level=3,Posterior=False)

        self._posterior = Hierarchical_Core(dim=self.dim,input_channels=list(self.input_channels),channels_per_block=list(self._channels_per_block),
               down_channels_per_block=list(self._down_channels_per_block), convs_per_block=3,
               blocks_per_level=3,Posterior=True)

        self._f_comb = StitchingDecoder(dim=self.dim,latent_dims=self._latent_dims,input_channels=list(self.input_channels),channels_per_block=list(self._channels_per_block),num_classes=self._num_classes,
               down_channels_per_block=list(self._down_channels_per_block), convs_per_block=3,
               blocks_per_level=3)

        #self.StitchingDETR=StitchingDETR(num_classes=2, hidden_dim=192, nheads=8,
                 #num_encoder_layers=6, num_decoder_layers=6)
        
        #self.StitchingDETR_=DETR(backbone=HPUnetBackbone,transformer=transformer,num_classes=2,num_queries=50)


    def forward(self, seg, img):
        """
        Args:
        seg: A tensor of shape (b, h, w, num_classes).
        img: A tensor of shape (b, h, w, c)
        Returns: None
        """

        self._q_sample = self._posterior.forward(torch.cat([seg, img], axis=1), mean=False)
        self._q_sample_mean = self._posterior.forward(torch.cat([seg, img], axis=1), mean=True)

        self._p_sample = self._prior.forward(img, mean=False, z_q=None)
        self._p_sample_z_q = self._prior.forward(img, z_q=self._q_sample['used_latents'])
        self._p_sample_z_q_mean = self._prior.forward(img, z_q=self._q_sample_mean['used_latents'])

        # self.q_decoder,self._q_sample = self._posterior.forward(torch.cat([seg, img], axis=1), mean=False)
        # self.q_decoder_mean,self._q_sample_mean = self._posterior.forward(torch.cat([seg, img], axis=1), mean=True)

        # self.p_decoder,self._p_sample = self._prior.forward(img, mean=False, z_q=None)
        # self.p_decoder_zq,self._p_sample_z_q = self._prior.forward(img, z_q=self._q_sample['used_latents'])
        # self.p_decoder_zq_mean,self._p_sample_z_q_mean = self._prior.forward(img, z_q=self._q_sample_mean['used_latents'])


    # def detection(self, img,mean=False,z_q=None):
    #     """
    #     Args:
    #     seg: A tensor of shape (b, h, w, num_classes).
    #     img: A tensor of shape (b, h, w, c)
    #     Returns: A dict with {'pred_logits','pred_boxes}
    #     """
        
    #     prior_out = self._prior(img, mean,z_q)
    #     decoder_features = prior_out['decoder_features']
        
    #     return self.StitchingDETR(decoder_features)


    def sample(self, img, mean=False, z_q=None):
        """Sample a segmentation from the prior, given an input image.

        Args:
        img: A tensor of shape (b, h, w, c).
        mean: A boolean or a list of booleans. If a boolean, it specifies whether
            or not to use the distributions' means in ALL latent scales. If a list,
            each bool therein specifies whether or not to use the scale's mean. If
            False, the latents of the scale are sampled.
        z_q: None or a list of tensors. If not None, z_q provides external latents
            to be used instead of sampling them. This is used to employ posterior
            latents in the prior during training. Therefore, if z_q is not None, the
            value of `mean` is ignored. If z_q is None, either the distributions
            mean is used (in case `mean` for the respective scale is True) or else
            a sample from the distribution is drawn
        Returns:
        A segmentation tensor of shape (b, h, w, num_classes).
        """
        prior_out = self._prior.forward(img, mean, z_q)
        # d_feature,prior_out = self._prior(img, mean, z_q)
        encoder_features = prior_out['encoder_features']
        decoder_features = prior_out['decoder_features']
        # decoder_features = d_feature


        return self._f_comb(encoder_features=encoder_features,decoder_features=decoder_features)
        



    def sample_and_detect(self, img, mean=False, z_q=None):
        """Sample a segmentation from the prior, given an input image.

        Args:
        img: A tensor of shape (b, h, w, c).
        mean: A boolean or a list of booleans. If a boolean, it specifies whether
            or not to use the distributions' means in ALL latent scales. If a list,
            each bool therein specifies whether or not to use the scale's mean. If
            False, the latents of the scale are sampled.
        z_q: None or a list of tensors. If not None, z_q provides external latents
            to be used instead of sampling them. This is used to employ posterior
            latents in the prior during training. Therefore, if z_q is not None, the
            value of `mean` is ignored. If z_q is None, either the distributions
            mean is used (in case `mean` for the respective scale is True) or else
            a sample from the distribution is drawn
        Returns:
        A segmentation tensor of shape (b, h, w, num_classes).
        """
        prior_out = self._prior(img, mean, z_q)
        #prior_out = self._prior(img, mean, z_q)
        encoder_features = prior_out['encoder_features']
        decoder_features = prior_out['decoder_features']
        #decoder_features = prior_out['decoder_features']
        #decoder_features = self.d_decoder


        return self._f_comb(encoder_features=encoder_features,decoder_features=decoder_features),self.StitchingDETR(decoder_features)

    def reconstruct(self, seg, img, mean=False):
        """Reconstruct a segmentation using the posterior.

        Args:
        seg: A tensor of shape (b, h, w, num_classes).
        img: A tensor of shape (b, h, w, c).
        mean: A boolean, specifying whether to sample from the full hierarchy of
        the posterior or use the posterior means at each scale of the hierarchy.
        Returns:
        A segmentation tensor of shape (b,h,w,num_classes).
        """
        self.forward(seg,img)

        if mean:
            prior_out = self._p_sample_z_q_mean
            # d_decoder=self.p_decoder_zq_mean
        else:
            prior_out = self._p_sample_z_q
            # d_decoder=self.p_decoder_zq

        encoder_features = prior_out['encoder_features']
        decoder_features = prior_out['decoder_features']
        # decoder_features = d_decoder

        return self._f_comb(encoder_features=encoder_features,
                            decoder_features=decoder_features)


    def kl_divergence_(self, seg, img):

        """Kullback-Leibler divergence between the posterior and the prior.
        Args:
        seg: A tensor of shape (b, h, w, num_classes).
        img: A tensor of shape (b, h, w, c).
        Returns:
        A dictionary with keys indexing the hierarchy's levels and corresponding
        values holding the KL-term for each level (per batch).
        """
        self.forward(seg,img)
        posterior_out = self._q_sample
        prior_out = self._p_sample_z_q

        q_dists = posterior_out['distributions']
        p_dists = prior_out['distributions']
        #print(np.array(q_dists).shape,np.array(p_dists).shape,"here")


        kl_dict = {}
        for level, (q, p) in enumerate(zip(q_dists, p_dists)):
            # print(q.event_shape,"++++++",q.batch_shape)
            # print("++++++++++++++++++++++++++")
            # print(p.event_shape,"++++++",p.batch_shape)
        # Shape (b, h, w).
            kl_per_pixel = kl.kl_divergence(q, p)
            # print(kl_per_pixel.shape, "kl_per_pixel")
            # Shape (b,).
            if self.dim == 2:
                kl_per_instance = torch.sum(kl_per_pixel, dim=[1, 2])
                
            else:
                kl_per_instance = torch.sum(kl_per_pixel, dim=[1 ,2, 3])
                

            # Shape (1,).
            kl_dict[level] = torch.mean(kl_per_instance)
        # return kl_dict
        #print(kl_dict)

        return [kl.item() for _,kl in kl_dict.items()],torch.sum(torch.stack([kl for _, kl in kl_dict.items()], axis=-1))

    def elbo(self, seg, img):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """

        self.beta = 9
        self.kl = self.kl_divergence_(seg,img)
        self.reconstruction = self.reconstruct(seg,img,mean=False)
        self.loss_bce=nn.BCEWithLogitsLoss(size_average=False,reduce=False,reduction=False)
        self.criterion_loss_bce = self.loss_bce(input=self.reconstruction,target=seg)

        #self.loss_ce=F.cross_entropy(self.reconstruction, seg[:, 0].long())
        self.reconstruction_loss = torch.sum(self.criterion_loss_bce)
        #self.mean_reconstruction_loss = torch.mean(self.loss_ce)

        return -(self.reconstruction_loss + self.beta * self.kl)

class HPUnetBackbone(HierarchicalProbUNet):

    def __init__(self):
        #super(Backbone,self).__init__()
        super().__init__()


        self.body = self._prior
        print(self.body)

    def forward(self, tensor_list: NestedTensor):
        xs,_ = self.body(tensor_list.tensors,mean=False, z_q=None)
        print(xs.shape,"xs shape")
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out






#     def train_forward(self,sgm,img):
#         pass



# def _set_aux_loss(outputs_class, outputs_coord):
#     # this is a workaround to make torchscript happy, as torchscript
#     # doesn't support dictionary with non-homogeneous values, such
#     # as a dict having both a Tensor and a list.
#     return [{'pred_logits': a, 'pred_boxes': b}
#             for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]






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

channels_per_block = default_channels_per_block
down_channels_per_block = tuple([i / 2 for i in default_channels_per_block])
# net=Hierarchical_Core(dim=2,input_channels=list(input_channels),channels_per_block=list(channels_per_block),
#               down_channels_per_block=list(down_channels_per_block), convs_per_block=3,
#               blocks_per_level=3,Posterior=False)

# HPUnetscri=StitchingDecoder(dim=2,latent_dims=[1,1,1,1],input_channels=list(input_channels),channels_per_block=list(channels_per_block),num_classes=6,
#               down_channels_per_block=list(down_channels_per_block), convs_per_block=3,
#               blocks_per_level=3)


#net=HierarchicalProbUNet(dim=2,latent_dims=[1,1,1,1],input_channels=list(input_channels),channels_per_block=list(channels_per_block),num_classes=2,
               #down_channels_per_block=list(down_channels_per_block), convs_per_block=3,
               #blocks_per_level=3)
# #net=net.cuda()               
# #print(net)
# #print(HPUnet.parameters())               
# #print(HPUnet)
# #print(HPUnetscri)
# checkpoint_path = './chkpoint_withgen_babinksss'
# best_model_path = './bestmodel_withgen_babinksss.pt'
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# net.to(device)
# net.train()
# optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
# epochs = 20
# beta=1.0




# # train_dataset = MRI2DSegmentationDataset(data_dir=data_dir, slice_axis=1,transform=mt_transforms.ToTensor())
# train_dataset = MRI2DSegmentationDataset(data_dir=data_dir, slice_axis=1)

# print(len(train_dataset))

# # data = train_dataset[70]

# # print(data["input"].shape)
# # print(data["gt"].shape)
# # print(data["boxes"])
# # print(data["labels"])

# def prepare_loader(dataset,batch_size=2,shuffle=True):
    
#     train_set,valid_set = random_split(dataset,[int(len(dataset)*0.8),int(len(dataset)*0.2)+1])

#     train_loader = DataLoader(train_set, batch_size=batch_size, num_workers = 4, shuffle=shuffle, pin_memory=torch.cuda.is_available())
#     val_loader = DataLoader(valid_set, batch_size=batch_size, num_workers = 4, shuffle=shuffle, pin_memory=torch.cuda.is_available())
#     return train_loader,val_loader


# data_dir='/media/hmn-mednuc/InternalDisk_1/datasets/GAINED/resampled_croped/'

# valid_loss_min=float('inf')

# # TODO lesy way of capturing the logs, find a more elegant way to capture the logs 

# train_loss,val_loss=[],[]
# dice_score_train,dice_score_val=[],[]
# kls_loss_train,kls_loss_val=[],[]
# recons_loss_train,recons_loss_val=[],[]
# detection_loss_train,detection_loss_val=[],[]


# #print("zHere")


# for epoch in range(epochs):
    
#     dataset = MRI2DSegmentationDataset(data_dir=data_dir, slice_axis=1)
#     train_loader,val_loader=prepare_loader(dataset)

#     running_train_Detr_loss,running_train_reconstruction,running_train_kl_loss,running_train_total_loss,running_train_score = [[] for _ in range(5)]

#     print('Numbers of epoch:{}/{}'.format(epoch+1,epochs))
#     started = time.time()
          
#     for batch_idx, (train_batch_input , train_batch_gt , targets) in enumerate(train_loader):
#         #print('Batch idx {}, data shape {}, target shape {}'.format(batch_idx, data.shape, target.shape))
#         target,data=train_batch_gt.to(device),train_batch_input.to(device)
#         _,outputs=net.sample_and_detect(data,mean=True,z_q=None)
#         #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#         targets = [{"labels": l.to(device),"boxes":b.to(device)} for l,b in zip(targets["labels"],targets["boxes"])]
        
#         # loss from the DeTr '3 losses'
        
#         loss_dict = criterion(outputs, targets)
#         weight_dict = criterion.weight_dict
#         losses_detr = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
#         #print(loss_dict)


#         # kl divergence loss/loss_per_level part of the ELBO

#         kl_loss_per_levels,kl_loss = net.kl_divergence_(target,data)


#         # binary cross-entropy reconstruction loss part of ELBO

#         reconstruction = net.reconstruct(target,data,mean=False)
#         loss_bce = nn.BCEWithLogitsLoss(size_average=False,reduce=False,reduction=None)
#         criterion_reconstruction = loss_bce(input=reconstruction,target=target)
#         reconstruction_loss = torch.sum(criterion_reconstruction)


#         # definition of the ELBO

#         elbo =  -(reconstruction_loss + beta * kl_loss)
#         reg_loss = l2_regularisation(net._prior)+l2_regularisation(net._posterior)+l2_regularisation(net._f_comb)

#         # Total loss that will be used to for nack propagete the gradient + regularisation term omit for the DeTr for now 

        
#         total_loss = -elbo + losses_detr + 1e-5*reg_loss 
#         score = batch_dice(F.softmax(net.sample(data,mean=False),dim=1),target)
#         #running_loss += loss.item() * inputs.size(0) 
#         #print(loss) 
#         optimizer.zero_grad() 
#         total_loss.backward() 
#         optimizer.step() 

#         running_train_Detr_loss.append([loss_dict[k].item() * weight_dict[k] for k in loss_dict.keys() if k in weight_dict])
#         #print(len(running_train_Detr_loss))
#         running_train_total_loss.append(total_loss.item())
#         running_train_kl_loss.append(kl_loss_per_levels)
#         #print(len(running_train_kl_loss))
#         running_train_reconstruction.append(reconstruction_loss.item())
#         running_train_score.append(score.item())

#         #running_train_score.append(score.item())
        
        
#         #print('loss batch: {},Dice score batch: {}, batch_idx: {}'.format(loss.item(),score.item(),batch_idx))
#         print('Loss DeTr loss over one batch: {} ---- KL divergence loss over one batch: {} ---- Reconstruction loss over one batch: {} ---- Overall loss batch: {} ---- Overall score batch: {} ---- Batch idx: {}'.format(losses_detr.item(),kl_loss.item(),reconstruction_loss.item(),total_loss.item(),score.item(),batch_idx))

#     else:
#         running_valid_Detr_loss,running_valid_reconstruction,running_valid_kl_loss,running_valid_total_loss,running_valid_score = [[] for _ in range(5)]
          
#         with torch.no_grad():

#             for batch_idx, (valid_batch_input , valid_batch_gt , targets) in enumerate(val_loader):
#                 target,data=valid_batch_gt.to(device),valid_batch_input.to(device)
#                 _,outputs=net.sample_and_detect(data,mean=True,z_q=None)
#                 #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#                 targets = [{"labels": l.to(device),"boxes":b.to(device)} for l,b in zip(targets["labels"],targets["boxes"])]
                
#                 # loss from the DeTr '3 losses'
                
#                 loss_dict = criterion(outputs, targets)
#                 weight_dict = criterion.weight_dict
#                 losses_detr = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)


#                 # kl divergence loss/loss_per_level part of the ELBO

#                 kl_loss_per_levels,kl_loss = net.kl_divergence_(target,data)


#                 # binary cross-entropy reconstruction loss part of ELBO

#                 reconstruction = net.reconstruct(target,data,mean=False)
#                 loss_bce = nn.BCEWithLogitsLoss(size_average=False,reduce=False,reduction=None)
#                 criterion_reconstruction = loss_bce(input=reconstruction,target=target)
#                 reconstruction_loss = torch.sum(criterion_reconstruction)


#                 # definition of the ELBO

#                 elbo =  -(reconstruction_loss + beta * kl_loss)
#                 reg_loss = l2_regularisation(net._prior)+l2_regularisation(net._posterior)+l2_regularisation(net._f_comb)

#                 # Total loss that will be used to for nack propagete the gradient + regularisation term omit for the DeTr for now 
#                 total_loss = -elbo + losses_detr + 1e-5*reg_loss 

#                 score = batch_dice(F.softmax(net.sample(data,mean=False),dim=1),target)

 
#                 running_valid_Detr_loss.append([loss_dict[k].item() * weight_dict[k] for k in loss_dict.keys() if k in weight_dict])
#                 #print(len(running_valid_Detr_loss[0]))
#                 running_valid_total_loss.append(total_loss.item())
#                 running_valid_kl_loss.append(kl_loss_per_levels)
#                 running_valid_reconstruction.append(reconstruction_loss.item())
#                 running_valid_score.append(score.item())
        
#     epoch_train_loss,epoch_train_kl,epoch_train_score,epoch_train_reconstruction,epoch_train_detr = np.mean(running_train_total_loss),np.mean(running_train_kl_loss,axis=0),np.mean(running_train_score),np.mean(running_train_reconstruction),np.mean(running_train_Detr_loss,axis=0)
#     print('Train total loss epoch : {} Dice score epoch : {}'.format(epoch_train_loss,epoch_train_score))
#     train_loss.append(epoch_train_loss)
#     dice_score_train.append(epoch_train_score)
#     kls_loss_train.append(epoch_train_kl)
#     recons_loss_train.append(epoch_train_reconstruction)
#     detection_loss_train.append(epoch_train_detr)

#     epoch_val_loss,epoch_val_kl,epoch_val_score,epoch_val_reconstruction,epoch_val_detr = np.mean(running_valid_total_loss),np.mean(running_valid_kl_loss,axis=0),np.mean(running_valid_score),np.mean(running_valid_reconstruction),np.mean(running_valid_Detr_loss,axis=0)
#     print('Valid total loss epoch: {} Dice score epoch : {}'.format(epoch_val_loss,epoch_val_score))
#     val_loss.append(epoch_val_loss)
#     dice_score_val.append(epoch_val_score)
#     kls_loss_val.append(epoch_val_kl)
#     recons_loss_val.append(epoch_val_reconstruction)
#     detection_loss_val.append(epoch_val_detr)
          
#     checkpoint = { 'epoch': epoch +1,
#                   'valid_loss_min':epoch_val_loss,
#                   'state_dict':net.state_dict(),
#                   'optimizer':optimizer.state_dict(),
        
#     }
#     save_ckp(checkpoint, False,checkpoint_path,best_model_path)
     
#     if epoch_val_loss <= valid_loss_min:
#           print('Validation loss decreased ({:.6f} =======> {:.6f}). Saving model ...'.format(valid_loss_min,epoch_val_loss))
          
#           save_ckp(checkpoint, True,checkpoint_path,best_model_path)
#           valid_loss_min = epoch_val_loss
          
#     time_passed = time.time() - started
#     print('{:.0f}m {:.0f}s'.format(time_passed//60, time_passed%60))

    # print(train_loss,
    # dice_score_train,
    # kls_loss_train,
    # recons_loss_train,
    # detection_loss_train)
#net.eval()
#sample_1 = net.sample(torch.from_numpy(all_pt_img[25550][np.newaxis][np.newaxis]/100).cuda(),mean=True,z_q=None)
#sample_2 = net.sample(torch.from_numpy(all_pt_img[25550][np.newaxis]/100).cuda(),mean=True,z_q=None)

# print(sample,sample.shape,"Sample shape")
# prekd = torch.argmax(sample,axis=1)
# print(pred,pred.shape)







# # # sample=net.sample(torch.ones(2,1,256,256).cuda(),mean=True,z_q=None)
# # # detection=net.detection(torch.ones(2,1,256,256).cuda(),mean=True,z_q=None)
# # # outputs = detection

# # # loss_dict = criterion(outputs, targets)
# # # print(detection['pred_logits'].shape)
# # # print(detection['pred_boxes'].shape)

































# # print(sample.shape)
# # reconstruction=HPUnet.reconstruct(torch.randn(1,6,128,128),torch.ones(1,1,128,128),mean=True)
# # print(reconstruction.shape,reconstruction)
# # print(sample.shape,"Sample shape")
# # print(sample)
# # HPUnetscri.forward(torch.ones(1,1,128,128))
# # summary(HPUnetscri.cuda(),input_size=[(1,128,128)])
# #summary(HPUnet.cuda(),input_size=[(1,128,128),(1,128,128)])
# #summary(HPUnet.cuda(),input_size=[(1,128,128)])

# #k_l=HPUnet.kl(torch.randn(2,6,128,128,128).cuda(),torch.ones(2,1,128,128,128).cuda())
# #print(k_l)
# #elbo=HPUnet.elbo(torch.ones(1,6,128,128,128),torch.ones(1,1,128,128,128))
# #print(elbo)


# #dummy train loop for debugging purpose

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net.to(device)
# net.train()
# optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
# epochs = 50
# train_loss=[]

# for epoch in range(epochs):

   
#     running_train_loss = []
#     print("Numbers epoch:",epoch)
#     for batch_idx, (data, target) in enumerate(loader):
#         print('Batch idx {}, data shape {}, target shape {}'.format(
#         batch_idx, data.shape, target.shape))


        
#         elbo = net.elbo(target.to(device),data.to(device))
#         loss = -elbo + 1e-5 
#         #running_loss += loss.item() * inputs.size(0)
#         #print(loss)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         running_train_loss.append(loss.item())
    
#     epoch_train_loss = np.mean(running_train_loss) 
#     print('Train loss : {}'.format(epoch_train_loss))                       
#     train_loss.append(epoch_train_loss)





# sample = net.sample(torch.randn(1,1,128,128).cuda(),mean=True,z_q=None)
# print(sample,sample.shape,"Sample shape")
# pred = torch.argmax(sample,axis=1)
# print(pred,pred.shape)







