import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
import numpy as np
import random
from einops import rearrange
from hyper_params import hp


class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super(SpatialGatingUnit, self).__init__()
        self.norm = nn.LayerNorm(d_ffn//2)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, 1)

    def forward(self, x):
        u,v = torch.chunk(x, 2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        out = u*v
        return out


class GMLPblock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super(GMLPblock, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn)
        self.channel_proj2 = nn.Linear(d_ffn//2, d_model)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)

    def forward(self, x):
        residual = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)+residual
        return x


class myencoder(nn.Module):
    def __init__(self, hps, d_model=128, d_ffn=768, patch_size=16, image_size=hp.graph_picture_size):
        super(myencoder, self).__init__()
        self.hps = hps
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.patcher = nn.Conv2d(1, d_model, patch_size, patch_size)

        self.pos_emb = nn.Parameter(torch.zeros(1,self.num_patches,d_model))

        self.downsampling1 = GMLPblock(d_model, d_ffn, self.num_patches)
        self.downsampling2 = GMLPblock(d_model, d_ffn, self.num_patches)
        self.downsampling3 = GMLPblock(d_model, d_ffn, self.num_patches)
        self.downsampling4 = GMLPblock(d_model, d_ffn, self.num_patches)

        self.upsampling4 = GMLPblock(d_model, d_ffn, self.num_patches)
        self.upsampling3 = GMLPblock(d_model, d_ffn, self.num_patches)
        self.upsampling2 = GMLPblock(d_model, d_ffn, self.num_patches)
        self.upsampling1 = GMLPblock(d_model, d_ffn, self.num_patches)
        # B * 64 * 128 -> B * 64 * 256 -> B * 1 * 128 * 128
        self.reclinear = nn.Linear(d_model, 256)

        self.active = nn.LayerNorm(d_model)
        self.fc_mu = nn.Linear(d_model, self.hps.Nz)
        self.fc_sigma = nn.Linear(d_model, self.hps.Nz)
        self.criterionL1 = nn.L1Loss()
        self.criterionL2 = nn.MSELoss()

    def forward(self, graph):
        gt = graph[:, 1].view(graph.shape[0],1,graph.shape[-1],graph.shape[-1])
        shape = gt.shape
        B, C, H, W = shape
        image = graph[:,0].view(graph.shape[0],1,graph.shape[-1],graph.shape[-1])

        top,  bottom = self.get_feature(image)
        #gt_top,  _ = self.get_feature(gt)

        latent_code = torch.mean(self.active(top),dim=1).view(shape[0], 128)
        mu = self.fc_mu(latent_code)
        sigma = self.fc_sigma(latent_code)
        sigma_e = torch.exp(sigma / 2.)
        z_size = mu.size()
        if mu.get_device() != -1:  # not in cpu
            n = torch.normal(torch.zeros(z_size), torch.ones(z_size)).cuda(mu.get_device())
        else:  # in cpu
            n = torch.normal(torch.zeros(z_size), torch.ones(z_size))
        # sample z
        z = mu + sigma_e * n

        return z, mu, sigma, latent_code, 0, self.criterionL2(bottom, gt)

    def get_feature(self, image):
        x = self.patcher(image)
        # B,d_model,8,8 -> B,64,d_model
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape((B, -1, C)) + self.pos_emb.repeat(B,1,1)
        x1 = self.downsampling1(x)
        x2 = self.downsampling2(x1)
        x3 = self.downsampling3(x2)
        x4 = self.downsampling4(x3)

        y3 = 0.5*self.upsampling4(x4) + 0.5*x3
        y2 = 0.5*self.upsampling3(y3) + 0.5*x2
        y1 = 0.5*self.upsampling2(y2) + 0.5*x1
        y0 = self.upsampling1(y1)
        # y0 [B, num_patches, 128]
        y0 = self.reclinear(y0)
        y0 = rearrange(y0, "b (ph pw) (ps1 ps2) -> b (ph ps1) (pw ps2)", ph=8, ps1=16)
        y0 = torch.tanh(y0)
        y0 = y0.unsqueeze(1)
        '''
        loop 2
        '''
        for i in range(hp.T):
            #print("shape",y0.shape)
            y0 = self.patcher(y0)
            y0 = y0.permute(0, 2, 3, 1).reshape((B, -1, C)) + self.pos_emb.repeat(B,1,1)
            x1 = 0.5*self.downsampling1(y0)+0.5*y1
            x2 = 0.5*self.downsampling2(x1)+0.5*y2
            x3 = 0.5*self.downsampling3(x2)+0.5*y3
            x4 = 0.5*self.downsampling4(x3)

            y3 = 0.5 * self.upsampling4(x4) + 0.5 * x3
            y2 = 0.5 * self.upsampling3(y3) + 0.5 * x2
            y1 = 0.5 * self.upsampling2(y2) + 0.5 * x1
            y0 = self.upsampling1(y1)
            y0 = self.reclinear(y0)
            y0 = rearrange(y0, "b (ph pw) (ps1 ps2) -> b (ph ps1) (pw ps2)", ph=8, ps1=16)
            y0 = torch.tanh(y0)
            y0 = y0.unsqueeze(1)
        return x4, y0
