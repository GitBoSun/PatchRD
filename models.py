import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import numpy as np

#cell = 4
#input 256
#output 120 (128-4-4)
#receptive field = 18

#            0  18
#conv 4x4 s1 4  15
#conv 3x3 s2 6  7
#conv 3x3 s1 10 5
#conv 3x3 s1 14 3
#conv 3x3 s1 18 1
#conv 1x1 s1 1  1

#            0  41
#conv 3x3 s1 4  39
#conv 3x3 s2 6  19
#conv 3x3 s1 10  17
#conv 3x3 s1 14 15
#conv 3x3 s2 18 7
#conv 3x3 s1 26 7

leaky_f = 0.02
USE_25=True

class CoarseCompletor_skip(nn.Module):
    def __init__(self, d_dim, ):
        super(CoarseCompletor_skip, self).__init__()
        self.d_dim = d_dim

        self.conv_1 = nn.Conv3d(1,             self.d_dim,    5, stride=1, padding=2, bias=True)
        self.conv_2 = nn.Conv3d(self.d_dim,    self.d_dim*2,  3, stride=2, padding=1, bias=True) # 64

        self.conv_3 = nn.Conv3d(self.d_dim*2,  self.d_dim*2,  3, stride=1, padding=1, bias=True)
        self.conv_4 = nn.Conv3d(self.d_dim*2,  self.d_dim*4,  3, stride=2, padding=1, bias=True) # 32

        self.conv_5 = nn.Conv3d(self.d_dim*4,  self.d_dim*4,  3, stride=1, padding=1, bias=True)
        self.conv_6 = nn.Conv3d(self.d_dim*4,  self.d_dim*8,  3, stride=2, padding=1, bias=True) # 16

        self.conv_7 = nn.Conv3d(self.d_dim*8,  self.d_dim*8,  3, stride=1, padding=1, bias=True)
        self.conv_8 = nn.Conv3d(self.d_dim*8,  self.d_dim*8,  3, stride=2, padding=1, bias=True) # 8

        self.conv_9 = nn.Conv3d(self.d_dim*8,  self.d_dim*8,  3, stride=2, padding=1, bias=True) # 4
        self.conv_10 = nn.Conv3d(self.d_dim*8,  self.d_dim*8,  3, stride=1, padding=1, bias=True) # 4
        
        self.dconv_1 = nn.ConvTranspose3d(self.d_dim*8,  self.d_dim*8, 4, stride=2, padding=1, bias=True) # 8

        self.dconv_2 = nn.Conv3d(self.d_dim*16,  self.d_dim*8, 3, stride=1, padding=1, bias=True)
        self.dconv_3 = nn.ConvTranspose3d(self.d_dim*8,  self.d_dim*8, 4, stride=2, padding=1, bias=True) # 16

        self.dconv_4 = nn.Conv3d(self.d_dim*16,  self.d_dim*8, 3, stride=1, padding=1, bias=True)
        self.dconv_5 = nn.ConvTranspose3d(self.d_dim*8,  self.d_dim*4, 4, stride=2, padding=1, bias=True) # 32

        self.dconv_6 = nn.Conv3d(self.d_dim*8,  self.d_dim*4, 3, stride=1, padding=1, bias=True)
        self.dconv_7 = nn.ConvTranspose3d(self.d_dim*4,  self.d_dim*2, 4, stride=2, padding=1, bias=True) # 64

        self.dconv_8 = nn.Conv3d(self.d_dim*4,  self.d_dim*2, 3, stride=1, padding=1, bias=True)
        self.dconv_9 = nn.ConvTranspose3d(self.d_dim*2,  self.d_dim, 4, stride=2, padding=1, bias=True) # 128

        self.dconv_10 = nn.Conv3d(self.d_dim,  1,  3, stride=1, padding=1, bias=True)

    def forward(self, partial_in, is_training=False):
        out = partial_in
        out = self.conv_1(out)
        out1 = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

        out = self.conv_2(out1)
        out2 = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

        out = self.conv_3(out2)
        out3 = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

        out = self.conv_4(out3)
        out4 = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

        out = self.conv_5(out4)
        out5 = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

        out = self.conv_6(out5)
        out6 = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

        out = self.conv_7(out6)
        out7 = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

        out = self.conv_8(out7)
        out8 = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

        out = self.conv_9(out8)
        out9 = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

        out = self.conv_10(out9)
        out10 = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

        out = self.dconv_1(out10)
        outd1 = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

        out = torch.cat([outd1, out8], dim=1)
        out = self.dconv_2(out)
        outd2 = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

        out = self.dconv_3(outd2)
        outd3 = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

        out = torch.cat([outd3, out6], dim=1)
        out = self.dconv_4(out)
        outd4 = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

        out = self.dconv_5(outd4)
        outd5 = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

        out = torch.cat([outd5, out4], dim=1)
        out = self.dconv_6(out)
        outd6 = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

        out = self.dconv_7(outd6)
        outd7 = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

        out = torch.cat([outd7, out2], dim=1)
        out = self.dconv_8(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

        out = self.dconv_9(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

        out = self.dconv_10(out)

        #out = torch.max(torch.min(out, out*0.002+0.998), out*0.002-0.998)
        out = torch.sigmoid(out)
        return out


class PatchEncoder(nn.Module):
    def __init__(self, d_dim, z_dim):
        super(PatchEncoder, self).__init__()
        self.d_dim = d_dim
        self.z_dim = z_dim

        self.conv_1 = nn.Conv3d(1,             self.d_dim,    4, stride=1, padding=0, bias=True)
        self.conv_2 = nn.Conv3d(self.d_dim,    self.d_dim*2,  3, stride=2, padding=0, bias=True)
        self.conv_3 = nn.Conv3d(self.d_dim*2,  self.d_dim*4,  3, stride=1, padding=0, bias=True)
        self.conv_4 = nn.Conv3d(self.d_dim*4,  self.d_dim*8,  3, stride=1, padding=0, bias=True)
        self.conv_5 = nn.Conv3d(self.d_dim*8,  self.d_dim*16, 3, stride=2, padding=0, bias=True)
        if USE_25:
            self.conv_6 = nn.Conv3d(self.d_dim*16, self.d_dim*16, 3, stride=1, padding=1, bias=True) # extra layer
            self.conv_7 = nn.Conv3d(self.d_dim*32, self.z_dim, 1, stride=1, padding=0, bias=True)
        else:

            self.conv_6 =  nn.Conv3d(self.d_dim*16, self.z_dim*2, 1, stride=1, padding=0, bias=True)
            self.conv_7 =  nn.Conv3d(self.z_dim*2, self.z_dim, 1, stride=1, padding=0, bias=True)

    def forward(self, voxels, return_feat=False, is_training=False):
        out = voxels

        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        feat1 = out.detach()

        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        feat2 = out.detach()

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        feat3 = out.detach()

        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        feat4 = out.detach()

        out_1 = self.conv_5(out)
        out_1 = F.leaky_relu(out_1, negative_slope=leaky_f, inplace=True)
        feat5 = out_1.detach()

        if USE_25:
            out_2 = self.conv_6(out_1)
            out_2 = F.leaky_relu(out_2, negative_slope=leaky_f, inplace=True)
            feat6 = out_2.detach()
            out = self.conv_7(torch.cat([out_1, out_2], dim=1))
        else:
            out = self.conv_6(out_1)
            out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
            out = self.conv_7(out)

        out = torch.max(torch.min(out, out*0.002+0.998), out*0.002-0.998)
        # out = torch.sigmoid(out)
        if return_feat:
            return out, feat1, feat2, feat3, feat4, feat5, feat6
        return out


def reshape_to_size_torch(v1, v2_shape):
    _,_,x1,y1,z1 = v1.shape
    x,y,z = v2_shape
    new_v1 = v1
    padding = np.zeros(6)
    if z1 < z:
        padding[0], padding[1] = int((z-z1)/2), z-z1-int((z-z1)/2)
    else:
        new_v1 = new_v1[:,:,:, :,int((z1-z)/2):int((z1-z)/2)+z ]
    if y1 < y:
        padding[2], padding[3] = int((y-y1)/2), y-y1-int((y-y1)/2)
    else:
        new_v1 = new_v1[:,:,:, int((y1-y)/2):int((y1-y)/2)+y,: ]
    if x1 < x:
        padding[4], padding[5] = int((x-x1)/2), x-x1-int((x-x1)/2)
    else:
        new_v1 = new_v1[:,:,int((x1-x)/2):int((x1-x)/2)+x,:,: ]
    new_v1 = F.pad(new_v1, tuple(padding.astype(np.int8)))
    return new_v1

class PatchDeformer(nn.Module):
    def __init__(self, d_dim, z_dim, pre_dis=False, include_coarse=False):
        super(PatchDeformer, self).__init__()
        self.d_dim = d_dim
        self.z_dim = z_dim
        self.pred_dis = pre_dis
        self.include_coarse = include_coarse

        self.conv_1 = nn.Conv3d(1,             self.d_dim,    3, stride=1, padding=1, bias=True)
        self.conv_2 = nn.Conv3d(self.d_dim,    self.d_dim*2,  3, stride=2, padding=1, bias=True)
        self.conv_3 = nn.Conv3d(self.d_dim*2,  self.d_dim*4,  3, stride=2, padding=1, bias=True)

        self.conv_d1 = nn.Conv3d(1,             self.d_dim,    3, stride=1, padding=1, bias=True)
        self.conv_d2 = nn.Conv3d(self.d_dim,    self.d_dim*2,  3, stride=2, padding=1, bias=True)
        self.conv_d3 = nn.Conv3d(self.d_dim*2,  self.d_dim*4,  3, stride=2, padding=1, bias=True)

        self.fc1 = nn.Linear(self.d_dim*8*5*5*5, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, self.z_dim)

        if self.pred_dis:
            self.dis_fc1 = nn.Linear(self.d_dim*8*5*5*5, 2048)
            self.dis_fc2 = nn.Linear(2048, 512)
            self.dis_fc3 = nn.Linear(512, 1)

        if self.include_coarse:
            self.cdis_fc1 = nn.Linear(self.d_dim*4*5*5*5, 1024)
            self.cdis_fc2 = nn.Linear(1024, 512)
            self.cdis_fc3 = nn.Linear(512, 1)

            self.ddis_fc1 = nn.Linear(self.d_dim*4*5*5*5, 1024)
            self.ddis_fc2 = nn.Linear(1024, 512)
            self.ddis_fc3 = nn.Linear(512, 1)

    def forward(self, c_vox, d_vox, is_training=False):

        out = c_vox
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out_1 = out.view(-1, 5*5*5*self.d_dim*4)

        out = d_vox
        out = self.conv_d1(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.conv_d2(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.conv_d3(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out_2 = out.view(-1, 5*5*5*self.d_dim*4)


        out = torch.cat([out_1, out_2], dim=1)
        out = self.fc1(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.fc2(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.fc3(out)
        # out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out_de = torch.max(torch.min(out, out*0.002+0.998), out*0.002-0.998)

        if self.pred_dis:
            out = torch.cat([out_1, out_2], dim=1)
            out = self.dis_fc1(out)
            out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
            out = self.dis_fc2(out)
            out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
            out = self.dis_fc3(out)

            out_dis = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
            # out_dis = torch.max(torch.min(out, out*0.002+0.998), out*0.002)
            if not self.include_coarse:
                return out_de, out_dis
            else:
                # out = torch.cat([out_1, out_2], dim=1)
                out = out_1
                out = self.cdis_fc1(out)
                out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
                out = self.cdis_fc2(out)
                out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
                out = self.cdis_fc3(out)
                out_c_dis = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

                out = out_2
                out = self.ddis_fc1(out)
                out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
                out = self.ddis_fc2(out)
                out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
                out = self.ddis_fc3(out)
                out_d_dis = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

                return out_de, out_dis, out_c_dis, out_d_dis

        return out_de

class JointDeformer(nn.Module):
    def __init__(self, d_dim, dw_dim, k_dim,loc_size, use_mean_x=False, wd_size=8):
        super(JointDeformer, self).__init__()
        self.d_dim = d_dim
        self.dw_dim = dw_dim
        self.k_dim = k_dim
        self.d = loc_size
        self.use_mean_x = use_mean_x
        self.wd_size = wd_size

        # input
        self.conv_1 = nn.Conv3d(2,             self.d_dim,    5, stride=1, padding=2, bias=True)
        self.conv_2 = nn.Conv3d(self.d_dim,    self.d_dim*2,  3, stride=2, padding=1, bias=True) # 64
        self.conv_3 = nn.Conv3d(self.d_dim*2,  self.d_dim*2,  3, stride=1, padding=1, bias=True)
        self.conv_4 = nn.Conv3d(self.d_dim*2,  self.d_dim*4,  3, stride=2, padding=1, bias=True) # 32

        self.conv_5 = nn.Conv3d(self.d_dim*4,  self.d_dim*4,  3, stride=1, padding=1, bias=True)
        self.conv_6 = nn.Conv3d(self.d_dim*4,  self.d_dim*8,  3, stride=2, padding=1, bias=True) # 16
        self.conv_7 = nn.Conv3d(self.d_dim*8,  self.d_dim*8,  3, stride=2, padding=1, bias=True) # 8
        self.conv_8 = nn.Conv3d(self.d_dim*8,  self.d_dim*8,  3, stride=2, padding=1, bias=True) # 4

        # partial
        self.conv_p1 = nn.Conv3d(1,             self.d_dim,   5, stride=1, padding=2, bias=True)
        self.conv_p2 = nn.Conv3d(self.d_dim,    self.d_dim*2,  3, stride=2, padding=1, bias=True) # 64
        self.conv_p3 = nn.Conv3d(self.d_dim*2,  self.d_dim*2,  3, stride=1, padding=1, bias=True)
        self.conv_p4 = nn.Conv3d(self.d_dim*2,  self.d_dim*4,  3, stride=2, padding=1, bias=True) # 32

        self.conv_p5 = nn.Conv3d(self.d_dim*4,  self.d_dim*4,  3, stride=1, padding=1, bias=True)
        self.conv_p6 = nn.Conv3d(self.d_dim*4,  self.d_dim*8,  3, stride=2, padding=1, bias=True) # 16
        self.conv_p7 = nn.Conv3d(self.d_dim*8,  self.d_dim*8,  3, stride=2, padding=1, bias=True) # 8
        self.conv_p8 = nn.Conv3d(self.d_dim*8,  self.d_dim*8,  3, stride=2, padding=1, bias=True) # 4

        # windows
        self.conv_w0 = nn.Conv3d(self.k_dim,     self.dw_dim,    5, stride=1, padding=2, bias=True)
        self.conv_w1 = nn.Conv3d(self.dw_dim,    self.dw_dim,    3, stride=1, padding=1, bias=True)
        self.conv_w2 = nn.Conv3d(self.dw_dim,    self.dw_dim*2,  3, stride=2, padding=1, bias=True) # 12 . 16
        self.conv_w3 = nn.Conv3d(self.dw_dim*2,  self.dw_dim*2,  3, stride=1, padding=1, bias=True)
        self.conv_w4 = nn.Conv3d(self.dw_dim*2,  self.dw_dim*4,  3, stride=2, padding=1, bias=True) # 6 . 8

        self.conv_w5 = nn.Conv3d(self.dw_dim*4,  self.dw_dim*4,  3, stride=1, padding=1, bias=True)
        self.conv_w6 = nn.Conv3d(self.dw_dim*4,  self.dw_dim*8,  3, stride=2, padding=1, bias=True) # 3 . 4
        if self.wd_size==16:
            self.conv_w7 = nn.Conv3d(self.dw_dim*8,  self.dw_dim*8,  3, stride=2, padding=1, bias=True) # 3 . 4


        self.fc1 = nn.Linear((self.dw_dim*8*self.d*self.d*self.d + self.d_dim*8*4*4*4*2), 4096)
        self.fc2 = nn.Linear(4096, 1024)

        # X
        self.fc_x1 = nn.Linear(1024, 256)
        self.fc_x2 = nn.Linear(256, 1024)
        self.fc_x3 = nn.Linear(1024, self.dw_dim*8*self.d*self.d*self.d)
        self.dconv_x0 = nn.Conv3d(self.dw_dim*8,  self.dw_dim*8,  3, stride=1, padding=1, bias=True)
        self.dconv_x1 = nn.ConvTranspose3d(self.dw_dim*8,  self.dw_dim*4, 4, stride=2, padding=1, bias=True) # 8
        self.dconv_x2 = nn.Conv3d(self.dw_dim*4,  self.dw_dim*4, 3, stride=1, padding=1, bias=True)
        self.dconv_x3 = nn.ConvTranspose3d(self.dw_dim*4,  self.dw_dim*2, 4, stride=2, padding=1, bias=True) # 16
        self.dconv_x4 = nn.Conv3d(self.dw_dim*2,  self.d_dim*2, 3, stride=1, padding=1, bias=True)
        self.dconv_x5 = nn.ConvTranspose3d(self.dw_dim*2,  self.dw_dim, 4, stride=2, padding=1, bias=True) # 32
        self.dconv_x6 = nn.Conv3d(self.dw_dim,  self.dw_dim, 3, stride=1, padding=1, bias=True)
        self.dconv_x7 = nn.Conv3d(self.dw_dim,  self.k_dim, 3, stride=1, padding=1, bias=True)

        # D
        self.fc_d1 = nn.Linear(1024, 1024)
        self.fc_d2 = nn.Linear(1024, 1024)
        self.fc_d3 = nn.Linear(1024, self.k_dim*3)


    def forward(self, partial_in, input_in, windows_in, window_masks_in, loc_mask_in,  is_training=False):
        leaky_f = 0.02
        out = torch.cat([input_in, window_masks_in], dim=1)
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.conv_6(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.conv_7(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.conv_8(out)
        out_coa = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

        out = partial_in
        out = self.conv_p1(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.conv_p2(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.conv_p3(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.conv_p4(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.conv_p5(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.conv_p6(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.conv_p7(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.conv_p8(out)
        out_p = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

        out = windows_in
        out = self.conv_w0(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.conv_w1(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.conv_w2(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.conv_w3(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.conv_w4(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.conv_w5(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.conv_w6(out)
        out_w = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        if self.wd_size==16:
            out_w = self.conv_w7(out_w)
            out_w = F.leaky_relu(out_w, negative_slope=leaky_f, inplace=True)

        out_coa = out_coa.view(1, self.d_dim*8*4*4*4)
        out_p = out_p.view(1, self.d_dim*8*4*4*4)
        out_w =out_w.view(1, self.dw_dim*8*self.d*self.d*self.d)

        out = torch.cat((out_w, out_coa, out_p), dim=1)
        out = self.fc1(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.fc2(out)
        out_latent = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

        # D
        out = self.fc_d1(out_latent)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.fc_d2(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.fc_d3(out)
        # out_D = torch.max(torch.min(out, out*0.002+0.998), out*0.002-0.998)
        out_D = torch.clip(out, -1.0, 1.0)
        out_D = out_D.view(self.k_dim, 3)

        # X
        out = self.fc_x1(out_latent)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.fc_x2(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.fc_x3(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)

        out = out.view(1, self.dw_dim*8, self.d,self.d,self.d)
        out = self.dconv_x0(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        #print('0',out.shape)
        out = self.dconv_x1(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        #print('1',out.shape)
        out = self.dconv_x2(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        #print('2',out.shape)
        out = self.dconv_x3(out)
        #print('3',out.shape)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        out = self.dconv_x4(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        #print('4',out.shape)
        out = self.dconv_x5(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        #print('5',out.shape)
        out = self.dconv_x6(out)
        out = F.leaky_relu(out, negative_slope=leaky_f, inplace=True)
        #print('6',out.shape)
        out_X = self.dconv_x7(out) # [1, kdim, vx, vy, vz]

        # out_X = torch.max(torch.min(out_X,out_X*0.002+0.998), out_X*0.002)
        # out_X = torch.sigmoid(out_X)
        out_X = torch.clip(out_X, 0.0, 1.0)
        # loc_mask_in = F.interpolate(loc_mask_in, scale_factor=8, mode='nearest')
        out_X = F.avg_pool3d(out_X, kernel_size=8, stride=8)

        if self.use_mean_x:
            out_X = out_X*loc_mask_in/(torch.sum(out_X*loc_mask_in, dim=1).unsqueeze(1)+1e-5)
        else:
            out_X = torch.exp(out_X)*loc_mask_in/(torch.sum(torch.exp(out_X)*loc_mask_in, dim=1).unsqueeze(1)+1e-5)

        #print('out_x', out_X.mean())
        return out_D, out_X


