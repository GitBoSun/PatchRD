import os
import time
import math
import random
from typing import get_type_hints
import numpy as np
import cv2
import h5py
import open3d as o3d
from psbody.mesh import Mesh
from glob import glob

from scipy.ndimage.filters import gaussian_filter
from sklearn.manifold import TSNE
from scipy.ndimage import rotate, zoom, shift, map_coordinates, sobel
from sklearn.neighbors import KernelDensity
from skimage import measure
from scipy.linalg import sqrtm

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from utils import *
from models import *

from kornia.augmentation import RandomAffine3D
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D
chamLoss = ChamferDistancePytorch.chamfer3D.dist_chamfer_3D.chamfer_3DDist()

class MODEL_COMPLETE(object):
    def __init__(self, config):
        self.real_size = 256
        #self.real_size = 128
        self.mask_margin = 8

        self.g_dim = config.g_dim
        self.input_size = config.input_size # 32
        self.output_size = config.output_size # 128

        # self.upsample_rate = self.output_size//self.input_size
        self.upsample_rate = 1
        self.csize= config.csize
        self.c_range = config.c_range

        self.asymmetry = True # config.asymmetry

        self.save_epoch = 5
        self.eval_epoch = 10
        self.start_epoch = 0

        self.gua_filter = True
        self.mode=config.mode

        self.w_posi = config.w_posi # 1.5
        self.w_mask = config.w_mask # 6

        self.dump_points = False
        self.sampling_threshold = 0.3
        self.render_view_id = 0
        self.voxel_renderer = voxel_renderer(self.real_size)

        self.checkpoint_dir = config.checkpoint_dir
        self.data_dir = config.data_dir
        self.data_content = config.data_content # content_chair
        
        self.log_path = os.path.join(config.log_dir, self.model_dir, config.model_name)
        self.sample_dir = os.path.join(config.sample_dir, self.model_dir, config.model_name)

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.sample_dir):
             os.makedirs(self.sample_dir)
        self.log_fout = open(os.path.join(self.log_path, 'log_train.txt'), 'a')
        self.log_fout.write(str(config)+'\n')

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        x_ = range(self.output_size)
        y_ = range(self.output_size)
        z_ = range(self.output_size)
        yy, xx, zz = np.meshgrid(x_,y_,z_)
        self.shape_coord = np.concatenate((np.expand_dims(xx, -1), np.expand_dims(yy, -1),np.expand_dims(zz, -1)), -1)

        print("preprocessing - start")

        self.imgout_0 = np.full([self.real_size*4, self.real_size*4*3], 255, np.uint8)

        if os.path.exists("splits/"+self.data_content+"_train.txt"):
            # load data
            fin = open("splits/"+self.data_content+"_train.txt")
            out_vsizes = {}

            self.dataset_names = [name.strip() for name in fin.readlines()]
            fin.close()

            self.dataset_len = len(self.dataset_names)
            self.dataset_len = 4

            self.mask_content  = []
            self.input_content = []
            self.partial_content = []
            self.partial_mask = []
            self.gt_content = []
            self.pos_content = []

            self.names = range(self.dataset_len)
            vox_name = "/model.binvox"

            if config.train_complete and self.mode=='train':
                for i in range(self.dataset_len):
                    print("preprocessing content - "+str(i+1)+"/"+str(self.dataset_len))
                    if not os.path.exists(os.path.join(self.data_dir,self.dataset_names[self.names[i]]+vox_name)):
                        print('non exists', os.path.join(self.data_dir,self.dataset_names[self.names[i]]+vox_name))
                        self.dataset_len = self.dataset_len -1
                        continue

                    tmp_raw = get_vox_from_binvox_1over2(os.path.join(self.data_dir,self.dataset_names[self.names[i]]+vox_name)).astype(np.uint8)
                    xmin,xmax,ymin,ymax,zmin,zmax = self.get_voxel_bbox(tmp_raw)

                    tmp = self.crop_voxel(tmp_raw,xmin,xmax,ymin,ymax,zmin,zmax)
                    gt_voxel = tmp
                    tmp_input, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)

                    partial_shape, partial_mask = self.random_crop(tmp, crop_size=self.csize,c_range=self.c_range)
                    partial_shape =  gaussian_filter(partial_shape.astype(np.float32), sigma=1)
                     

                    self.input_content.append(tmp_input)
                    self.gt_content.append(gt_voxel)
                    self.mask_content.append(tmp_mask)
                    self.partial_content.append(partial_shape)
                    self.partial_mask.append(partial_mask)
                    self.pos_content.append([xmin,xmax,ymin,ymax,zmin,zmax])

                    img_y = i//4
                    img_x = (i%4)*3+2
                    if img_y<4:
                        tmpvox = self.recover_voxel(gt_voxel,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

                    img_y = i//4
                    img_x = (i%4)*3
                    if img_y<4:
                        input_vox = tmp_input
                        tmpvox = self.recover_voxel(input_vox,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

                    img_y = i//4
                    img_x = (i%4)*3 + 1
                    if img_y<4:
                        tmpvox = self.recover_voxel(partial_shape,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

                    cv2.imwrite(self.sample_dir+"/a_content_train.png", self.imgout_0)
        else:
            print("ERROR: cannot load dataset txt: "+"splits/"+self.data_content+"_train.txt")
            exit(-1)

        if os.path.exists("splits/"+self.data_content+"_test.txt"):
            #load data
            fin = open("splits/"+self.data_content+"_test.txt")

            self.test_dataset_names = [name.strip() for name in fin.readlines()]
            fin.close()

            self.test_dataset_len = len(self.test_dataset_names)
            self.test_dataset_len = 2

            self.mask_test  = []
            self.input_test = []
            self.gt_test = []
            self.pos_test = []
            self.partial_test = []

            self.names = range(self.test_dataset_len)

            if config.train_complete:
                for i in range(self.test_dataset_len):
                    print("preprocessing test content - "+str(i+1)+"/"+str(self.test_dataset_len))
                    if not os.path.exists(os.path.join(self.data_dir,self.test_dataset_names[self.names[i]]+vox_name)):
                        print('non exits')
                        self.test_dataset_len = self.test_dataset_len - 1
                        continue
                    tmp_raw = get_vox_from_binvox_1over2(os.path.join(self.data_dir,self.test_dataset_names[self.names[i]]+vox_name)).astype(np.uint8)
                    xmin,xmax,ymin,ymax,zmin,zmax = self.get_voxel_bbox(tmp_raw)
                    crop_locs = None 
                    
                    tmp = self.crop_voxel(tmp_raw,xmin,xmax,ymin,ymax,zmin,zmax)
                    gt_voxel = tmp
                    # gt_voxel= gaussian_filter(tmp.astype(np.float32), sigma=1)
                    tmp_input, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)

                    partial_shape, partial_mask = self.random_crop(tmp, crop_size=self.csize,\
                                    c_range=self.c_range, c_locs=crop_locs)
                    partial_shape =  gaussian_filter(partial_shape.astype(np.float32), sigma=1)

                    # self.im_content.append(im)
                    self.partial_test.append(partial_shape)
                    self.input_test.append(tmp_input)
                    self.gt_test.append(gt_voxel)
                    self.mask_test.append(tmp_mask)
                    self.pos_test.append([xmin,xmax,ymin,ymax,zmin,zmax] )
                    # xmin,xmax,ymin,ymax,zmin,zmax = 64+8, 192-8, 64+8, 192-8, 64+8, 192-8

                    img_y = i//4
                    img_x = (i%4)*3+2
                    if img_y<4:
                        tmpvox = self.recover_voxel(gt_voxel,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

                    img_y = i//4
                    img_x = (i%4)*3
                    if img_y<4:
                        input_vox = tmp_input
                        tmpvox = self.recover_voxel(input_vox,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

                    img_y = i//4
                    img_x = (i%4)*3 + 1
                    if img_y<4:
                        tmpvox = self.recover_voxel(partial_shape,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

                    cv2.imwrite(self.sample_dir+"/a_content_test.png", self.imgout_0)

        self.generator = CoarseCompletor_skip(self.g_dim,)
        self.generator.to(self.device)
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 30, gamma=0.99)

        self.max_to_keep = 2
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir, config.model_name)
        self.checkpoint_name='checkpoint'

        self.checkpoint_manager_list = [None] * self.max_to_keep
        self.checkpoint_manager_pointer = 0

    def log_string(self, out_str, ):
        self.log_fout.write(out_str+'\n')
        self.log_fout.flush()
        print(out_str)


    def get_voxel_input_Dmask_mask(self,vox):

        vox_tensor = torch.from_numpy(vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
        #input
        down_rate = self.output_size//self.input_size

        input_vox = F.max_pool3d(vox_tensor, kernel_size = down_rate, stride = 1, padding = down_rate//2)
        input_vox = input_vox[:,:,:-1,:-1,:-1]
        if self.output_size==256 and down_rate==4:
            input_vox = F.max_pool3d(input_vox, kernel_size = 3, stride = 1, padding = 1)
        #Dmask
        # smallmask_tensor = vox_tensor[:,:,crop_margin:-crop_margin,crop_margin:-crop_margin,crop_margin:-crop_margin]
        smallmask_tensor = F.max_pool3d(vox_tensor, kernel_size = 5, stride = 1, padding = 2)

        #to numpy
        input_in = input_vox.detach().cpu().numpy()[0,0]
        input_in = np.round(input_in).astype(np.uint8)

        mask_in = smallmask_tensor.detach().cpu().numpy()[0,0]
        mask_in = np.round(mask_in).astype(np.uint8)
        return input_in, mask_in

    
    def random_crop(self, vox, crop_size=20, c_range=20, prob=None, c_locs=None):
        vx, vy, vz = vox.shape
        edges = sobel(vox)
        edges = edges.astype(np.float32)/255.0

        csize = crop_size + (np.random.rand(3)-0.5)*c_range
        csize = csize.astype(np.int32)
        csize[0] = min(vx-16, csize[0])
        csize[1] = min(vy-16, csize[1])
        csize[2] = min(vz-16, csize[2])

        loc_starts = np.zeros(3,np.int32)
        new_vox = np.zeros(vox.shape)
        new_vox[:,:,:] = vox[:,:,:].copy()
        loc_starts[0] = np.random.randint(0, vx-csize[0])
        loc_starts[1] = np.random.randint(0, vy-csize[1])
        loc_starts[2] = np.random.randint(0, vz-csize[2])

        if np.array(edges>0.4,np.int32).sum()<100:
            p = 0
        else:
            if prob is None:
                p = np.random.random()
            else:
                p = prob

        if (p<=0.4 or self.data_content!='content_chair' ) and c_locs is None :

            while(vox[loc_starts[0]:loc_starts[0]+csize[0], loc_starts[1]:loc_starts[1]+csize[1],\
                loc_starts[2]:loc_starts[2]+csize[2],].sum()<30):
                csize = crop_size + (np.random.rand(3)-0.5)*c_range
                csize[0] = min(vx-16, csize[0])
                csize[1] = min(vy-16, csize[1])
                csize[2] = min(vz-16, csize[2])
                csize = csize.astype(np.int32)

                loc_starts[0] = np.random.randint(0, vx-csize[0])
                loc_starts[1] = np.random.randint(0, vy-csize[1])
                loc_starts[2] = np.random.randint(0, vz-csize[2])

        elif c_locs is None:
            if self.data_content=='content_chair':
                cnt=0
                while(vox[loc_starts[0]:loc_starts[0]+csize[0], loc_starts[1]:loc_starts[1]+csize[1],\
                    loc_starts[2]:loc_starts[2]+csize[2],].sum()<30 or loc_starts[1]<vy/2):

                    csize = crop_size + (np.random.rand(3)-0.5)*c_range
                    csize[0] = min(vx-16, csize[0])
                    csize[1] = min(vy-16, csize[1])
                    csize[2] = min(vz-16, csize[2])
                    csize = csize.astype(np.int32)

                    loc_starts[0] = np.random.randint(0, vx-csize[0])
                    loc_starts[1] = np.random.randint(0, vy-csize[1])
                    loc_starts[2] = np.random.randint(0, vz-csize[2])

                    cnt+=1
                    if cnt>10:
                        break

        else:
            loc_starts = c_locs[0:3]
            csize = c_locs[3:] - loc_starts

        new_vox[loc_starts[0]:loc_starts[0]+csize[0], loc_starts[1]:loc_starts[1]+csize[1],\
            loc_starts[2]:loc_starts[2]+csize[2],] = 0

        mask_in = np.zeros(vox.shape)
        mask_in[loc_starts[0]:loc_starts[0]+csize[0], loc_starts[1]:loc_starts[1]+csize[1],\
            loc_starts[2]:loc_starts[2]+csize[2],] = 1.0
        return new_vox, mask_in

    def point_cloud_to_volume(self, points, vsize=256, radius=0.5):
        """ input is Nx3 points.
            output is vsize*vsize*vsize
            assumes points are in range [-radius, radius]
        """
        vol = np.zeros((vsize,vsize,vsize))
        voxel = 2*radius/float(vsize)
        locations = (points + radius)/voxel
        locations = np.clip(locations, 0, vsize-1e-4)
        locations = locations.astype(int)
        vol[locations[:,0],locations[:,1],locations[:,2]] = 1.0
        return vol

    def rotate_pc_along_y(self,pc, rot_angle):
        cosval = np.cos(rot_angle)
        sinval = np.sin(rot_angle)
        rotmat = np.array([[cosval, -sinval],[sinval, cosval]])
        pc[:,[0,2]] = np.dot(pc[:,[0,2]], np.transpose(rotmat))
        return pc


    def get_voxel_bbox(self,vox):
        #minimap
        vox_tensor = torch.from_numpy(vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
        #smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size = 2, stride = 2, padding = 0)
        #smallmaskx_tensor = F.interpolate(smallmaskx_tensor, scale_factor = 2, mode='nearest')

        smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size=self.upsample_rate, stride = self.upsample_rate, padding = 0)
        smallmaskx_tensor = F.interpolate(smallmaskx_tensor, scale_factor=self.upsample_rate, mode='nearest')

        smallmaskx = smallmaskx_tensor.detach().cpu().numpy()[0,0]
        smallmaskx = np.round(smallmaskx).astype(np.uint8)
        smallx,smally,smallz = smallmaskx.shape
        #x
        ray = np.max(smallmaskx,(1,2))
        xmin = 0
        xmax = 0
        for i in range(smallx):
            if ray[i]>0:
                if xmin==0:
                    xmin = i
                xmax = i
        #y
        ray = np.max(smallmaskx,(0,2))
        ymin = 0
        ymax = 0
        for i in range(smally):
            if ray[i]>0:
                if ymin==0:
                    ymin = i
                ymax = i
        #z
        ray = np.max(smallmaskx,(0,1))
        if self.asymmetry:
            zmin = 0
            zmax = 0
            for i in range(smallz):
                if ray[i]>0:
                    if zmin==0:
                        zmin = i
                    zmax = i
        else:
            zmin = smallz//2
            zmax = 0
            for i in range(zmin,smallz):
                if ray[i]>0:
                    zmax = i

        return xmin,xmax+1,ymin,ymax+1,zmin,zmax+1

    def get_voxel_mask_exact(self,vox):
        #256 -maxpoolk4s4- 64 -upsample- 256
        vox_tensor = torch.from_numpy(vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
        #input
        smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size = self.upsample_rate, stride = self.upsample_rate, padding = 0)
        #mask
        smallmask_tensor = F.interpolate(smallmaskx_tensor, scale_factor = self.upsample_rate, mode='nearest')
        #to numpy
        smallmask = smallmask_tensor.detach().cpu().numpy()[0,0]
        smallmask = np.round(smallmask).astype(np.uint8)
        return smallmask

    def crop_voxel(self,vox,xmin,xmax,ymin,ymax,zmin,zmax):
        xspan = xmax-xmin
        yspan = ymax-ymin
        zspan = zmax-zmin
        tmp = np.zeros([xspan*self.upsample_rate+self.mask_margin*2,yspan*self.upsample_rate+self.mask_margin*2,zspan*self.upsample_rate+self.mask_margin*2], np.uint8)
        if self.asymmetry:
            tmp[self.mask_margin:-self.mask_margin,self.mask_margin:-self.mask_margin,self.mask_margin:-self.mask_margin] = vox[xmin*self.upsample_rate:xmax*self.upsample_rate,ymin*self.upsample_rate:ymax*self.upsample_rate,zmin*self.upsample_rate:zmax*self.upsample_rate]
        else:
            #note z is special: only get half of the shape in z:  0     0.5-----1
            tmp[self.mask_margin:-self.mask_margin,self.mask_margin:-self.mask_margin,:-self.mask_margin] = vox[xmin*self.upsample_rate:xmax*self.upsample_rate,ymin*self.upsample_rate:ymax*self.upsample_rate,zmin*self.upsample_rate-self.mask_margin:zmax*self.upsample_rate]
        return tmp

    def recover_voxel(self,vox,xmin,xmax,ymin,ymax,zmin,zmax):
        tmpvox = np.zeros([self.real_size,self.real_size,self.real_size], np.float32)
        xmin_,ymin_,zmin_ = (0,0,0)
        xmax_,ymax_,zmax_ = vox.shape
        xmin = xmin*self.upsample_rate-self.mask_margin
        xmax = xmax*self.upsample_rate+self.mask_margin
        ymin = ymin*self.upsample_rate-self.mask_margin
        ymax = ymax*self.upsample_rate+self.mask_margin
        if self.asymmetry:
            zmin = zmin*self.upsample_rate-self.mask_margin
        else:
            zmin = zmin*self.upsample_rate
            zmin_ = self.mask_margin
        zmax = zmax*self.upsample_rate+self.mask_margin
        if xmin<0:
            xmin_ = -xmin
            xmin = 0
        if xmax>self.real_size:
            xmax_ = xmax_+self.real_size-xmax
            xmax = self.real_size
        if ymin<0:
            ymin_ = -ymin
            ymin = 0
        if ymax>self.real_size:
            ymax_ = ymax_+self.real_size-ymax
            ymax = self.real_size
        if zmin<0:
            zmin_ = -zmin
            zmin = 0
        if zmax>self.real_size:
            zmax_ = zmax_+self.real_size-zmax
            zmax = self.real_size
        if self.asymmetry:
            tmpvox[xmin:xmax,ymin:ymax,zmin:zmax] = vox[xmin_:xmax_,ymin_:ymax_,zmin_:zmax_]
        else:
            tmpvox[xmin:xmax,ymin:ymax,zmin:zmax] = vox[xmin_:xmax_,ymin_:ymax_,zmin_:zmax_]
            if zmin*2-zmax-1<0:
                tmpvox[xmin:xmax,ymin:ymax,zmin-1::-1] = vox[xmin_:xmax_,ymin_:ymax_,zmin_:zmax_]
            else:
                tmpvox[xmin:xmax,ymin:ymax,zmin-1:zmin*2-zmax-1:-1] = vox[xmin_:xmax_,ymin_:ymax_,zmin_:zmax_]
        return tmpvox

    def reshape_to_size_torch(self, v1, v2_shape):
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

    def load(self):
        #load previous checkpoint
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            print('loading from', model_dir)
            fin.close()
            checkpoint = torch.load(model_dir)
            self.generator.load_state_dict(checkpoint['generator'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.start_epoch = checkpoint['epoch'] + 1
            print(" [*] Load SUCCESS")
            return True
        else:
            print(" [!] Load failed...")
            return False

    def save(self,epoch):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        save_dir = os.path.join(self.checkpoint_path,self.checkpoint_name+"-"+str(epoch)+".pth")
        self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer+1)%self.max_to_keep
        #delete checkpoint
        if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
            if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
                os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])
        #save checkpoint
        torch.save({
            'generator': self.generator.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch
            }, save_dir)

        #update checkpoint manager
        self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
        #write file
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        fout = open(checkpoint_txt, 'w')
        for i in range(self.max_to_keep):
            pointer = (self.checkpoint_manager_pointer+self.max_to_keep-i)%self.max_to_keep
            if self.checkpoint_manager_list[pointer] is not None:
                fout.write(self.checkpoint_manager_list[pointer]+"\n")
        fout.close()

    def compute_iou(self, recons, gt, thresh=0.4):
       x = (recons>thresh).float()
       y = (gt>thresh).float()
       intersection = torch.sum(x*y, dim=(1,2,3,4))
       union = torch.sum(x, dim=(1,2,3,4)) + torch.sum(y, dim=(1,2,3,4)) - torch.sum(x*y, dim=(1,2,3,4))
       iou = (intersection/(union+1e-5)).mean()
       return iou

    @property
    def model_dir(self):
        return "{}_complete".format(self.data_content)
    

    def train(self, config):
        if config.continue_train or self.mode=='test':
            self.load()
        if self.mode=='test':
            _ = self.eval_one_epoch(self.start_epoch-1, config)
            return
        best_iou = 0.0
        self.imgout_0 = np.full([self.real_size*4, self.real_size*4*4], 255, np.uint8)

        start_time = time.time()
        training_epoch = config.epoch

        batch_index_list = np.arange(self.dataset_len)

        for epoch in range(self.start_epoch,  training_epoch):
            np.random.shuffle(batch_index_list)

            total_loss = 0.0
            total_iou = 0.0
            self.generator.train()

            for idx in range(self.dataset_len):
                # ready a fake image
                dxb = batch_index_list[idx]
                mask_in =  torch.from_numpy(self.mask_content[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                input_in = torch.from_numpy(self.input_content[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                gt_train = torch.from_numpy(gaussian_filter(self.gt_content[dxb].astype(np.float32),sigma=1)\
                        ).to(self.device).unsqueeze(0).unsqueeze(0).float()

                partial_shape, partial_mask = self.random_crop(self.gt_content[dxb],crop_size=self.csize,c_range=self.c_range)
                partial_shape =  gaussian_filter(partial_shape.astype(np.float32), sigma=1)

                partial_in = torch.from_numpy(partial_shape).to(self.device).unsqueeze(0).unsqueeze(0).float()
                mask_partial_in = torch.from_numpy(partial_mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
                partial_in = self.reshape_to_size_torch(partial_in, (self.output_size,self.output_size,self.output_size))
                mask_partial_in = self.reshape_to_size_torch(mask_partial_in, (self.output_size,self.output_size,self.output_size))
                gt_train = self.reshape_to_size_torch(gt_train, (self.output_size,self.output_size,self.output_size))

                input_in =  self.reshape_to_size_torch(input_in, (self.output_size,self.output_size,self.output_size))

                self.optimizer.zero_grad()

                voxel_out = self.generator(partial_in, is_training=True)

                down_rate = self.output_size//self.input_size
                mask_vox = mask_partial_in

                loss_r_masked = -(self.w_posi*(input_in*mask_vox*torch.log(voxel_out+1e-5)\
                             ).sum()/(mask_vox.sum()+1e-5) + ((1-input_in)*mask_vox*torch.log(1-\
                                 voxel_out+1e-5)).sum()/(mask_vox.sum()+1e-5))

                loss_r_all = -(self.w_posi*(input_in*torch.log(voxel_out + 1e-5)).sum()/input_in.sum() + \
                            ((1-input_in)*torch.log(1-voxel_out+1e-5)).sum()/(1-input_in).sum())
                loss_r = loss_r_all + self.w_mask*loss_r_masked 
                iou_train = self.compute_iou(voxel_out, input_in)
                
                total_loss = total_loss + loss_r.item()
                total_iou = total_iou + iou_train.item()

                loss_r.backward()
                self.optimizer.step()

                if epoch%20==0:

                    img_y = dxb//4
                    img_x = (dxb%4)*4
                    if img_y<4:
                        input_in = self.reshape_to_size_torch(input_in, self.gt_content[dxb].shape)
                        tmp_voxel_fake = input_in.detach().cpu().numpy()[0,0]
                        xmin,xmax,ymin,ymax,zmin,zmax = self.pos_content[dxb]

                        tmpvox = self.recover_voxel(tmp_voxel_fake,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

                    img_y = dxb//4
                    img_x = (dxb%4)*4+1
                    if img_y<4:
                        tmp_voxel_fake = partial_shape
                        xmin,xmax,ymin,ymax,zmin,zmax = self.pos_content[dxb]
                        tmpvox = self.recover_voxel(tmp_voxel_fake,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

                    img_y = dxb//4
                    img_x = (dxb%4)*4+2
                    if img_y<4:
                        xmin,xmax,ymin,ymax,zmin,zmax = self.pos_content[dxb]
                        voxel_out = self.reshape_to_size_torch(voxel_out, self.gt_content[dxb].shape)
                        tmp_voxel_fake = voxel_out.detach().cpu().numpy()[0,0]
                        tmpvox = self.recover_voxel(tmp_voxel_fake,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

                    img_y = dxb//4
                    img_x = (dxb%4)*4+3
                    if img_y<4:
                        tmp_voxel_fake = self.gt_content[dxb]
                        xmin,xmax,ymin,ymax,zmin,zmax = self.pos_content[dxb]
                        tmpvox = self.recover_voxel(tmp_voxel_fake,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)


            self.log_string("Epoch: [%d/%d] time: %.0f, train_loss_r: %.5f, loss_mask: %.5f,train_iou: %.4f," % (epoch, training_epoch, \
                time.time() - start_time,  loss_r_all.item(),  loss_r_masked.item(), total_iou/self.dataset_len,))
            #self.logger.add_scalars('total train', {'total_loss_l2': total_loss/self.dataset_len, 'total_iou': total_iou/self.dataset_len},total_steps)

            if epoch%20==0:
                cv2.imwrite(self.sample_dir+"/train_"+str(epoch)+"_0.png", self.imgout_0)

            if epoch%self.eval_epoch==0:
                eval_iou = self.eval_one_epoch(epoch,config)
                if eval_iou > best_iou and epoch%self.save_epoch==0:
                    self.save(epoch)

            self.scheduler.step()

        self.save(epoch)

    def eval_one_epoch(self, epoch, config):

        start_time = time.time()
        total_loss = 0.0
        total_iou = 0.0
        total_num=0

        self.imgout_0 = np.full([self.real_size*4, self.real_size*4*4], 255, np.uint8)
        
        if self.test_dataset_len==0:
            return 0
        self.generator.eval()

        for idx in range(self.test_dataset_len):
            total_steps = epoch*self.test_dataset_len + idx

            #ready a fake image
            dxb = idx
            mask_in =  torch.from_numpy(self.mask_test[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_in = torch.from_numpy(self.input_test[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
            partial_in = torch.from_numpy(self.partial_test[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
            gt_test = torch.from_numpy(gaussian_filter(self.gt_test[dxb].astype(np.float32),sigma=1)).to(self.device).unsqueeze(0).unsqueeze(0).float()

            partial_shape, partial_mask = self.random_crop(self.gt_test[dxb], crop_size=self.csize,c_range=self.c_range,prob=1)
            partial_shape =  gaussian_filter(partial_shape.astype(np.float32), sigma=1)

            partial_in = torch.from_numpy(partial_shape).to(self.device).unsqueeze(0).unsqueeze(0).float()
            mask_partial_in = torch.from_numpy(partial_mask).to(self.device).unsqueeze(0).unsqueeze(0).float()

            partial_in = self.reshape_to_size_torch(partial_in, (self.output_size,self.output_size,self.output_size))
            gt_test = self.reshape_to_size_torch(gt_test, (self.output_size,self.output_size,self.output_size))
            input_in =  self.reshape_to_size_torch(input_in, (self.output_size,self.output_size,self.output_size))

            with torch.no_grad():
                voxel_out = self.generator(partial_in, is_training=False)
                loss_r = -(self.w_posi*(input_in*torch.log(voxel_out+1e-5)).mean() + ((1-input_in)*torch.log(1-voxel_out+1e-5)).mean())
                iou_test = self.compute_iou(voxel_out, input_in)

                total_loss = total_loss + loss_r.item()
                total_iou = total_iou + iou_test.item()
                pad_out = voxel_out.detach().cpu()
                out_shape = voxel_out.detach().cpu().numpy()[0,0]

 
            img_y = dxb//4
            img_x = (dxb%4)*4
            if img_y<4:
                input_in = self.reshape_to_size_torch(input_in, self.gt_test[dxb].shape)
                tmp_voxel_fake = input_in.detach().cpu().numpy()[0,0]
                xmin,xmax,ymin,ymax,zmin,zmax = self.pos_test[dxb]

                tmpvox = self.recover_voxel(tmp_voxel_fake,xmin,xmax,ymin,ymax,zmin,zmax)
                self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

            img_y = dxb//4
            img_x = (dxb%4)*4+1
            if img_y<4:
                xmin,xmax,ymin,ymax,zmin,zmax = self.pos_test[dxb]
                tmp_voxel_fake = self.reshape_to_size_torch(partial_in, self.gt_test[dxb].shape)
                tmp_voxel_fake = tmp_voxel_fake.detach().cpu().numpy()[0,0]

                tmpvox = self.recover_voxel(tmp_voxel_fake,xmin,xmax,ymin,ymax,zmin,zmax)
                self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

            img_y = dxb//4
            img_x = (dxb%4)*4+2
            if img_y<4:
                voxel_out = self.reshape_to_size_torch(voxel_out, self.gt_test[dxb].shape)
                tmp_voxel_fake = voxel_out.detach().cpu().numpy()[0,0]

                tmpvox = self.recover_voxel(tmp_voxel_fake,xmin,xmax,ymin,ymax,zmin,zmax)
                self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

            img_y = dxb//4
            img_x = (dxb%4)*4+3
            if img_y<4:
                #tmp_voxel_fake = gt_test.detach().cpu().numpy()[0,0]
                tmp_voxel_fake = self.gt_test[dxb]
                tmpvox = self.recover_voxel(tmp_voxel_fake,xmin,xmax,ymin,ymax,zmin,zmax)
                self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)
        
        cv2.imwrite(self.sample_dir+"/eval_"+str(epoch)+".png", self.imgout_0)

        self.log_string("[eval] Epoch: [%d/%d] time: %.0f, eval_loss_r: %.6f, eval_iou: %.4f," % (epoch, config.epoch, time.time() - start_time,  total_loss/self.test_dataset_len, total_iou/self.test_dataset_len))


        return total_iou/self.test_dataset_len
