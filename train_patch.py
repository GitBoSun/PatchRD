import os
import time
import math
import random
import numpy as np
import cv2
import h5py
from glob import glob
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import sobel

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from utils import *
from models import *

import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D
chamLoss = ChamferDistancePytorch.chamfer3D.dist_chamfer_3D.chamfer_3DDist()

class MODEL_PATCH(object):
    def __init__(self, config):
        self.real_size = 256
        #self.real_size = 128
        self.mask_margin = 8

        self.g_dim = config.g_dim
        self.z_dim = config.z_dim
        self.input_size = config.input_size # 32
        self.output_size = config.output_size # 128
        self.patch_size = config.patch_size
        self.csize = config.csize
        self.c_range = config.c_range

        self.stride = 4
        self.pt_num = 768
        self.K = config.K
        self.max_sample_num = config.max_sample_num
        self.w_ident = config.w_ident

        # self.up_rate = self.output_size//self.input_size
        self.upsample_rate = 1
        self.asymmetry = True 
        self.save_epoch = 5
        self.eval_epoch = 2
        self.start_epoch = 0

        self.mode = config.mode
        self.gua_filter = True
        self.small_dataset = config.small_dataset 
        self.dump_deform =config.dump_deform
        self.top_nums = [8]

        self.pre_fix = ''
        self.pred_coarse = True

        self.dis_thres= 0.035
        self.sampling_threshold = 0.3
        self.pt_sampling_threshold = 0.3

        self.render_view_id = 0
        self.voxel_renderer = voxel_renderer(self.real_size)

        self.checkpoint_dir = config.checkpoint_dir
        self.data_dir = config.data_dir

        self.dump_deform_path = config.dump_deform_path

        if self.dump_deform and not os.path.exists(self.dump_deform_path):
            os.makedirs(self.dump_deform_path)

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

        x_ = range(self.patch_size)
        y_ = range(self.patch_size)
        z_ = range(self.patch_size)
        yy, xx, zz = np.meshgrid(x_,y_,z_)
        self.patch_coord = np.concatenate((np.expand_dims(xx, -1), np.expand_dims(yy, -1),np.expand_dims(zz, -1)), -1)

        #load data
        print("preprocessing - start")

        self.imgout_0 = np.full([self.real_size*4, self.real_size*4*3], 255, np.uint8)

        if os.path.exists("splits/"+self.data_content+"_train.txt"):
            #load data
            fin = open("splits/"+self.data_content+"_%s.txt"%(self.mode))
            
            self.dataset_names = [name.strip() for name in fin.readlines()]
            fin.close()

            self.dataset_len = len(self.dataset_names)
            self.dataset_len = 1000
            
            if self.mode=='test':
                self.dataset_len = 200
            
            if self.dump_deform and self.mode=='train': 
                fin2 = open("splits/"+self.data_content+"_test.txt")
                test_names = [name.strip() for name in fin2.readlines()]
                fin2.close()
                self.dataset_names = self.dataset_names + test_names[:200]
                self.dataset_len += 200 

            if self.small_dataset:
                self.dataset_len = 4 
            

            self.mask_content  = []
            self.input_content = []
            self.partial_content = []
            self.partial_mask = []
            self.gt_content = []
            self.pos_content = []
            self.plane_param = []
            self.crop_locs=[]

            self.pcn_partial = []
            self.pcn_gt = []
            self.gt_pc = []
            self.shape_names = []
            self.partial_or = []
            self.pcn_pt_ids = []
            self.pcn_rand_ids = []
            # self.im_content = []

            self.names = range(self.dataset_len)

            vox_name = "/model.binvox"

            if config.train_patch:
                for i in range(self.dataset_len):
                    print("preprocessing content - "+str(i+1)+"/"+str(self.dataset_len))

                    if not os.path.exists(os.path.join(self.data_dir,self.dataset_names[self.names[i]]+vox_name)):
                        print('non exists')
                        self.dataset_len = self.dataset_len-1
                        continue
                        
                    tmp_raw = get_vox_from_binvox_1over2(os.path.join(self.data_dir,self.dataset_names[self.names[i]]+vox_name)).astype(np.uint8)
                    xmin,xmax,ymin,ymax,zmin,zmax = self.get_voxel_bbox(tmp_raw)

                    tmp = self.crop_voxel(tmp_raw,xmin,xmax,ymin,ymax,zmin,zmax)
                    gt_voxel = tmp
                    tmp_input, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)
                    
                    if self.mode=='test' or self.dump_deform : prob=1
                    else: prob = None
                    partial_shape, partial_mask, c_locs = self.random_crop(tmp, crop_size=self.csize, \
                                c_range=self.c_range, prob=prob,return_locs=True)
                    partial_mask = tmp_mask
                    self.crop_locs.append(c_locs)

                    self.shape_names.append(self.dataset_names[self.names[i]])

                    if self.gua_filter:
                        tmp_input = gaussian_filter(tmp_input.astype(np.float32), sigma=1)
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
                        # xmin,xmax,ymin,ymax,zmin,zmax = self.pos_content[i]
                        tmpvox = self.recover_voxel(gt_voxel,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)
                    img_y = i//4
                    img_x = (i%4)*3
                    if img_y<4:
                        tmpvox = self.recover_voxel(tmp_input,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)
                    img_y = i//4
                    img_x = (i%4)*3 + 1
                    if img_y<4:
                        tmpvox = self.recover_voxel(partial_shape,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)
                cv2.imwrite(self.sample_dir+"/a_content_%s.png"%(self.mode), self.imgout_0)
        else:
            print("ERROR: cannot load dataset txt: "+"splits/"+self.data_content+"_train.txt")
            exit(-1)

        self.c_encoder = PatchEncoder(self.g_dim, self.z_dim)
        self.d_encoder = PatchEncoder(self.g_dim, self.z_dim)
        self.c_encoder.to(self.device)
        self.d_encoder.to(self.device)

        c_params = list(self.c_encoder.parameters())
        d_params = list(self.d_encoder.parameters())

        self.optimizer = torch.optim.Adam(c_params+d_params, lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 30, gamma=0.99)
            
        if self.pred_coarse:
            print('coarse predictor', self.data_content)
            if self.data_content =='content_chair':
                self.dis_thres = 0.035 # 0.055 
                checkpoint = torch.load('checkpoint_complete/content_chair_complete/coarse_comp/checkpoint-90.pth')
            elif self.data_content=='content_boat':
                self.dis_thres=0.12
                checkpoint = torch.load('checkpoint_complete/content_boat_complete/coarse_comp/checkpoint-90.pth')
            elif self.data_content=='content_lamp':
                self.dis_thres=0.04
                checkpoint = torch.load('checkpoint_complete/content_lamp_complete/coarse_comp/checkpoint-90.pth')
            elif self.data_content=='content_couch':
                self.dis_thres=0.12
                checkpoint = torch.load('checkpoint_complete/content_couch_complete/coarse_comp/checkpoint-150.pth')
            elif self.data_content=='content_cabinet':
                self.dis_thres=0.045
                checkpoint = torch.load('checkpoint_complete/content_cabinet_complete/coarse_comp/checkpoint-90.pth')
            elif self.data_content=='content_plane':
                self.dis_thres = 0.045
                checkpoint = torch.load('checkpoint_complete/content_plane_complete/coarse_comp/checkpoint-90.pth')
            elif self.data_content=='content_car':
                self.dis_thres=0.05
                checkpoint = torch.load('checkpoint_complete/content_car_complete/coarse_comp/checkpoint-90.pth') #
            elif self.data_content=='content_table':
                self.dis_thres = 0.045
                checkpoint = torch.load('checkpoint_complete/content_table_complete/coarse_comp/checkpoint-90.pth')

            self.coarse_predictor = CoarseCompletor_skip(32)
            self.coarse_predictor.to(self.device)
            self.coarse_predictor.load_state_dict(checkpoint['generator'])
            self.coarse_predictor.eval()

        self.max_to_keep = 3
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
        if self.output_size==128: pool_rate=3
        else: pool_rate = 5

        if down_rate==2:
            smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size = down_rate, stride = down_rate, padding = 0)
            input_vox = F.interpolate(smallmaskx_tensor, scale_factor = down_rate, mode='nearest')

        else:
            input_vox = F.max_pool3d(vox_tensor, kernel_size = down_rate, stride = 1, padding = down_rate//2)
            input_vox = input_vox[:,:,:-1,:-1,:-1]

        #Dmask
        # smallmask_tensor = vox_tensor[:,:,crop_margin:-crop_margin,crop_margin:-crop_margin,crop_margin:-crop_margin]
        smallmask_tensor = F.max_pool3d(input_vox, kernel_size = self.patch_size, stride = self.stride, padding = 0)

        #to numpy
        input_in = input_vox.detach().cpu().numpy()[0,0]
        input_in = np.round(input_in).astype(np.uint8)

        mask_in = smallmask_tensor.detach().cpu().numpy()[0,0]
        mask_in = np.round(mask_in).astype(np.uint8)
        return input_in, mask_in

    def random_crop(self, vox, crop_size=20, c_range=20, prob=None, return_locs=False ):
        vx, vy, vz = vox.shape
        edges = sobel(vox)
        edges = edges.astype(np.float32)/255.0

        csize = crop_size + (np.random.rand(3)-0.5)*c_range
        csize[0] = min(vx-16, csize[0])
        csize[1] = min(vy-16, csize[1])
        csize[2] = min(vz-16, csize[2])

        csize = csize.astype(np.int32)
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

        if p<0.4 or not self.data_content=='content_chair':
            cnt=0
            while(vox[loc_starts[0]:loc_starts[0]+csize[0], loc_starts[1]:loc_starts[1]+csize[1],\
                loc_starts[2]:loc_starts[2]+csize[2],].sum()<30 ):
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
            cnt=0
            while(vox[loc_starts[0]:loc_starts[0]+csize[0], loc_starts[1]:loc_starts[1]+csize[1],\
                loc_starts[2]:loc_starts[2]+csize[2],].sum()<50 or loc_starts[1]<vy/2):

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
        
        new_vox[loc_starts[0]:loc_starts[0]+csize[0], loc_starts[1]:loc_starts[1]+csize[1],\
            loc_starts[2]:loc_starts[2]+csize[2],] = 0

        mask_in = np.zeros(vox.shape)
        mask_in[loc_starts[0]:loc_starts[0]+csize[0], loc_starts[1]:loc_starts[1]+csize[1],\
            loc_starts[2]:loc_starts[2]+csize[2],] = 1.0
        if return_locs==True:
            c_locs = np.zeros(6, np.int32)
            c_locs[0:3] = loc_starts
            c_locs[3:] = loc_starts+csize
            return new_vox, mask_in,c_locs
        else:
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


    def get_voxel_bbox(self,vox):
        #minimap
        vox_tensor = torch.from_numpy(vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
        smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size = self.upsample_rate, stride = self.upsample_rate, padding = 0)
        smallmaskx_tensor = F.interpolate(smallmaskx_tensor, scale_factor = self.upsample_rate, mode='nearest')

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

    def pad_to_same_size(self, v1, v2):
        _,_,x1,y1,z1 = v1.shape
        _,_,x2,y2,z2 = v2.shape
        x = max(x1, x2)
        y = max(y1, y2)
        z = max(z1, z2)
        new_v1 = F.pad(v1, (int((z-z1)/2), z-z1-int((z-z1)/2), int((y-y1)/2), y-y1-int((y-y1)/2), int((x-x1)/2), x-x1-int((x-x1)/2)))
        new_v2 = F.pad(v2, (int((z-z2)/2), z-z2-int((z-z2)/2), int((y-y2)/2), y-y2-int((y-y2)/2), int((x-x2)/2), x-x2-int((x-x2)/2)))
        return new_v1, new_v2

    def reshape_to_patch_size(self, v1, v2):
        x1,y1,z1 = v1.shape
        x,y,z = v2.shape

        new_v1 = v1
        padding = np.zeros(6)
        if z1 < z:
            padding[0], padding[1] = int((z-z1)/2), z-z1-int((z-z1)/2)
        else:
            new_v1 = new_v1[:, :,int((z1-z)/2):int((z1-z)/2)+z ]

        if y1 < y:
            padding[2], padding[3] = int((y-y1)/2), y-y1-int((y-y1)/2)
        else:
            new_v1 = new_v1[:, int((y1-y)/2):int((y1-y)/2)+y,: ]
        if x1 < x:
            padding[4], padding[5] = int((x-x1)/2), x-x1-int((x-x1)/2)
        else:
            new_v1 = new_v1[int((x1-x)/2):int((x1-x)/2)+x,:,: ]

        new_v1 = torch.from_numpy(new_v1,).unsqueeze(0).unsqueeze(0)

        new_v1 = F.pad(new_v1, tuple(padding.astype(np.int8)))

        return new_v1.numpy()[0,0]

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

    def get_exact_partial_mask(self, partial_in, gt_in, p_mask_in):
        partial_in = (partial_in>0.3).float()
        vox_dif = torch.abs(gt_in - partial_in)
        patch_sum = F.avg_pool3d(partial_in, kernel_size = self.patch_size, stride = self.stride, padding = 0)
        patch_sum = patch_sum*float(self.patch_size**3)

        patch_dif = F.avg_pool3d(vox_dif, kernel_size = self.patch_size, stride = self.stride, padding = 0)
        patch_dif = patch_dif*float(self.patch_size**3)
        #patch_dif = (patch_dif<220).float()
        patch_dif = ((patch_dif/(patch_sum+1e-3))<0.8).float()
        mask_final = p_mask_in*patch_dif
        return mask_final

    def load(self):
        #load previous checkpoint
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        print(self.checkpoint_path)
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            print('loading from', model_dir)
            fin.close()
            checkpoint = torch.load(model_dir)
            self.c_encoder.load_state_dict(checkpoint['c_encoder'])
            self.d_encoder.load_state_dict(checkpoint['d_encoder'])
            if 'optimizer' in checkpoint.keys():
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
            'c_encoder': self.c_encoder.state_dict(),
            'd_encoder': self.d_encoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch
            }, save_dir)

        # update checkpoint manager
        self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
        #write file
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        fout = open(checkpoint_txt, 'w')
        for i in range(self.max_to_keep):
            pointer = (self.checkpoint_manager_pointer+self.max_to_keep-i)%self.max_to_keep
            if self.checkpoint_manager_list[pointer] is not None:
                fout.write(self.checkpoint_manager_list[pointer]+"\n")
        fout.close()

    def filter_feat(self, feat, mask):
        _, _, x_size, y_size, z_size = mask.shape
        x_ = range(x_size)
        y_ = range(y_size)
        z_ = range(z_size)
        feats = feat[0].permute(1,2,3,0)
        yy, xx, zz = np.meshgrid(y_,x_,z_)
        feat_coord = np.concatenate((np.expand_dims(xx, -1), np.expand_dims(yy, -1),np.expand_dims(zz, -1)), -1)
        filtered_feat = feats[mask[0,0]>0]
        filtered_coord = feat_coord[mask[0,0].cpu().numpy()>0]
        return filtered_feat,filtered_coord

    def distance_retrieve(self, c_feat, d_feat, K=20,sigma=1.0, c_max_num=200, d_max_num=200):
        """
        feat: [1, nf, z_dim]
        return:
            c_ids, d_ids, distance matrix
        """

        c_num = min(c_feat.shape[0], c_max_num)
        d_num = min(d_feat.shape[0], d_max_num)

        c_ids = np.array(sorted(np.random.randint(0, c_feat.shape[0], c_num)))
        d_ids = np.array(sorted(np.random.randint(0, d_feat.shape[0], d_num)))

        c_feat_chosen =  c_feat[c_ids,:].detach().cpu()
        d_feat_chosen =  d_feat[d_ids,:].detach().cpu()
        c_feat_chosen = c_feat_chosen.view(c_num, 1, -1)
        d_feat_chosen = d_feat_chosen.view(1, d_num, -1)

        dis_matrix = torch.sqrt(torch.sum(torch.square(c_feat_chosen - d_feat_chosen), -1)) #[c_num, d_num]
        dis_prob = dis_matrix / sigma
        dis_prob = torch.exp(-(dis_prob - torch.max(dis_prob, dim=1).values))
        dis_prob = dis_prob.numpy()
        dis_prob = dis_prob/(dis_prob.sum(1)+1e-7)

        #dis_prob[:,0] = dis_prob[:,0] + 1- dis_prob.sum(1)
        #dis_prob[:,0] = dis_prob[:,0] + 1- dis_prob.sum(1)

        out_ids = np.zeros((c_num, K))
        for i in range(c_num):
            if dis_prob[i].sum()<1e-5:
                out_ids[i] = np.random.randint(0, d_num, size=K)
                continue
            if not dis_prob[i].sum()<1000:
                out_ids[i] = np.random.randint(0, d_num, size=K)
                continue

            if dis_prob[i].sum()!=1:
                dis_prob[i] = dis_prob[i]/dis_prob[i].sum()
                #print(dis_prob[i].sum())

            tmp_d_ids = np.random.choice(np.arange(d_num), size=K, p=dis_prob[i])
            tmp_d_ids = tmp_d_ids.astype(np.int32)
            out_ids[i] = d_ids[tmp_d_ids]
        return c_ids, out_ids.astype(np.int32)

    def random_retrieve(self,num_c, num_d, K=20, c_max_num=200,):
        c_num = min(num_c, c_max_num)
        c_ids = sorted(np.random.randint(0, num_c, c_num))
        d_ids = np.zeros((c_num, K))
        for i in range(c_num):
            d_ids[i] = np.random.randint(0,num_d, size=K)
        return c_ids, d_ids

    def compute_feat_dis(self,c_feats_less, d_feats_less, re_c_ids, re_d_ids):
        c_feats_less = c_feats_less[re_c_ids]
        out_dis = torch.zeros(re_d_ids.shape).to(self.device)

        for i in range(re_d_ids.shape[1]):
            d_feat_tmp = d_feats_less[re_d_ids[:,i]]
            out_dis[:,i] = torch.sqrt(torch.sum(torch.square(c_feats_less - d_feat_tmp), -1))

        return  out_dis

    def regression_loss(self, embedding_distance, actual_distance, mask=None,obj_sigmas=1.0):
        if mask is not None:
            loss = torch.sum(torch.square(mask*(embedding_distance-actual_distance)),)/(mask.sum()+1e-4)
        else:
            loss = torch.mean(torch.square(embedding_distance-actual_distance),)
        return loss

    def compute_iou(self, recons, gt, thresh=0.4):
        x = (recons>thresh).astype(np.float32)

        y = (gt>thresh).astype(np.float32)

        intersection = np.sum(x*y)
        union = np.sum(x) + np.sum(y) - np.sum(x*y)
        iou = intersection/(union+1e-5)
        return iou

    def get_ids_with_gt(self, c_locs, total_d_locs):
         c_num = c_locs.shape[0]
         valid_ids = []
         for i in range(c_num):
             dif = np.abs(total_d_locs - c_locs[i,]).sum(1).min()
             if dif==0:
                 valid_ids.append(i)
         return valid_ids

    @property
    def model_dir(self):
        return "{}_retriv".format(self.data_content)

    def train(self, config):
        
        if self.dump_deform:
            self.mode = 'test'

        if config.continue_train or self.mode=='test':
            self.load()

        start_time = time.time()
        training_epoch = config.epoch
        
        if self.mode=='test':
            training_epoch += 1

        batch_index_list = np.arange(self.dataset_len)

        for epoch in range(self.start_epoch,  training_epoch):
            if self.mode!='test':
                np.random.shuffle(batch_index_list)

            total_loss = 0.0
            total_iou = 0.0
            total_c_iou = 0.0

            total_cd = 0.0

            self.c_encoder.train()
            self.d_encoder.train()

            if self.mode=='test':
                self.c_encoder.eval()
                self.d_encoder.eval()
                print('model eval..')

            #self.dataset_len = len(self.input_content)

            for idx in range(self.dataset_len):
                dxb = batch_index_list[idx]
                qxp = dxb

                if not self.pred_coarse:
                    input_in = torch.from_numpy(self.input_content[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                    mask_in =  torch.from_numpy(self.mask_content[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()

                gt_in = torch.from_numpy(self.gt_content[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()

                if self.mode=='test':
                    partial_shape = self.partial_content[qxp] 
                    partial_mask  =  self.partial_mask[qxp] 
                else:
                    partial_shape, partial_mask = self.random_crop(self.gt_content[dxb],crop_size=self.csize, \
                            c_range=self.c_range)
                    partial_mask = self.mask_content[dxb]

                    if self.gua_filter:
                        partial_shape =  gaussian_filter(partial_shape.astype(np.float32), sigma=1)


                partial_in = torch.from_numpy(partial_shape).to(self.device).unsqueeze(0).unsqueeze(0).float()
                mask_partial_in = torch.from_numpy(partial_mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
                

                if self.pred_coarse:
                    if self.mode=='test': 
                        partial_tmp = self.partial_content[dxb]
                    else:
                        partial_tmp, partial_mask_tmp = self.random_crop(self.gt_content[dxb],crop_size=self.csize, \
                                c_range=self.c_range)
                        if self.gua_filter:
                            partial_tmp = gaussian_filter(partial_tmp.astype(np.float32), sigma=1)
                    
                    partial_tmp_in = torch.from_numpy(partial_tmp).to(self.device).unsqueeze(0).unsqueeze(0).float()
                    partial_mask_2 = F.max_pool3d((partial_tmp_in>self.sampling_threshold).float(), \
                            kernel_size = self.patch_size, stride = self.stride, padding = 0)
                    partial_full = self.reshape_to_size_torch(partial_tmp_in, (self.output_size,self.output_size,self.output_size))
                    
                    partial_in = partial_tmp_in
                    mask_partial_in = partial_mask_2

                    coarse_out = self.coarse_predictor(partial_full)
                    coarse_out = coarse_out.detach()
                    
                    input_in = self.reshape_to_size_torch(coarse_out, partial_shape.shape)
                    mask_in = F.max_pool3d((input_in>self.sampling_threshold).float(), kernel_size = self.patch_size, \
                            stride = self.stride, padding = 0)

                self.optimizer.zero_grad()

                c_feat = self.c_encoder(input_in,is_training=True)
                d_feat = self.d_encoder(partial_in,is_training=True)

                # 1. choose coarse, detailed pairs, compute feature distance
                # 2. compute gt distance
                # 3. compute loss

                c_feats_less, c_locs = self.filter_feat(c_feat, mask_in)
                d_feats_less, d_locs = self.filter_feat(d_feat, mask_partial_in)
                

                for top_num in self.top_nums:
                    if self.dump_deform and (epoch==self.start_epoch or epoch%20==0):

                        out_ids, out_dis, out_trans = self.retrieve_to_full_shape_topk(c_feats_less, \
                                c_locs, d_feats_less, d_locs, gt_in, partial_in,input_in, \
                                dxb, coarse=False, return_k=True, top_num=top_num, target_pt_num=self.pt_num)

                        out_ids_c, out_dis_c, out_trans_c = self.retrieve_to_full_shape_topk(c_feats_less, c_locs, \
                                d_feats_less, d_locs, gt_in, partial_in,input_in, dxb,coarse=True, return_k=True,\
                                top_num=top_num, sigma=1, target_pt_num=self.pt_num)

                        out_path = os.path.join(self.dump_deform_path, self.dataset_names[self.names[dxb]],)
                        crop_locs = self.crop_locs[dxb]

                        if not os.path.exists(out_path):
                            os.makedirs(out_path)
                        
                        if self.pred_coarse:
                            coarse_shape = input_in.cpu().numpy()[0,0]
                        else:
                            coarse_shape = 0

                        np.savez(os.path.join(out_path, 'dump_deform_wc_top%d_%d.npz'%(top_num, epoch)), re_ids=out_ids, \
                                re_dis=out_dis, re_trans=out_trans, re_ids_c=out_ids_c, re_dis_c=out_dis_c, \
                                re_trans_c=out_trans_c, \
                                coarse_shape=coarse_shape,  crop_locs=crop_locs)
                        if dxb>20 and self.dataset_len>50:
                            continue

                    if self.dump_deform:
                        out_path_im = self.dump_deform_path
                    else:
                        out_path_im = self.sample_dir

                    if self.mode=='test' and (epoch==self.start_epoch or epoch%20==0):

                        if self.pred_coarse:
                            imgout_0 = np.full([self.real_size, self.real_size*7], 255, np.uint8)
                        else:
                            imgout_0 = np.full([self.real_size, self.real_size*5], 255, np.uint8)


                        xmin,xmax,ymin,ymax,zmin,zmax = self.pos_content[dxb]

                        if self.pred_coarse:

                            tmpvox = self.recover_voxel(partial_shape, xmin,xmax,ymin,ymax,zmin,zmax)
                            imgout_0[:,(-2)*self.real_size:-self.real_size] = self.voxel_renderer.render_img(tmpvox, \
                                    self.sampling_threshold, self.render_view_id)
                            tmpvox = self.recover_voxel(input_in.cpu().numpy()[0][0], xmin,xmax,ymin,ymax,zmin,zmax)
                            imgout_0[:,-self.real_size:] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, \
                                    self.render_view_id)

                        tmpvox = self.recover_voxel(self.input_content[dxb],xmin,xmax,ymin,ymax,zmin,zmax)
                        imgout_0[:,0:self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

                        tmpvox = self.recover_voxel(partial_in.cpu().numpy()[0][0],xmin,xmax,ymin,ymax,zmin,zmax)
                        imgout_0[:,self.real_size:self.real_size*2] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold,\
                                self.render_view_id)

                        tmpvox = self.recover_voxel(gt_in.cpu().numpy()[0][0],xmin,xmax,ymin,ymax,zmin,zmax)
                        imgout_0[:,4*self.real_size:self.real_size*5] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold,\
                                self.render_view_id)

                        out_vox, out_vox_or, dis = self.retrieve_to_full_shape_topk(c_feats_less, c_locs, d_feats_less,\
                                d_locs, gt_in, partial_in,input_in,dxb,top_num=top_num, \
                                apply_mask=True,target_pt_num=self.pt_num,)
                        iou = self.compute_iou(out_vox, gt_in.cpu().numpy()[0][0],thresh=self.sampling_threshold)
                        total_iou += iou
                        print(idx ,iou, total_iou/(idx+1))


                        out_vox = self.reshape_to_patch_size(out_vox, partial_shape)
                        out_vox_or = self.reshape_to_patch_size(out_vox_or, partial_shape)

                        tmpvox = self.recover_voxel(out_vox,xmin,xmax,ymin,ymax,zmin,zmax)
                        imgout_0[:,3*self.real_size:4*self.real_size] = self.voxel_renderer.render_img(tmpvox, \
                                self.sampling_threshold, self.render_view_id)

                        tmpvox = self.recover_voxel(out_vox_or,xmin,xmax,ymin,ymax,zmin,zmax)
                        imgout_0[:,2*self.real_size:3*self.real_size] = self.voxel_renderer.render_img(tmpvox, \
                                self.sampling_threshold, self.render_view_id)
                        

                        if self.dump_deform:
                            cv2.imwrite(self.dump_deform_path+"/top%d_%d_%d_%s_%.3f.png"%(top_num,epoch, self.names[dxb], \
                                                                                              self.pre_fix, iou), imgout_0)
                        else:
                            cv2.imwrite(self.sample_dir+"/top%d_%s_%d_%d_%.3f.png"%(top_num, self.pre_fix, epoch, \
                                    self.names[dxb], iou), imgout_0)
                
                if self.mode=='test' and not config.continue_train:
                    continue


                num_c = c_feats_less.shape[0]
                num_d = d_feats_less.shape[0]

                if epoch < 5000:
                    try:
                        re_c_ids, re_d_ids = self.random_retrieve(num_c, num_d, K=self.K, c_max_num=self.max_sample_num) # [300], [300, 20]
                    except:
                        continue
                else:
                    re_c_ids, re_d_ids, = self.distance_retrieve(c_feats_less, d_feats_less, K=self.K)

                re_dis = self.compute_feat_dis(c_feats_less, d_feats_less, re_c_ids, re_d_ids)

                gt_dis = self.get_icp_dis(gt_in, partial_in, input_in,  c_locs, d_locs, re_c_ids,re_d_ids, \
                            target_pt_num=self.pt_num, check=False)

                # print("gt_dis", gt_dis, gt_dis.mean(), gt_dis.min(), gt_dis.max())
                gt_dis = gt_dis*(gt_dis<self.dis_thres) + 10*gt_dis*(gt_dis>=self.dis_thres)
                gt_dis = torch.from_numpy(gt_dis).float().to(self.device)

                same_d_ids = np.arange(d_locs.shape[0]).reshape(-1, 1)
                same_c_ids = np.zeros(same_d_ids.shape[0])

                same_mask = np.ones(d_locs.shape[0])

                for i in range(same_d_ids.shape[0]):
                    loc_dif = np.sum(np.square((c_locs - d_locs[i])), axis=1)
                    if loc_dif.min()>0:
                        same_mask[i]=0
                        continue

                    same_c_ids[i] = np.where(loc_dif==0)[0][0]
                same_mask = same_mask.astype(np.bool)
                same_c_ids = same_c_ids.astype(np.int32)

                same_c_ids = same_c_ids[same_mask]
                same_d_ids = same_d_ids[same_mask]

                same_gt_dis = self.get_icp_dis(gt_in, partial_in,input_in, c_locs, d_locs, same_c_ids, same_d_ids, \
                                target_pt_num=self.pt_num, check=False)

                #print('same_gt_dis', same_gt_dis, same_gt_dis.mean(), same_gt_dis.min(), same_gt_dis.max())
                same_re_dis = self.compute_feat_dis(c_feats_less, d_feats_less, same_c_ids, same_d_ids)
                same_gt_dis = torch.from_numpy(same_gt_dis).float().to(self.device)
                # print('same_gt_dis', same_gt_dis, same_gt_dis.mean(), same_gt_dis.min(), same_gt_dis.max())

                same_mask = (same_gt_dis<self.dis_thres).float().to(self.device) 
                loss_identity = self.regression_loss(same_re_dis, same_gt_dis, mask=same_mask)
                loss_rand = self.regression_loss(re_dis, gt_dis)
                # print(dxb,partial_in.sum(), loss_rand.item(),loss_identity.item())

                loss = loss_rand + self.w_ident*loss_identity 

                loss.backward()
                self.optimizer.step()

            if self.mode=='test':
                return 

            self.log_string("Epoch: [%d/%d] time: %.0f, loss_rand: %.5f, loss_itent:  %.5f" % \
                (epoch, training_epoch, time.time() - start_time, loss_rand.item(),loss_identity.item(), ))


            if epoch % 5==0 and self.mode=='train':

                imgout_0 = np.full([self.real_size, self.real_size*5], 255, np.uint8)
                xmin,xmax,ymin,ymax,zmin,zmax = self.pos_content[dxb]

                if self.pred_coarse:
                    imgout_0 = np.full([self.real_size, self.real_size*7], 255, np.uint8)
                    tmpvox = self.recover_voxel(partial_tmp, xmin,xmax,ymin,ymax,zmin,zmax)
                    imgout_0[:,5*self.real_size:6*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, \
                            self.render_view_id)
                    tmpvox = self.recover_voxel(input_in.cpu().numpy()[0][0], xmin,xmax,ymin,ymax,zmin,zmax)
                    imgout_0[:,6*self.real_size:7*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, \
                            self.render_view_id)


                tmpvox = self.recover_voxel(input_in.cpu().numpy()[0,0],xmin,xmax,ymin,ymax,zmin,zmax)
                imgout_0[:,0:self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

                tmpvox = self.recover_voxel(partial_in.cpu().numpy()[0][0],xmin,xmax,ymin,ymax,zmin,zmax)
                imgout_0[:,self.real_size:self.real_size*2] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, \
                        self.render_view_id)

                tmpvox = self.recover_voxel(gt_in.cpu().numpy()[0,0],xmin,xmax,ymin,ymax,zmin,zmax)
                imgout_0[:,4*self.real_size:self.real_size*5] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, \
                        self.render_view_id)

                out_vox, out_vox_or, dis = \
                    self.retrieve_to_full_shape(c_feats_less, c_locs, d_feats_less, d_locs, gt_in, partial_in,input_in, dxb,\
                    target_pt_num=self.pt_num)


                tmpvox = self.recover_voxel(out_vox,xmin,xmax,ymin,ymax,zmin,zmax)
                imgout_0[:,3*self.real_size:4*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, \
                        self.render_view_id)

                tmpvox = self.recover_voxel(out_vox_or,xmin,xmax,ymin,ymax,zmin,zmax)
                imgout_0[:,2*self.real_size:3*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, \
                        self.render_view_id)

                cv2.imwrite(self.sample_dir+"/%s_%d_%.4f.png"%(self.mode, epoch, dis), imgout_0)

            if epoch%self.save_epoch==0 and self.mode!='test':
                self.save(epoch)

            self.scheduler.step()
        if self.mode=='train':
            self.save(epoch)

    def get_ids_from_locs(self, locs, total_locs, or_ids):
        out_ids = np.zeros(locs.shape[0])
        for i in range(locs.shape[0]):
            loc_dif = np.sum(np.abs(np.expand_dims(locs[i],0) - total_locs), 1)
            if loc_dif.min()>0:
                out_ids[i] = or_ids[i]
            else:
                out_ids[i] = np.where(loc_dif==0)[0][0]
        return out_ids

    def get_icp_dis(self, gt_in, partial_in,coarse_in, c_locs, d_locs, re_c_ids,re_d_ids, target_pt_num=768, check=False, return_vox=False):

        gt_np = gt_in.cpu().numpy()[0,0]
        partial_np = partial_in.cpu().numpy()[0,0]

        # get point clouds
        k = re_d_ids.shape[1]
        c_starts =  c_locs[re_c_ids]*self.stride
        all_d_ids = np.unique(re_d_ids).astype(np.int32)
        d_starts = d_locs[all_d_ids]*self.stride

        if return_vox==True:
            coarse_np = coarse_in.cpu().numpy()[0,0]
            real_d_starts = d_locs[re_d_ids[:,0]]*self.stride


        target_points = np.zeros((c_starts.shape[0], target_pt_num, 3),)
        source_points_all = np.zeros((d_starts.shape[0], target_pt_num, 3),)
        source_points_all_re = np.zeros((d_starts.shape[0], target_pt_num, 3),)

        if return_vox==True:
            or_c_vox = np.zeros((c_starts.shape[0], self.patch_size, self.patch_size, self.patch_size))
            or_d_vox = np.zeros((c_starts.shape[0], self.patch_size, self.patch_size, self.patch_size))
            gt_vox = np.zeros((c_starts.shape[0], self.patch_size, self.patch_size, self.patch_size))

        for i in range(c_starts.shape[0]):
            vox = gt_np[c_starts[i, 0]:c_starts[i, 0]+self.patch_size, \
                c_starts[i, 1]:c_starts[i, 1]+self.patch_size, c_starts[i, 2]:c_starts[i, 2]+self.patch_size]
            target_points[i] = self.volume_to_point_cloud_to_num(vox, self.patch_coord,target_pt_num)
            if return_vox==True:
                or_c_vox[i] = coarse_np[c_starts[i, 0]:c_starts[i, 0]+self.patch_size, \
                    c_starts[i, 1]:c_starts[i, 1] + self.patch_size, c_starts[i, 2]:c_starts[i, 2]+self.patch_size]

                or_d_vox[i] = partial_np[real_d_starts[i, 0]:real_d_starts[i, 0]+self.patch_size, \
                    real_d_starts[i, 1]:real_d_starts[i, 1]+self.patch_size, real_d_starts[i, 2]:real_d_starts[i, 2]+self.patch_size]
                gt_vox[i] = vox
        if return_vox==True:
            return gt_vox, or_c_vox, or_d_vox

        for i in range(d_starts.shape[0]):
            vox = partial_np[d_starts[i, 0]:d_starts[i, 0]+self.patch_size, \
                d_starts[i, 1]:d_starts[i, 1]+self.patch_size, d_starts[i, 2]:d_starts[i, 2]+self.patch_size]
            source_points_all[i] = self.volume_to_point_cloud_to_num(vox, self.patch_coord,target_pt_num)
            source_points_all_re[i] = source_points_all[i]*np.array([1.0,1.0,-1.0])

        out_dis = np.ones((len(re_c_ids), k))

        for i in range(k):
            source_points = np.zeros((c_starts.shape[0], target_pt_num, 3))
            source_points_re = np.zeros((c_starts.shape[0], target_pt_num, 3))
            for j in range(c_starts.shape[0]):
                # print(re_d_ids[j,i], np.where(all_d_ids==re_d_ids[j,i])[0][0])
                source_pt_idx = np.where(all_d_ids==re_d_ids[j,i])[0][0]
                source_points[j] =  source_points_all[source_pt_idx]
                source_points_re[j] = source_points_all_re[source_pt_idx]
            dis_i, trans_i = self.icp_optim(source_points, target_points,)
            dis_i_re, trans_i_re = self.icp_optim(source_points_re, target_points,)
            out_dis[:,i] = np.minimum(dis_i, dis_i_re)


            if check==True:
                patch_renderer = voxel_renderer(self.patch_size)
                imgout_0 = np.full([self.patch_size, self.patch_size*3], 255, np.uint8)
                trans_i = trans_i*float(self.patch_size//2)
                trans_i_re = trans_i_re*float(self.patch_size//2)

                out_path = os.path.join(self.sample_dir, 'check_patch_dis')
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                d_starts = d_locs[re_d_ids[:,0].astype(np.int32)]*self.stride
                for j in range(c_starts.shape[0]):
                    vox_t = gt_np[c_starts[j, 0]:c_starts[j, 0]+self.patch_size, \
                        c_starts[j, 1]:c_starts[j, 1]+self.patch_size, c_starts[j, 2]:c_starts[j, 2]+self.patch_size]

                    vox_s = partial_np[d_starts[j, 0]:d_starts[j, 0]+self.patch_size, \
                        d_starts[j, 1]:d_starts[j, 1]+self.patch_size, d_starts[j, 2]:d_starts[j, 2]+self.patch_size]
                    if dis_i[j]>dis_i_re[j]:
                        vox_re = np.flip(vox_s, 2)
                        vox_out = self.move_vox(vox_re, trans_i_re[j,0].astype(np.int32))
                        im_dis = dis_i_re[j]
                    else:
                        vox_out = self.move_vox(vox_s, trans_i[j,0].astype(np.int32))
                        im_dis = dis_i[j]

                    imgout_0[:, :self.patch_size] = patch_renderer.render_img(vox_s, self.sampling_threshold,0)
                    imgout_0[:, self.patch_size:2*self.patch_size] = patch_renderer.render_img(vox_out, self.sampling_threshold,0)
                    imgout_0[:, 2*self.patch_size:] = patch_renderer.render_img(vox_t, self.sampling_threshold, 0)
                    cv2.imwrite(os.path.join(out_path, "%d_%.4f.png"%(j, im_dis)), imgout_0)
        return out_dis

    def get_icp_dis_from_patches(self, d_patches, gt_patches,target_pt_num=768, ):
        target_points = np.zeros((gt_patches.shape[0], target_pt_num, 3),)
        source_points = np.zeros((d_patches.shape[0], target_pt_num, 3),)
        source_points_re = np.zeros((d_patches.shape[0], target_pt_num, 3),)

        for i in range(d_patches.shape[0]):
            target_points[i] = self.volume_to_point_cloud_to_num(gt_patches[i], self.patch_coord,target_pt_num)
            source_points[i] = self.volume_to_point_cloud_to_num(d_patches[i], self.patch_coord,target_pt_num)
            source_points_re[i] = source_points[i]*np.array([1.0,1.0,-1.0])
        dis_i, trans_i = self.icp_optim(source_points, target_points,)
        dis_i_re, trans_i_re = self.icp_optim(source_points_re, target_points,)
        out_dis = np.minimum(dis_i, dis_i_re)
        return out_dis

    def icp_optim(self, source, target,icp_learning_rate = 0.5,icp_n_iters = 20, device='cuda:0'):
        """
        source, target: [p,p,p,]
        """

        pc_s = torch.from_numpy(source).float().to(self.device)
        pc_t = torch.from_numpy(target).float().to(self.device)

        c_s = pc_s.mean(1)
        c_t = pc_t.mean(1)

        param_init = (c_t-c_s).float().to(device).view(-1,1,3) # translation
        param_init.requires_grad_(True)
        optimizer = torch.optim.SGD([param_init], lr=icp_learning_rate)

        ## ICP Training loop
        loss_pre = 0.0

        for icp_epoch in range(icp_n_iters):
            optimizer.zero_grad()

            output_pc = pc_s + param_init
            # output_pc = torch.clamp(output_pc, min=-1.0, max=1.0)

            dist1, dist2, idx1, idx2 = chamLoss(output_pc, pc_t)
            loss = dist1.mean() + dist2.mean()
            dis = dist1.mean(1)+dist2.mean(1)
            # loss = torch.mean((output_pc-pc_t)**2)

            loss.backward(retain_graph=True)
            # loss.backward()
            optimizer.step()
            if np.abs(loss.item()-loss_pre)/loss.item() < 0.0001:
                break
            # if icp_epoch % 20 == 0:
            #     print('epoch ', icp_epoch+1, ' loss = ', loss.data.cpu().numpy(), loss_pre)
            loss_pre = loss.item()

        return dis.data.cpu().numpy(), param_init.detach().cpu().numpy()

    def retrieve_to_full_shape(self, c_feat, c_locs, d_feat, d_locs, gt_in, partial_in,coarse_in,idx,  coarse=False, dump_patch=False, sigma=1, target_pt_num=768):
        print('retrieve to full shape...')

        c_feat = c_feat.detach().cpu()
        d_feat = d_feat.detach().cpu()
        gt_np = gt_in.cpu().numpy()[0, 0]
        partial_np = partial_in.cpu().numpy()[0, 0]

        if coarse:
            coarse_np = coarse_in.cpu().numpy()[0, 0]

        c_locs_mask =  (c_locs % (self.patch_size//self.stride)).sum(1)
        c_locs_mask2 = (c_locs[:,0]!=c_locs[:,0].max())*(c_locs[:,1]!=c_locs[:,1].max())*(c_locs[:,2]!=c_locs[:,2].max())
        c_locs_mask = c_locs_mask2.astype(np.int32)*c_locs_mask

        c_locs_final = c_locs[c_locs_mask==0]
        c_feat_chosen = c_feat[c_locs_mask==0]

        c_num = c_feat_chosen.shape[0]
        d_num = d_feat.shape[0]

        c_feat_chosen = c_feat_chosen.view(c_num, 1, -1)
        d_feat_chosen = d_feat.view(1, d_num, -1)

        dis_matrix = torch.sqrt(torch.sum(torch.square(c_feat_chosen - d_feat_chosen), -1)) #[c_num, d_num]

        re_ids = np.argmin(dis_matrix, axis=1)
        d_locs_final = d_locs[re_ids]

        source_points = np.zeros((c_num, target_pt_num, 3))
        source_points_re = np.zeros((c_num, target_pt_num, 3))
        target_points = np.zeros((c_num, target_pt_num, 3))
        out_vox = np.zeros(gt_np.shape)
        out_vox_or = np.zeros(gt_np.shape)

        c_starts = c_locs_final*self.stride
        d_starts = d_locs_final*self.stride
        # print(d_starts, c_starts)

        for i in range(c_num):
            vox_gt = gt_np[c_starts[i, 0]:c_starts[i, 0]+self.patch_size, \
                c_starts[i, 1]:c_starts[i, 1]+self.patch_size, c_starts[i, 2]:c_starts[i, 2]+self.patch_size]
            # target_points[i] = self.volume_to_point_cloud_to_num(vox_gt, self.patch_coord,target_pt_num)

            vox_partial = partial_np[d_starts[i, 0]:d_starts[i, 0]+self.patch_size, \
                d_starts[i, 1]:d_starts[i, 1]+self.patch_size, d_starts[i, 2]:d_starts[i, 2]+self.patch_size]
            # source_points[i] = self.volume_to_point_cloud_to_num(vox_partial, self.patch_coord,target_pt_num)

            out_vox_or[c_starts[i, 0]:c_starts[i, 0]+self.patch_size, c_starts[i, 1]:c_starts[i, 1]+self.patch_size, \
                c_starts[i, 2]:c_starts[i, 2]+self.patch_size] = vox_partial

            if coarse:
                vox_gt = coarse_np[c_starts[i, 0]:c_starts[i, 0]+self.patch_size, \
                    c_starts[i, 1]:c_starts[i, 1]+self.patch_size, c_starts[i, 2]:c_starts[i, 2]+self.patch_size]

                down_rate = self.output_size//self.input_size
                vox_tensor = torch.from_numpy(vox_partial).float().unsqueeze(0).unsqueeze(0).cuda()
                if self.upsample_rate==2:
                    smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size = down_rate, stride = down_rate, padding = 0)
                    input_vox = F.interpolate(smallmaskx_tensor, scale_factor = down_rate, mode='nearest')
                else:
                    input_vox = F.max_pool3d(vox_tensor, kernel_size = down_rate, stride = 1, padding = down_rate//2)
                    input_vox = input_vox[:,:,:-1,:-1,:-1]

                vox_partial = input_vox.cpu().numpy()[0,0]

            target_points[i] = self.volume_to_point_cloud_to_num(vox_gt, self.patch_coord,target_pt_num)
            source_points[i] = self.volume_to_point_cloud_to_num(vox_partial, self.patch_coord,target_pt_num)

            vox_partial_re = np.flip(vox_partial, 2)
            source_points_re[i] = self.volume_to_point_cloud_to_num(vox_partial_re, self.patch_coord,target_pt_num)

        dis, trans = self.icp_optim(source_points, target_points,)
        dis_re, trans_re = self.icp_optim(source_points_re, target_points,)

        trans = trans*float(self.patch_size//2)
        trans_re = trans_re*float(self.patch_size//2)

        for i in range(c_num):
            vox_partial = partial_np[d_starts[i, 0]:d_starts[i, 0]+self.patch_size, \
                d_starts[i, 1]:d_starts[i, 1]+self.patch_size, d_starts[i, 2]:d_starts[i, 2]+self.patch_size]
            if dis[i]>dis_re[i]:
                vox_partial_re = np.flip(vox_partial, 2)
                out_vox_i = self.move_vox(vox_partial_re, np.around(trans_re[i,0]).astype(np.int32))
            else:
                out_vox_i = self.move_vox(vox_partial, np.around(trans[i,0]).astype(np.int32))

            out_vox[c_starts[i, 0]:c_starts[i, 0]+self.patch_size, c_starts[i, 1]:c_starts[i, 1]+self.patch_size, \
                c_starts[i, 2]:c_starts[i, 2]+self.patch_size] = out_vox_i


        return out_vox, out_vox_or, dis.mean()

    def retrieve_to_full_shape_topk(self, c_feat, c_locs, d_feat, d_locs, gt_in, partial_in, coarse_in, idx,  return_k=False, \
            coarse=False, apply_mask=False,  top_num=5, sigma=1, \
            target_pt_num=768, ):
        print('retrieve to full shape top %d...'%(top_num))

        c_feat = c_feat.detach().cpu()
        d_feat = d_feat.detach().cpu()
        gt_np = gt_in.cpu().numpy()[0, 0]
        partial_np = partial_in.cpu().numpy()[0, 0]
        coarse_np = coarse_in.cpu().numpy()[0, 0]
        c_locs_mask =  (c_locs % (self.patch_size//self.stride)).sum(1)

        if return_k:
            c_locs_mask = (c_locs % 2).sum(1)

        c_locs_mask2 = (c_locs[:,0]!=c_locs[:,0].max())*(c_locs[:,1]!=c_locs[:,1].max())*(c_locs[:,2]!=c_locs[:,2].max())
        c_locs_mask = c_locs_mask2.astype(np.int32)*c_locs_mask

        c_locs_final = c_locs[c_locs_mask==0]
        c_feat_chosen = c_feat[c_locs_mask==0]

        if return_k:
            c_locs_mask = np.zeros(c_locs.shape[0])
            c_locs_final = c_locs
            c_feat_chosen = c_feat

        c_num = c_feat_chosen.shape[0]
        d_num = d_feat.shape[0]

        c_feat_chosen = c_feat_chosen.view(c_num, 1, -1)
        d_feat_chosen = d_feat.view(1, d_num, -1)

        dis_matrix = torch.sqrt(torch.sum(torch.square(c_feat_chosen - d_feat_chosen), -1)) #[c_num, d_num]
        sorted_dis, indices = torch.sort(dis_matrix)

        total_dis = np.zeros((c_num,top_num))
        total_trans = np.zeros((c_num, top_num, 4))
        total_flip = np.zeros((c_num,top_num))

        if return_k:
            # total_locs = np.zeros((c_num, 1+top_num, 3))
            # total_locs[:, 0] = c_locs_final
            total_ids = np.zeros((c_num, 1+top_num,))
            total_ids[:, 0] = np.where(c_locs_mask==0)[0]

        for k_i in range(top_num):
            re_ids = indices[:, k_i]
            d_locs_final = d_locs[re_ids]

            if return_k:
                # total_locs[:, 1+k_i] = d_locs_final
                total_ids[:, 1+k_i] = re_ids

            source_points = np.zeros((c_num, target_pt_num, 3))
            source_points_re = np.zeros((c_num, target_pt_num, 3))
            target_points = np.zeros((c_num, target_pt_num, 3))
            out_vox = np.zeros(gt_np.shape)
            out_vox_or = np.zeros(gt_np.shape)

            c_starts = c_locs_final*self.stride
            d_starts = d_locs_final*self.stride
            # print(d_starts, c_starts)

            for i in range(c_num):
                vox_gt = gt_np[c_starts[i, 0]:c_starts[i, 0]+self.patch_size, \
                    c_starts[i, 1]:c_starts[i, 1]+self.patch_size, c_starts[i, 2]:c_starts[i, 2]+self.patch_size]

                vox_partial = partial_np[d_starts[i, 0]:d_starts[i, 0]+self.patch_size, \
                    d_starts[i, 1]:d_starts[i, 1]+self.patch_size, d_starts[i, 2]:d_starts[i, 2]+self.patch_size]

                if coarse:
                    vox_gt = coarse_np[c_starts[i, 0]:c_starts[i, 0]+self.patch_size, \
                        c_starts[i, 1]:c_starts[i, 1]+self.patch_size, c_starts[i, 2]:c_starts[i, 2]+self.patch_size]

                    down_rate = self.output_size//self.input_size
                    vox_tensor = torch.from_numpy(vox_partial).float().unsqueeze(0).unsqueeze(0).cuda()
                    if self.input_size==64:
                        smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size = down_rate, stride = down_rate, padding = 0)
                        input_vox = F.interpolate(smallmaskx_tensor, scale_factor = down_rate, mode='nearest')
                    else:
                        input_vox = F.max_pool3d(vox_tensor, kernel_size = down_rate, stride = 1, padding = down_rate//2)
                        input_vox = input_vox[:,:,:-1,:-1,:-1]
                    vox_partial = input_vox.cpu().numpy()[0,0]

                target_points[i] = self.volume_to_point_cloud_to_num(vox_gt, self.patch_coord,target_pt_num)
                source_points[i] = self.volume_to_point_cloud_to_num(vox_partial, self.patch_coord,target_pt_num)

                vox_partial_re = np.flip(vox_partial, 2)
                source_points_re[i] = self.volume_to_point_cloud_to_num(vox_partial_re, self.patch_coord,target_pt_num)

            dis, trans = self.icp_optim(source_points, target_points,)
            dis_re, trans_re = self.icp_optim(source_points_re, target_points,)

            trans = trans*float(self.patch_size//2)
            trans_re = trans_re*float(self.patch_size//2)

            for i in range(c_num):
                if dis[i]> dis_re[i]:
                    total_dis[i,k_i] = dis_re[i]
                    total_flip[i, k_i] = 1.0
                    total_trans[i, k_i, 0:3] = trans_re[i]
                    total_trans[i, k_i, 3] = 1.0
                else:
                    total_dis[i,k_i] = dis[i]
                    total_trans[i, k_i,0:3] = trans[i]
                    total_trans[i, k_i, 3] = 0.0

        if return_k:
            return total_ids, total_dis, total_trans

        d_ids_tmp = np.argmin(total_dis, axis=1)
        out_dis = np.zeros(c_num)

        for i in range(c_num):

            d_id = indices[i, d_ids_tmp[i]]
            d_start = d_locs[d_id]*self.stride
            out_dis[i] = total_dis[i, d_ids_tmp[i]]

            loc_mask =  (c_locs_final[i] % (self.patch_size//self.stride)).sum()
            loc_mask2 = (c_locs_final[i,0]!=c_locs[:,0].max())*(c_locs_final[i,1]!=c_locs[:,1].max())*(c_locs_final[i,2]!=c_locs[:,2].max())
            loc_mask =loc_mask2.astype(np.int32)*loc_mask

            vox_partial = partial_np[d_start[0]:d_start[ 0]+self.patch_size, \
                d_start[1]:d_start[1]+self.patch_size, d_start[2]:d_start[2]+self.patch_size]

            if loc_mask==0:
                out_vox_or[c_starts[i, 0]:c_starts[i, 0]+self.patch_size, c_starts[i, 1]:c_starts[i, 1]+self.patch_size, \
                    c_starts[i, 2]:c_starts[i, 2]+self.patch_size] = vox_partial

            if total_flip[i, d_ids_tmp[i]]==1:
                vox_partial_re = np.flip(vox_partial, 2)
                out_vox_i = self.move_vox(vox_partial_re, np.around(total_trans[i,d_ids_tmp[i], 0:3]).astype(np.int32))
            else:
                out_vox_i = self.move_vox(vox_partial, np.around(total_trans[i,d_ids_tmp[i], 0:3]).astype(np.int32))

            if apply_mask:
                vox_coarse = coarse_np[c_starts[i, 0]:c_starts[i, 0]+self.patch_size, \
                     c_starts[i, 1]:c_starts[i, 1]+self.patch_size, c_starts[i, 2]:c_starts[i, 2]+self.patch_size]
                out_vox_i = out_vox_i*vox_coarse

            if loc_mask==0:
                out_vox[c_starts[i, 0]:c_starts[i, 0]+self.patch_size, c_starts[i, 1]:c_starts[i, 1]+self.patch_size, \
                    c_starts[i, 2]:c_starts[i, 2]+self.patch_size] = out_vox_i

        return out_vox, out_vox_or, out_dis.mean()


    def pad_mask(self, v1, v2,mode='replicate'):
        x1, y1, z1 = v1.shape
        x2, y2, z2 = v2.shape
        x = max(x1, x2)
        y = max(y1, y2)
        z = max(z1, z2)

        new_v1 = torch.from_numpy(v1).float().to(self.device).unsqueeze(0).unsqueeze(0)
        new_v1 = F.pad(new_v1, (int((z-z1)/2), z-z1-int((z-z1)/2),int((y-y1)/2), \
            y-y1-int((y-y1)/2), int((x-x1)/2), x-x1-int((x-x1)/2)),mode=mode)

        return new_v1.cpu().numpy()[0][0]


    def get_voxel_bbox_simple(self,vox):
        #minimap
        smallmaskx = vox
        smallx,smally,smallz = smallmaskx.shape
        #x
        ray = np.max(smallmaskx,(1,2))
        xmin = -1
        xmax = 0
        for i in range(smallx):
            if ray[i]>0:
                if xmin==-1:
                    xmin = i
                xmax = i
        #y
        ray = np.max(smallmaskx,(0,2))
        ymin = -1
        ymax = 0
        for i in range(smally):
            if ray[i]>0:
                if ymin==-1:
                    ymin = i
                ymax = i
        #z
        ray = np.max(smallmaskx,(0,1))
        zmin = -1
        zmax = 0
        for i in range(smallz):
            if ray[i]>0:
                if zmin==-1:
                    zmin = i
                zmax = i
        return xmin,xmax+1,ymin,ymax+1,zmin,zmax+1


    def move_vox(self, vox, shift):
        out_vox = np.zeros(vox.shape)
        start_1 = np.maximum(shift, np.zeros(shift.shape)).astype(np.int32)
        start_2 = np.maximum(np.zeros(shift.shape), -shift).astype(np.int32)

        span = self.patch_size - np.abs(shift)
        end_1= np.minimum([start_1[0]+span[0], start_1[1]+span[1], start_1[2]+span[2]], \
            [self.patch_size, self.patch_size, self.patch_size]).astype(np.int32)
        r_span = np.array([end_1[0]-start_1[0], end_1[1]-start_1[1], end_1[2]-start_1[2],])

        out_vox[start_1[0]:end_1[0], start_1[1]:end_1[1],start_1[2]:end_1[2]]= \
            vox[start_2[0]:start_2[0]+r_span[0], start_2[1]:start_2[1]+r_span[1], start_2[2]:start_2[2]+r_span[2]]
        return out_vox


    def move_vox_torch(self, vox, shift):
        out_vox = torch.zeros(vox.shape).to(self.device)
        start_1 = np.maximum(shift, np.zeros(shift.shape)).astype(np.int32)
        start_2 = np.maximum(np.zeros(shift.shape), -shift).astype(np.int32)

        span = self.patch_size - np.abs(shift)
        end_1= np.minimum([start_1[0]+span[0], start_1[1]+span[1], start_1[2]+span[2]], \
             [self.patch_size, self.patch_size, self.patch_size]).astype(np.int32)
        r_span = np.array([end_1[0]-start_1[0], end_1[1]-start_1[1], end_1[2]-start_1[2],])

        out_vox[start_1[0]:end_1[0], start_1[1]:end_1[1],start_1[2]:end_1[2]]= \
            vox[start_2[0]:start_2[0]+r_span[0], start_2[1]:start_2[1]+r_span[1], start_2[2]:start_2[2]+r_span[2]]
        return out_vox

    def volume_to_point_cloud_to_num(self,vol, coord, target_num=768):
        """
        vol: [vsize, vsize, vsize], torch
        """
        vsize = vol.shape[0]
        # num_points = (vol>0.4).astype(np.int32).sum()
        # or_points = coord[vol>0.4].astype(np.float32)

        pt_mask = (vol>self.pt_sampling_threshold).astype(np.int32)


        or_points= coord[pt_mask.astype(np.bool)].astype(np.float32)
        num_points = pt_mask.sum()
        # print('num_points', num_points)

        if num_points==0:
            return np.zeros((target_num,3))
        if num_points>=target_num:
            point_ids = np.random.randint(0,target_num, size=target_num)
            points = or_points[point_ids]
        else:
            points=[]
            points.append(or_points)
            up_rate = target_num//num_points
            for i in range(up_rate):
                pert_points = or_points + 0.*np.random.rand(num_points, 3)
                points.append(pert_points)
            points = np.array(points).reshape((up_rate+1)*num_points, 3)
            point_ids = np.random.randint(0,points.shape[0], size=target_num)
            points = points[point_ids]
        # [-1, 1]
        half_p = float(vsize/2)
        points = (points - half_p)/half_p
        return points

    def patch_to_xyzrgb(self, patch, c_loc, d_loc):
        points = self.volume_to_point_cloud_new(patch, self.patch_coord)
        num_points = points.shape[0]
        if num_points==0:
            return np.zeros((0,6))
        points = points + np.expand_dims(c_loc, axis=0)
        colors = np.repeat(np.expand_dims(d_loc, 0), num_points, 0)
        xyzrgb = np.concatenate([points, colors], axis=1)
        return xyzrgb


    def volume_to_point_cloud_new(self, vol,coord, thres=0.3,target_num=100):
        pt_mask = (vol>thres).astype(np.int32)

        padded_mask = np.pad(pt_mask, ((1,1), (1,1), (1,1)))
        inter_mask = padded_mask[0:-2, 1:-1, 1:-1] *padded_mask[2:, 1:-1, 1:-1] \
            *padded_mask[1:-1, 0:-2, 1:-1] *padded_mask[1:-1, 2:, 1:-1] \
            *padded_mask[1:-1, 1:-1, 0:-2] *padded_mask[1:-1, 1:-1, 2:]
        pt_mask = (1-inter_mask)*pt_mask

        or_points= coord[pt_mask.astype(np.bool)].astype(np.float32)
        num_points = pt_mask.sum()

        if num_points==0:
            return np.zeros((0,3))
        if num_points>=target_num:
            point_ids = np.random.randint(0,target_num, size=target_num)
            points = or_points[point_ids]
        else:
            points = or_points
        return points



    def volume_to_point_cloud(self, vol,coord, thres=0.3,):
        pt_mask = (vol>thres).astype(np.int32)
        or_points= coord[pt_mask.astype(np.bool)].astype(np.float32)
        num_points = pt_mask.sum()

        if num_points==0:
            return np.zeros((0,3))
        else:
            return or_points

    def vol_to_point_cloud2(self, vol, vsize, radius=0.5):
        pt_mask = (vol>self.sampling_threshold).astype(np.int32)
        or_points= self.shape_coord[pt_mask.astype(np.bool)].astype(np.float32)
        num_points = pt_mask.sum()
        if num_points==0:
            return np.zeros((0,3))

        pad_size = (self.output_size - vsize)/2
        or_points = or_points - pad_size

        voxel = 2*radius/float(vsize)
        or_points = or_points*voxel
        or_points = or_points - radius
        return or_points

