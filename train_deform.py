import os
import time
import math
import random
import numpy as np
import cv2
import h5py
from glob import glob
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import  sobel

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable


from utils import *
from models import *

from kornia.geometry import get_affine_matrix3d, warp_affine3d
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D
chamLoss = ChamferDistancePytorch.chamfer3D.dist_chamfer_3D.chamfer_3DDist()

class MODEL_DEFORM(object):
    def __init__(self, config):
        self.real_size = 256
        #self.real_size = 128
        self.mask_margin = 8

        self.g_dim = config.g_dim
        self.z_dim = 3

        self.input_size = config.input_size # 32
        self.output_size = config.output_size # 128
        self.patch_size = 18
        self.stride = 4
        self.pt_num = 512
        self.csize = config.csize
        self.c_range = config.c_range

        self.max_sample_num = config.max_sample_num
        # self.upsample_rate = self.output_size//self.input_size
        self.upsample_rate = 1

        self.asymmetry = True 

        self.save_epoch = 5
        self.eval_epoch = 2
        self.start_epoch = 0

        self.mode = config.mode
        self.gua_filter = True
        self.small_dataset = config.small_dataset 
        self.predict_dis = True
        self.pred_coarse = True
        self.dump_deform = config.dump_deform

        self.w_dis = config.w_dis
        self.top_num = 8
        self.pre_fix = ''
        self.sampling_threshold = 0.3

        self.render_view_id = 0
        self.voxel_renderer = voxel_renderer(self.real_size)

        self.checkpoint_dir = config.checkpoint_dir
        self.data_dir = config.data_dir
        
        self.dump_deform_path = config.dump_deform_path

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
        vox_name = "/model.binvox"

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
                test_names = [name.strip() for name in fin.readlines()]
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
            self.gt_pc = []

            self.re_ids = []
            self.re_dis = []
            self.re_dis_c = []
            self.re_trans = []
            self.re_trans_c = []
            self.coarse_dis = []
            self.partial_dis = []
            self.shape_names = []

            self.names = range(self.dataset_len)
            if config.train_deform:
                for i in range(self.dataset_len):
                    print("preprocessing content - "+str(i+1)+"/"+str(self.dataset_len))

                    dump_files = glob(os.path.join(self.dump_deform_path, self.dataset_names[self.names[i]], 'dump_deform_wc_top*.npz'))
                    if len(dump_files)==0:
                        print('not exist', os.path.join(self.dump_deform_path, self.dataset_names[self.names[i]]))
                        self.dataset_len = self.dataset_len -1
                        continue
                    else:
                        dump_f = dump_files[-1]

                    tmp_raw = get_vox_from_binvox_1over2(os.path.join(self.data_dir,self.dataset_names[self.names[i]]+vox_name)).astype(np.uint8)
    
                    xmin,xmax,ymin,ymax,zmin,zmax = self.get_voxel_bbox(tmp_raw)
                    xmin2,xmax2,ymin2,ymax2,zmin2,zmax2 = xmin,xmax,ymin,ymax,zmin,zmax

                    a = np.load(dump_f, allow_pickle=True)
                    re_ids = a['re_ids']
                    re_dis = a['re_dis']*10.0
                    re_trans = a['re_trans']

                    re_dis_c = a['re_dis_c']*10.0
                    re_trans_c = a['re_trans_c']
                    crop_locs = a['crop_locs']

                    tmp = self.crop_voxel(tmp_raw,xmin,xmax,ymin,ymax,zmin,zmax)
                    gt_voxel = tmp
                    tmp_input, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)
                    partial_shape, partial_mask = self.random_crop(tmp, c_locs=crop_locs)
                    partial_mask = tmp_mask

                    if self.gua_filter:
                        tmp_input = gaussian_filter(tmp_input.astype(np.float32), sigma=1)
                        partial_shape =  gaussian_filter(partial_shape.astype(np.float32), sigma=1)

                    self.input_content.append(tmp_input)
                    self.gt_content.append(gt_voxel)
                    self.mask_content.append(tmp_mask)
                    self.partial_content.append(partial_shape)
                    self.partial_mask.append(partial_mask)
                    self.pos_content.append([xmin,xmax,ymin,ymax,zmin,zmax])

                    self.re_dis.append(re_dis)
                    self.re_ids.append(re_ids)
                    self.re_dis_c.append(re_dis_c)
                    self.re_trans.append(re_trans)
                    self.re_trans_c.append(re_trans_c)

                    self.shape_names.append(self.dataset_names[self.names[i]])

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

        self.model = PatchDeformer(self.g_dim, self.z_dim, self.predict_dis,) # 3: translate only
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 100, gamma=0.99)

        if self.pred_coarse:
            print('coarse predictor', self.data_content)
            if self.data_content =='content_chair':
                checkpoint = torch.load('checkpoint_complete/content_chair_complete/coarse_comp/checkpoint-90.pth')
            elif self.data_content=='content_boat':
                checkpoint = torch.load('checkpoint_complete/content_boat_complete/coarse_comp/checkpoint-90.pth')
            elif self.data_content=='content_lamp':
                checkpoint = torch.load('checkpoint_complete/content_lamp_complete/coarse_comp/checkpoint-90.pth')
            elif self.data_content=='content_couch':
                checkpoint = torch.load('checkpoint_complete/content_couch_complete/coarse_comp/checkpoint-150.pth')
            elif self.data_content=='content_cabinet':
                checkpoint = torch.load('checkpoint_complete/content_cabinet_complete/coarse_comp/checkpoint-90.pth')
            elif self.data_content=='content_plane':
                checkpoint = torch.load('checkpoint_complete/content_plane_complete/coarse_comp/checkpoint-90.pth')
            elif self.data_content=='content_car':
                checkpoint = torch.load('checkpoint_complete/content_car_complete/coarse_comp/checkpoint-90.pth') #
            elif self.data_content=='content_table':
                checkpoint = torch.load('checkpoint_complete/content_table_complete/coarse_comp/checkpoint-90.pth')

            self.coarse_predictor = CoarseCompletor_skip(32)
            self.coarse_predictor.to(self.device)
            self.coarse_predictor.load_state_dict(checkpoint['generator'])
            self.coarse_predictor.eval()


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

        if (p<=0.4 or not self.data_content=='content_chair') and c_locs is None:
            while(vox[loc_starts[0]:loc_starts[0]+csize[0], loc_starts[1]:loc_starts[1]+csize[1],\
                loc_starts[2]:loc_starts[2]+csize[2],].sum()<30):
                csize = crop_size + (np.random.rand(3)-0.5)*c_range
                csize = csize.astype(np.int32)

                loc_starts[0] = np.random.randint(0, vx-csize[0])
                loc_starts[1] = np.random.randint(0, vy-csize[1])
                loc_starts[2] = np.random.randint(0, vz-csize[2])
        elif c_locs is None:
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

    def get_voxel_bbox(self,vox):
        #minimap
        vox_tensor = torch.from_numpy(vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
        #smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size = 2, stride = 2, padding = 0)
        #smallmaskx_tensor = F.interpolate(smallmaskx_tensor, scale_factor = 2, mode='nearest')
        smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size=self.upsample_rate, stride=self.upsample_rate, padding=0)
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

    def get_voxel_mask_exact(self,vox):
        #256 -maxpoolk4s4- 64 -upsample- 256
        vox_tensor = torch.from_numpy(vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
        #input
        smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size=self.upsample_rate, stride=self.upsample_rate, padding=0)
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

    def load(self):
        #load previous checkpoint
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            print('loading from', model_dir)
            fin.close()
            checkpoint = torch.load(model_dir)
            self.model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint.keys():
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint.keys():
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
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': epoch
                }, save_dir)

        # update checkpoint manager
        self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
        # write file
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        fout = open(checkpoint_txt, 'w')
        for i in range(self.max_to_keep):
            pointer = (self.checkpoint_manager_pointer+self.max_to_keep-i)%self.max_to_keep
            if self.checkpoint_manager_list[pointer] is not None:
                fout.write(self.checkpoint_manager_list[pointer]+"\n")
        fout.close()

    def filter_locs(self, c_mask, d_mask):
        x_size, y_size, z_size = c_mask.shape
        x_ = range(x_size)
        y_ = range(y_size)
        z_ = range(z_size)
        yy, xx, zz = np.meshgrid(y_,x_,z_)
        feat_coord = np.concatenate((np.expand_dims(xx, -1), np.expand_dims(yy, -1),np.expand_dims(zz, -1)), -1)
        c_coord = feat_coord[c_mask>0]

        x_size, y_size, z_size = d_mask.shape
        x_ = range(x_size)
        y_ = range(y_size)
        z_ = range(z_size)
        yy, xx, zz = np.meshgrid(y_,x_,z_)
        feat_coord = np.concatenate((np.expand_dims(xx, -1), np.expand_dims(yy, -1),np.expand_dims(zz, -1)), -1)
        d_coord = feat_coord[d_mask>0]
        return c_coord, d_coord

    def get_locs(self, re_ids, c_locs, d_locs):
        out_locs = np.zeros((re_ids.shape[0], re_ids.shape[1], 3))
        out_locs[:, 0, :] = c_locs[re_ids[:,0]]

        for i in range(out_locs.shape[1]-1):
            out_locs[:, i+1, :] = d_locs[re_ids[:,i+1]]
        return out_locs.astype(np.int32)

    def get_ids_with_gt(self, re_pairs, total_d_locs):
        c_num = re_pairs.shape[0]
        valid_ids = []
        for i in range(c_num):
            dif = np.abs(total_d_locs - re_pairs[i, 0,]).sum(1).min()
            if dif==0:
                valid_ids.append(i)
        return valid_ids

    def get_dis_by_locs(self, dis_dict, locs, return_mask=False):
        out_dis = np.zeros(locs.shape[0])
        dis_dict = dis_dict.tolist()
        mask = torch.ones(locs.shape[0]).to(self.device)

        for i in range(locs.shape[0]):
            if not tuple(locs[i]) in dis_dict.keys():
                mask[i] = 0.0
                out_dis[i] = 1.0
                continue

            out_dis[i] = 10.0*dis_dict[tuple(locs[i])]
        if return_mask:
            return out_dis, mask
        return out_dis

    def get_exact_partial_mask(self, partial_in, gt_in, p_mask_in):
        partial_in = (partial_in>0.3).float()
        vox_dif = torch.abs(gt_in - partial_in)
        patch_sum = F.avg_pool3d(partial_in, kernel_size = self.patch_size, stride = self.stride, padding = 0)
        patch_sum = patch_sum*float(self.patch_size**3)

        patch_dif = F.avg_pool3d(vox_dif, kernel_size = self.patch_size, stride = self.stride, padding = 0)
        patch_dif = patch_dif*float(self.patch_size**3)

        # patch_dif = (patch_dif<220).float()

        patch_dif = ((patch_dif/(patch_sum+1e-3))<0.8).float()
        mask_final = p_mask_in*patch_dif
        return mask_final

    def distance_retrieve(self, re_ids, re_locs, re_dis, re_trans, max_sample_num=300,sigma=1.0):
        """
        re_locs: [nc, (1+k), 3]
        re_dis: [nc, k]
        flip_list: [nc, k]
        return:
            c_ids, d_ids, distance matrix
        """

        c_num = min(re_locs.shape[0], max_sample_num)
        c_ids = np.array(sorted(np.random.randint(0, re_locs.shape[0], c_num)))
        K = re_locs.shape[1] -1

        dis_prob = re_dis / sigma
        dis_prob = np.exp(-(dis_prob - np.expand_dims(np.max(dis_prob, axis=1), -1)))
        dis_prob = dis_prob/(np.expand_dims(dis_prob.sum(1), axis=1)+1e-7)

        out_locs = np.zeros((c_num, 2, 3), np.int32)
        out_locs[:, 0,] = re_locs[c_ids, 0]

        # out_ids = np.zeros(c_num, np.int32)
        out_d_ids = np.zeros(c_num, np.int32)
        out_c_ids = re_ids[c_ids, 0,]
        out_flip = np.zeros(c_num, np.int32)
        out_trans = np.zeros((c_num, 3))
        out_dis = np.zeros(c_num)

        for i in range(c_num):
            idx = c_ids[i]
            #if dis_prob[idx].sum()<1e-5 or not dis_prob[idx].sum()<1000:
            pid = np.random.randint(0, K,)
            out_locs[i, 1] = re_locs[idx, 1+pid,]
            out_flip[i] = re_trans[idx, pid, 3]
            out_trans[i] = re_trans[idx, pid, 0:3]
            out_d_ids[i] = re_ids[idx, 1+pid]
            out_dis[i] = re_dis[idx, pid]
        return c_ids, out_c_ids, out_d_ids, out_locs, out_dis, out_trans, out_flip

    def regression_loss(self, embedding_distance, actual_distance, mask=None,obj_sigmas=1.0):
        if mask is not None:
            loss = torch.sum(torch.square(mask*(embedding_distance-actual_distance)),)/mask.sum()
        else:
            loss = torch.mean(torch.square(embedding_distance-actual_distance),)
        return loss

    def compute_iou(self, recons, gt, thresh=0.3):
        x = (recons>thresh).astype(np.float32)

        y = (gt>thresh).astype(np.float32)

        intersection = np.sum(x*y)
        union = np.sum(x) + np.sum(y) - np.sum(x*y)
        iou = intersection/(union+1e-5)
        return iou

    @property
    def model_dir(self):
        return "{}_retriv".format(self.data_content)

    def train(self, config):
        if self.dump_deform:
            self.mode='test'

        if config.continue_train or self.mode=='test':
            self.load()
        

        start_time = time.time()
        training_epoch = config.epoch
        if self.mode=='test':
            training_epoch += 1 

        self.dataset_len = len(self.input_content)
        batch_index_list = np.arange(self.dataset_len)

        for epoch in range(self.start_epoch,  training_epoch):
            if self.mode!='test':
                np.random.shuffle(batch_index_list)

            total_loss = 0.0
            total_iou = 0.0
            total_c_iou = 0.0
            total_coa_iou = 0.0

            self.model.train()

            if self.mode=='test':
                self.model.eval()

            for idx in range(self.dataset_len):
                dxb = batch_index_list[idx]

                if not self.pred_coarse:
                    input_in = torch.from_numpy(self.input_content[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                    mask_in =  torch.from_numpy(self.mask_content[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()

                partial_in = torch.from_numpy(self.partial_content[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                mask_partial_in = torch.from_numpy(self.partial_mask[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                gt_in = torch.from_numpy(self.gt_content[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()

                if self.pred_coarse:
                    mask_partial_in = F.max_pool3d((partial_in>self.sampling_threshold).float(), kernel_size=self.patch_size, \
                            stride = self.stride, padding = 0)
                    partial_full = self.reshape_to_size_torch(partial_in, (self.output_size, self.output_size, self.output_size))

                    coarse_out = self.coarse_predictor(partial_full)
                    coarse_out = coarse_out.detach()
                    input_in = self.reshape_to_size_torch(coarse_out, self.input_content[dxb].shape)
                    mask_in = F.max_pool3d((input_in>self.sampling_threshold).float(), kernel_size = self.patch_size, stride = self.stride, padding = 0)

                # re_pairs = self.re_pairs[dxb]
                re_ids = self.re_ids[dxb].astype(np.int32)
                re_dis = self.re_dis[dxb]
                re_dis_c = self.re_dis_c[dxb]
                re_trans = self.re_trans[dxb]
                re_trans_c = self.re_trans_c[dxb]

                # total_c_locs, total_d_locs = self.filter_locs(self.mask_content[dxb], self.partial_mask[dxb])
                total_c_locs, total_d_locs = self.filter_locs(mask_in.cpu().numpy()[0][0], mask_partial_in.cpu().numpy()[0][0])
                try:
                    re_pairs = self.get_locs(re_ids, total_c_locs, total_d_locs)
                    re_pairs_or = re_pairs
                except:
                    print(idx,'get locs error')
                    continue

                if self.dump_deform and (epoch%20==0 or epoch == self.start_epoch):
                    total_pred_trans, total_pred_dis, total_init_trans = self.retrieve_to_full_shape_fromdis(\
                                        self.gt_content[dxb], partial_in.cpu().numpy()[0][0], \
                                        input_in.cpu().numpy()[0][0], re_pairs_or, \
                                        self.re_dis[dxb], self.re_trans_c[dxb], align=True, dump_deform=True)
                    np.savez(os.path.join(self.dump_deform_path, self.shape_names[dxb], "pred_deform_%d.npz"%(epoch)), \
                        pred_trans=total_pred_trans, pred_dis=total_pred_dis, init_trans=total_init_trans)
                

                if self.mode=='test' and (epoch%20==0 or epoch==self.start_epoch) and dxb<20:

                    xmin,xmax,ymin,ymax,zmin,zmax = self.pos_content[dxb]
                    imgout_0 = np.full([self.real_size, self.real_size*6], 255, np.uint8)
                    pred_shape, retrieved_patches, aligned_patches, = self.retrieve_to_full_shape_fromdis(\
                            self.gt_content[dxb], partial_in.cpu().numpy()[0][0], input_in.cpu().numpy()[0][0], \
                            re_pairs_or, self.re_dis[dxb], self.re_trans_c[dxb], align=True,)

                    iou = self.compute_iou(pred_shape, self.gt_content[dxb],thresh=self.sampling_threshold)
                    total_iou += iou
                    print(idx ,iou, total_iou/(idx+1))

                    iou = self.compute_iou(input_in.cpu().numpy()[0,0], self.gt_content[dxb],thresh=self.sampling_threshold)
                    total_coa_iou += iou
                    print('coa iou', idx ,iou, total_coa_iou/(idx+1))

                    # imgout_0 = np.full([self.real_size, self.real_size*6], 255, np.uint8)
                    xmin,xmax,ymin,ymax,zmin,zmax = self.pos_content[dxb]
                    tmpvox = self.recover_voxel(input_in.cpu().numpy()[0][0],xmin,xmax,ymin,ymax,zmin,zmax)
                    imgout_0[:,0:self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, \
                            self.render_view_id)
                    tmpvox = self.reshape_to_patch_size(partial_in.cpu().numpy()[0][0], np.zeros((256,256,256)))
                    # tmpvox = self.recover_voxel(self.partial_content[dxb],xmin,xmax,ymin,ymax,zmin,zmax)
                    imgout_0[:,self.real_size:self.real_size*2] = self.voxel_renderer.render_img(tmpvox, \
                            self.sampling_threshold, self.render_view_id)

                    tmpvox = self.recover_voxel(self.gt_content[dxb],xmin,xmax,ymin,ymax,zmin,zmax)
                    imgout_0[:,5*self.real_size:self.real_size*6] = self.voxel_renderer.render_img(tmpvox, \
                            self.sampling_threshold, self.render_view_id)

                    tmpvox = self.recover_voxel(pred_shape,xmin,xmax,ymin,ymax,zmin,zmax)
                    imgout_0[:,4*self.real_size:5*self.real_size] = self.voxel_renderer.render_img(tmpvox, \
                            self.sampling_threshold, self.render_view_id)

                    tmpvox = self.recover_voxel(retrieved_patches,xmin,xmax,ymin,ymax,zmin,zmax)
                    imgout_0[:,2*self.real_size:3*self.real_size] = self.voxel_renderer.render_img(tmpvox, \
                            self.sampling_threshold, self.render_view_id)

                    tmpvox = self.recover_voxel(aligned_patches,xmin,xmax,ymin,ymax,zmin,zmax)
                    imgout_0[:,3*self.real_size:4*self.real_size] = self.voxel_renderer.render_img(tmpvox, \
                            self.sampling_threshold, self.render_view_id)

                    if self.dump_deform:
                        cv2.imwrite(self.dump_deform_path+"/preddeform_%s_%s_%d_%d.png"%(self.mode, self.pre_fix, epoch, \
                            self.names[dxb]), imgout_0)

                    else:
                        cv2.imwrite(self.sample_dir+"/%s_%s_%d_%d.png"%(self.mode, self.pre_fix, epoch, \
                            self.names[dxb]), imgout_0)

                if self.mode=='test':
                    continue

                self.optimizer.zero_grad()

                if re_ids.shape[1]>2:
                    sample_ids, re_c_ids, re_d_ids, out_locs, out_dis, out_trans, out_flip = self.distance_retrieve(\
                            re_ids, re_pairs, re_dis, re_trans_c, self.max_sample_num, sigma=1.0)
                else:
                    c_num = min(re_pairs.shape[0], self.max_sample_num)
                    c_ids = np.array(sorted(np.random.randint(0, re_pairs.shape[0], c_num)))

                    re_c_ids = re_ids[c_ids,0]
                    re_d_ids = re_ids[c_ids, 1]
                    out_locs = re_pairs[c_ids, :, :]
                    out_trans = re_trans_c[c_ids, 0,0:3]
                    out_flip = re_trans_c[c_ids, 0, 3]
                    out_dis = re_dis[c_ids, ]

                c_patches, d_patches, gt_patches = self.get_patch_pairs(self.gt_content[dxb], \
                    self.partial_content[dxb], input_in.cpu().numpy()[0][0], out_locs[:,0], \
                    out_locs[:,1], out_trans, out_flip,)

                c_patches_in = torch.from_numpy(c_patches).to(self.device).unsqueeze(1).float()
                d_patches_in = torch.from_numpy(d_patches).to(self.device).unsqueeze(1).float()
                gt_patches_in = torch.from_numpy(gt_patches).to(self.device).unsqueeze(1).float()
                gt_dis = torch.from_numpy(out_dis).to(self.device).float()

                d_patches_in_or = d_patches_in
                batch_size = c_patches_in.shape[0]

                pred_deform, pred_dis = self.model(c_patches_in, d_patches_in)
                center = torch.Tensor([self.patch_size,self.patch_size,self.patch_size,]).float().cuda().view(1,3).repeat(batch_size, 1)/2.0 -0.5
                scale = torch.Tensor([1.0]).view(1,1).cuda().repeat(batch_size, 1)
                angle = torch.Tensor([0, 0, 0]).float().view(1,3).cuda().repeat(batch_size, 1)

                aff_matrix = get_affine_matrix3d(pred_deform[:,0:3]*(self.patch_size/2),center, scale, angle)
                out_vox = warp_affine3d(d_patches_in_or, aff_matrix[:,:3,:], (self.patch_size,self.patch_size,\
                        self.patch_size,),)

                loss_de = torch.sum(torch.square(out_vox - gt_patches_in),(1,2,3,4))/(torch.sum(gt_patches_in, (1,2,3,4))\
                    + out_vox.sum((1,2,3,4)) - (out_vox*gt_patches_in).sum((1,2,3,4))+1e-5)
                loss_de = torch.mean(loss_de)

                iou_pred = (out_vox*gt_patches_in).sum((1,2,3,4))/(out_vox.sum((1,2,3,4)) \
                    + gt_patches_in.sum((1,2,3,4)) - (out_vox*gt_patches_in).sum((1,2,3,4)) + 1e-5)


                pred_dis = pred_dis[:, 0]
                loss_dis = torch.mean(torch.square(pred_dis - gt_dis))

                loss = loss_de + self.w_dis * loss_dis

                loss.backward()
                self.optimizer.step()
            
            if self.mode=='test':
                return 

            self.log_string("Epoch: [%d/%d] time: %.0f, loss_recons: %.5f, loss_dis: %.4f, iou:  %.5f," % \
                (epoch, training_epoch, time.time() - start_time, loss_de.item(), loss_dis.item(), \
                 torch.mean(iou_pred).item()))

            if epoch%10==0 and self.mode=='train':
                pred_shape, retrieved_patches, aligned_patches, = self.retrieve_to_full_shape_fromdis(\
                        self.gt_content[dxb], self.partial_content[dxb], input_in.cpu().numpy()[0][0], \
                        re_pairs_or, self.re_dis[dxb], self.re_trans_c[dxb], align=True,)

                imgout_0 = np.full([self.real_size, self.real_size*6], 255, np.uint8)
                xmin,xmax,ymin,ymax,zmin,zmax = self.pos_content[dxb]
                tmpvox = self.recover_voxel(input_in.cpu().numpy()[0][0],xmin,xmax,ymin,ymax,zmin,zmax)
                imgout_0[:,0:self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, \
                        self.render_view_id)

                tmpvox = self.recover_voxel(self.partial_content[dxb],xmin,xmax,ymin,ymax,zmin,zmax)
                imgout_0[:,self.real_size:self.real_size*2] = self.voxel_renderer.render_img(tmpvox, \
                        self.sampling_threshold, self.render_view_id)

                tmpvox = self.recover_voxel(self.gt_content[dxb],xmin,xmax,ymin,ymax,zmin,zmax)
                imgout_0[:,5*self.real_size:self.real_size*6] = self.voxel_renderer.render_img(tmpvox, \
                        self.sampling_threshold, self.render_view_id)

                tmpvox = self.recover_voxel(pred_shape,xmin,xmax,ymin,ymax,zmin,zmax)
                imgout_0[:,4*self.real_size:5*self.real_size] = self.voxel_renderer.render_img(tmpvox, \
                        self.sampling_threshold, self.render_view_id)

                tmpvox = self.recover_voxel(retrieved_patches,xmin,xmax,ymin,ymax,zmin,zmax)
                imgout_0[:,2*self.real_size:3*self.real_size] = self.voxel_renderer.render_img(tmpvox, \
                        self.sampling_threshold, self.render_view_id)

                tmpvox = self.recover_voxel(aligned_patches,xmin,xmax,ymin,ymax,zmin,zmax)
                imgout_0[:,3*self.real_size:4*self.real_size] = self.voxel_renderer.render_img(tmpvox, \
                        self.sampling_threshold, self.render_view_id)

                cv2.imwrite(self.sample_dir+"/%s_%d.png"%(self.mode, epoch,), imgout_0)

            if epoch%self.save_epoch==0 and self.mode=='train':
            # if epoch%self.save_epoch==0 and (self.mode=='train' or config.continue_train):
                self.save(epoch)

            self.scheduler.step()
        if self.mode=='train':
            self.save(epoch)

    def get_patch_pairs(self, gt_np, partial_np, coarse_np, c_locs, d_locs,re_trans, flip_list, align=True):
        c_starts =  c_locs*self.stride
        d_starts = d_locs*self.stride

        or_c_vox = np.zeros((c_starts.shape[0], self.patch_size, self.patch_size, self.patch_size))
        or_d_vox = np.zeros((c_starts.shape[0], self.patch_size, self.patch_size, self.patch_size))
        gt_vox = np.zeros((c_starts.shape[0], self.patch_size, self.patch_size, self.patch_size))

        for i in range(c_starts.shape[0]):
            gt_vox[i] = gt_np[c_starts[i, 0]:c_starts[i, 0]+self.patch_size, \
                c_starts[i, 1]:c_starts[i, 1]+self.patch_size, c_starts[i, 2]:c_starts[i, 2]+self.patch_size]

            or_c_vox[i] = coarse_np[c_starts[i, 0]:c_starts[i, 0]+self.patch_size, \
                c_starts[i, 1]:c_starts[i, 1] + self.patch_size, c_starts[i, 2]:c_starts[i, 2]+self.patch_size]

            d_vox = partial_np[d_starts[i, 0]:d_starts[i, 0]+self.patch_size, \
                d_starts[i, 1]:d_starts[i, 1]+self.patch_size, d_starts[i, 2]:d_starts[i, 2]+self.patch_size]

            if flip_list[i]==1:
                d_vox = np.flip(d_vox, 2)
            if align:
                # d_vox = self.move_vox(d_vox, np.around(re_trans[i]))

                mask_s = np.array(d_vox>0.3, np.float32)
                mask_t = np.array(or_c_vox[i]>0.3, np.float32)

                c_s = np.sum(self.patch_coord*np.expand_dims(mask_s, -1), axis=(0,1,2))/ (np.sum(mask_s, )+1e-5)
                c_t = np.sum(self.patch_coord*np.expand_dims(mask_t, -1), axis=(0,1,2))/ (np.sum(mask_t, )+1e-5)
                d_vox = self.move_vox(d_vox, np.around(c_t - c_s).astype(np.int32))
            or_d_vox[i] = d_vox
        return or_c_vox, or_d_vox, gt_vox

    def retrieve_to_full_shape_fromdis(self,  gt_np, partial_np, coarse_np, re_pairs, re_dis, re_trans, \
            align=True, mask_in=None, dump_deform=False):
        print('retrieve to full shape...')

        # self.model.eval()

        c_locs = re_pairs[:, 0]
        # c_locs_mask =  (c_locs % (self.patch_size//self.stride)).sum(1)
        c_locs_mask =  (c_locs % 4).sum(1)

        c_locs_mask2 = (c_locs[:,0]!=c_locs[:,0].max())*(c_locs[:,1]!=c_locs[:,1].max())*(c_locs[:,2]!=c_locs[:,2].max())
        c_locs_mask = c_locs_mask2.astype(np.int32)*c_locs_mask

        c_ids = np.where(c_locs_mask==0)[0]

        if dump_deform:
            c_ids = range(c_locs.shape[0])

        c_locs_final = c_locs[c_ids, :].astype(np.int32)
        re_pairs = re_pairs[c_ids, :, :].astype(np.int32)
        re_dis = re_dis[c_ids, :]
        flip_list = re_trans[c_ids, :, 3].astype(np.int32)
        re_trans = re_trans[c_ids, :, 0:3]
        c_num = c_locs_final.shape[0]

        max_c_num = 1200
        real_c_num = min(max_c_num, c_num)
        c_vox = np.zeros((real_c_num, self.patch_size, self.patch_size, self.patch_size))
        d_vox = np.zeros((real_c_num, self.patch_size, self.patch_size, self.patch_size))
        gt_vox = np.zeros((real_c_num, self.patch_size, self.patch_size, self.patch_size))

        out_shape = np.zeros(gt_np.shape)
        out_or_shape = np.zeros(gt_np.shape)
        out_align_shape = np.zeros(gt_np.shape)

        c_starts = c_locs_final*self.stride

        total_pred_trans = torch.zeros(re_trans.shape).to(self.device)
        total_pred_dis = torch.zeros(re_dis.shape).to(self.device)
        if dump_deform:
            total_init_trans = np.zeros((c_num, re_trans.shape[1], 4))


        # d_starts = d_locs_final*self.stride
        for k_i in range(re_trans.shape[1]):
            d_starts = re_pairs[:, 1+k_i]*self.stride

            for real_ci in range(c_num//real_c_num+1):
                s_c = real_ci*real_c_num
                e_c = min((real_ci+1)*real_c_num, c_num)


                for i in range(s_c, e_c):
                    gt_vox[i-s_c] = gt_np[c_starts[i, 0]:c_starts[i, 0]+self.patch_size, \
                        c_starts[i, 1]:c_starts[i, 1]+self.patch_size, c_starts[i, 2]:c_starts[i, 2]+self.patch_size]
                    c_vox[i-s_c] = coarse_np[c_starts[i, 0]:c_starts[i, 0]+self.patch_size, \
                            c_starts[i, 1]:c_starts[i, 1]+self.patch_size, c_starts[i, 2]:c_starts[i, 2]+self.patch_size]

                    # d_start = re_pairs[i, 1]*self.stride
                    d_start = d_starts[i]
                    d_vox_i = partial_np[d_start[0]:d_start[0]+self.patch_size, \
                        d_start[1]:d_start[1]+self.patch_size, d_start[2]:d_start[2]+self.patch_size]

                    #out_or_shape[c_starts[i, 0]:c_starts[i, 0]+self.patch_size, \
                    #    c_starts[i, 1]:c_starts[i, 1]+self.patch_size, c_starts[i, 2]:c_starts[i, 2]+self.patch_size] = d_vox_i

                    if flip_list[i, k_i]==1:
                        d_vox_i = np.flip(d_vox_i, 2)
                        if dump_deform:
                            total_init_trans[i, k_i, 3] = 1.0

                    if align:
                        mask_s = np.array(d_vox_i>0.3, np.float32)
                        mask_t = np.array(c_vox[i-s_c]>0.3, np.float32)

                        c_s = np.sum(self.patch_coord*np.expand_dims(mask_s, -1), axis=(0,1,2))/ (np.sum(mask_s, )+1e-5)
                        c_t = np.sum(self.patch_coord*np.expand_dims(mask_t, -1), axis=(0,1,2))/ (np.sum(mask_t, )+1e-5)

                        d_vox_i = self.move_vox(d_vox_i, np.around(c_t - c_s).astype(np.int32))
                        if dump_deform:
                            total_init_trans[i, k_i, 0:3] = c_t - c_s
                        # d_vox_i = self.move_vox(d_vox_i, np.around(re_trans[i]))

                    d_vox[i-s_c] = d_vox_i

                c_patches_in = torch.from_numpy(c_vox).to(self.device).unsqueeze(1).float()
                d_patches_in = torch.from_numpy(d_vox).to(self.device).unsqueeze(1).float()

                pred_deform, pred_dis = self.model(c_patches_in, d_patches_in)

                total_pred_trans[s_c:e_c, k_i, ] = pred_deform.detach()[:e_c-s_c]
                total_pred_dis[s_c:e_c, k_i] = pred_dis.detach()[:e_c-s_c,0]

        if dump_deform:
            return total_pred_trans.cpu().numpy(), total_pred_dis.cpu().numpy(), total_init_trans

        total_pred_partial_dis = {}

        sorted_dis, indices = torch.sort(total_pred_dis, dim=1)
        indices = indices.cpu().numpy()

        final_deform = torch.zeros(c_num, 3).to(self.device)
        final_d_vox =  np.zeros((c_num, self.patch_size, self.patch_size, self.patch_size))

        for i in range(c_num):
            final_deform[i] = total_pred_trans[i, indices[i, 0]]
            d_start = re_pairs[i, 1+indices[i,0]]*self.stride
            d_vox_i = partial_np[d_start[0]:d_start[0]+self.patch_size, \
                    d_start[1]:d_start[1]+self.patch_size, d_start[2]:d_start[2]+self.patch_size]

            out_or_shape[c_starts[i, 0]:c_starts[i, 0]+self.patch_size, \
                c_starts[i, 1]:c_starts[i, 1]+self.patch_size, c_starts[i, 2]:c_starts[i, 2]+self.patch_size] = d_vox_i

            if flip_list[i, indices[i, 0]]==1:
                d_vox_i = np.flip(d_vox_i, 2)

            if align:
                mask_s = np.array(d_vox_i>0.3, np.float32)
                c_vox_i = coarse_np[c_starts[i, 0]:c_starts[i, 0]+self.patch_size, \
                        c_starts[i, 1]:c_starts[i, 1]+self.patch_size, c_starts[i, 2]:c_starts[i, 2]+self.patch_size]
                mask_t = np.array(c_vox_i>0.3, np.float32)

                c_s = np.sum(self.patch_coord*np.expand_dims(mask_s, -1), axis=(0,1,2))/ (np.sum(mask_s, )+1e-5)
                c_t = np.sum(self.patch_coord*np.expand_dims(mask_t, -1), axis=(0,1,2))/ (np.sum(mask_t, )+1e-5)

                d_vox_i = self.move_vox(d_vox_i, np.around(c_t - c_s).astype(np.int32))
                # d_vox_i = self.move_vox(d_vox_i, np.around(re_trans[i]))

            out_align_shape[c_starts[i, 0]:c_starts[i, 0]+self.patch_size, \
                c_starts[i, 1]:c_starts[i, 1]+self.patch_size, c_starts[i, 2]:c_starts[i, 2]+self.patch_size] = d_vox_i
            final_d_vox[i] = d_vox_i

        d_patches_in = torch.from_numpy(final_d_vox).to(self.device).unsqueeze(1).float()

        batch_size = d_patches_in.shape[0]
        center = torch.Tensor([self.patch_size,self.patch_size,self.patch_size,]).float().cuda().view(1,3).repeat(batch_size, 1)/2.0 -0.5
        angle = torch.Tensor([0, 0, 0]).float().view(1,3).cuda().repeat(batch_size, 1)
        scale = torch.Tensor([1.0]).view(1,1).cuda().repeat(batch_size, 1)

        aff_matrix = get_affine_matrix3d(final_deform[:,0:3]*(self.patch_size/2),center, scale, angle)
        out_vox = warp_affine3d(d_patches_in, aff_matrix[:,:3,:], (self.patch_size,self.patch_size,self.patch_size,),)
        out_vox = out_vox.detach().cpu().numpy()[:,0]

        for i in range(c_num):

            loc_off= (c_locs_final[i]%4).sum()
            loc_off2 = (c_locs_final[i,0]!=c_locs[:,0].max())*(c_locs_final[i,1]!=c_locs[:,1].max())*(c_locs_final[i,2]!=c_locs[:,2].max())
            #if loc_off!=0 and loc_off2.sum()==1:
            #    continue
            if loc_off==0 or loc_off2==0:
                out_shape[c_starts[i, 0]:c_starts[i, 0]+self.patch_size, \
                    c_starts[i, 1]:c_starts[i, 1]+self.patch_size, c_starts[i, 2]:c_starts[i, 2]+self.patch_size] = out_vox[i]

            if tuple(c_locs_final[i]) in total_pred_partial_dis.keys():
                dis_p = total_pred_partial_dis[tuple(c_locs_final[i])]
            else:
                dis_p = 100.0

        return out_shape, out_or_shape, out_align_shape

    def find_loc_id(self, loc, loc_total):
        loc_dif = np.sum(np.square((loc - loc_total)), axis=1)
        if loc_dif.min()>0:
            return -1
        else:
            c_id = np.where(loc_dif==0)[0][0]
            return c_id

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


    def move_vox(self, vox, shift):
        out_vox = np.zeros(vox.shape)
        start_1 = np.maximum(shift, np.zeros(shift.shape)).astype(np.int32)
        start_2 = np.maximum(np.zeros(shift.shape), -shift).astype(np.int32)

        vsize = vox.shape[0]
        span = vsize - np.abs(shift)
        end_1= np.minimum([start_1[0]+span[0], start_1[1]+span[1], start_1[2]+span[2]], \
            [vsize, vsize, vsize]).astype(np.int32)
        r_span = np.array([end_1[0]-start_1[0], end_1[1]-start_1[1], end_1[2]-start_1[2],])

        out_vox[start_1[0]:end_1[0], start_1[1]:end_1[1],start_1[2]:end_1[2]]= \
            vox[start_2[0]:start_2[0]+r_span[0], start_2[1]:start_2[1]+r_span[1], start_2[2]:start_2[2]+r_span[2]]
        return out_vox


    def move_vox_torch(self, vox, shift):
        out_vox = torch.zeros(vox.shape).to(self.device)
        start_1 = np.maximum(shift, np.zeros(shift.shape)).astype(np.int32)
        start_2 = np.maximum(np.zeros(shift.shape), -shift).astype(np.int32)

        vsize = vox.shape[0]

        span = vsize - np.abs(shift)
        end_1= np.minimum([start_1[0]+span[0], start_1[1]+span[1], start_1[2]+span[2]], \
             [vsize, vsize, vsize,]).astype(np.int32)
        r_span = np.array([end_1[0]-start_1[0], end_1[1]-start_1[1], end_1[2]-start_1[2],])

        out_vox[start_1[0]:end_1[0], start_1[1]:end_1[1],start_1[2]:end_1[2]]= \
            vox[start_2[0]:start_2[0]+r_span[0], start_2[1]:start_2[1]+r_span[1], start_2[2]:start_2[2]+r_span[2]]
        return out_vox

    def volume_to_point_cloud_to_num(self,vol, coord, target_num=768):
        """
        vol: [vsize, vsize, vsize], torch
        """
        vsize = vol.shape[0]

        surf_mask = pt_mask
        or_points= coord[surf_mask.astype(np.bool)].astype(np.float32)
        num_points = surf_mask.sum()

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
