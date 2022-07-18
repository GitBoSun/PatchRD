from functools import partial, partialmethod
import os
import time
import math
import random
import numpy as np
import cv2
import h5py
import open3d as o3d

from glob import glob
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import sobel
from skimage import measure

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from utils import *
from models import *

from psbody.mesh import Mesh
from kornia.geometry import get_affine_matrix3d, warp_affine3d

import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D
chamLoss = ChamferDistancePytorch.chamfer3D.dist_chamfer_3D.chamfer_3DDist()

class MODEL_JOINT(object):
    def __init__(self, config):
        self.real_size = 256
        self.mask_margin = 8

        self.g_dim = config.g_dim
        self.gw_dim = config.gw_dim

        self.z_dim = 3

        self.input_size = config.input_size # 64
        self.output_size = config.output_size # 128
        self.patch_size = 18
        self.stride = 4
        self.pt_num = 512
        self.wd_size = config.wd_size
        self.loc_size = config.loc_size
        self.K = 8
        self.sample_step = config.sample_step
        self.max_wd_num = config.max_wd_num
        self.trans_limit = config.trans_limit

        self.use_mean_x = False
        self.compute_cd = config.compute_cd
        self.dump_mesh = False

        self.lr = config.lr
        self.w_r = config.w_r
        self.w_s = config.w_s

        self.upsample_rate = 1
        self.asymmetry = True
        self.save_epoch = 1
        self.eval_epoch = 1
        self.img_epoch = 1
        self.start_epoch = 0

        self.mode = config.mode
        self.gua_filter = True
        self.pred_coarse = True
        self.small_dataset = config.small_dataset
        self.top_num = 8

        self.sampling_threshold = 0.3
        self.render_view_id = 0
        self.voxel_renderer = voxel_renderer(self.real_size)

        self.checkpoint_dir = config.checkpoint_dir
        self.data_dir = config.data_dir
        self.c3d_data_dir = '/scratch/cluster/bosun/detailization/data/completion3d/train/gt/%s'%(self.data_dir[-9:])

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

        x_ = range(self.output_size)
        y_ = range(self.output_size)
        z_ = range(self.output_size)
        yy, xx, zz = np.meshgrid(x_,y_,z_)
        self.shape_coord = np.concatenate((np.expand_dims(xx, -1), np.expand_dims(yy, -1),np.expand_dims(zz, -1)), -1)
        #load data

        vox_name = "/model.binvox" 

        print("preprocessing - start")
        if os.path.exists("splits/"+self.data_content+"_train.txt"):
            #load data
            # fin = open("splits/"+self.data_content+"_%s.txt"%(self.mode))
            fin = open("splits/"+self.data_content+"_train.txt")

            self.dataset_names = [name.strip() for name in fin.readlines()]
            fin.close()

            self.dataset_len = len(self.dataset_names)
            self.dataset_len = 1000
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
            self.re_pairs = []
            self.re_trans = []
            self.pred_trans = []
            self.init_trans = []
            self.refine_trans = []
            self.pred_dis = []
            self.shape_names = []

            names = range(self.dataset_len)
            self.names = names

            self.imgout_0 = np.full([self.real_size*4, self.real_size*4*3], 255, np.uint8)

            if config.train_joint and self.mode=='train':
                for i in range(self.dataset_len):
                    print("preprocessing content - "+str(i+1)+"/"+str(self.dataset_len))
                    tmp_raw = get_vox_from_binvox_1over2(os.path.join(self.data_dir,self.dataset_names[names[i]]+vox_name)).astype(np.uint8)
                    xmin,xmax,ymin,ymax,zmin,zmax = self.get_voxel_bbox(tmp_raw)

                    pred_files = glob(os.path.join(self.dump_deform_path, self.dataset_names[names[i]], "pred_deform_*.npz"))
                    if len(pred_files)==0:
                        print('not exists', os.path.join(self.dump_deform_path, self.dataset_names[names[i]]))
                        self.dataset_len = self.dataset_len -1
                        continue
                    else:
                        pred_f = pred_files[-1]

                    dump_files = glob(os.path.join(self.dump_deform_path, self.dataset_names[names[i]], 'dump_deform_wc_top*.npz'))
                    if len(dump_files)==0:
                        print('not exist')
                        self.dataset_len = self.dataset_len -1
                        continue
                    else:
                        dump_f = dump_files[-1]

                    a = np.load(dump_f, allow_pickle=True)
                    re_ids = a['re_ids']
                    re_dis = a['re_dis']*10.0
                    crop_locs = a['crop_locs']
                    re_trans = a['re_trans']

                    re_trans_c = a['re_trans_c']
                    coarse_out = a['coarse_shape']

                    b = np.load(pred_f,allow_pickle=True)
                    pred_trans = b['pred_trans']
                    pred_dis = b['pred_dis']
                    init_trans = b['init_trans']
                    pred_dis = np.clip(pred_dis, 0, 1.0)

                    tmp = self.crop_voxel(tmp_raw,xmin,xmax,ymin,ymax,zmin,zmax)

                    gt_voxel = tmp
                    tmp_input, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)
                    partial_shape, partial_mask =  self.random_crop(tmp, c_locs=crop_locs)
                    partial_mask = tmp_mask
                    

                    if self.pred_coarse:
                        tmp_input = coarse_out
                        input_in = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()
                        mask_in = F.max_pool3d((input_in>self.sampling_threshold).float(), kernel_size=self.patch_size, stride=self.stride, padding=0)
                        tmp_mask = mask_in.cpu().numpy()[0,0]

                        partial_shape2 = gaussian_filter(partial_shape.astype(np.float32), sigma=1)
                        partial_in2 = torch.from_numpy(partial_shape2).to(self.device).unsqueeze(0).unsqueeze(0).float()
                        mask_partial_in = F.max_pool3d((partial_in2>self.sampling_threshold).float(), \
                                kernel_size = self.patch_size, stride = self.stride, padding = 0)
                        partial_mask = mask_partial_in.cpu().numpy()[0,0]

                    re_ids = re_ids.astype(np.int32)
                    total_c_locs, total_d_locs = self.filter_locs(tmp_mask, partial_mask)
                    re_pairs, step_ids = self.get_locs(re_ids, total_c_locs, total_d_locs, step=self.sample_step)

                    re_pairs = re_pairs[:, :self.K+1,:]
                    valid_ids = step_ids
                    re_dis = re_dis[valid_ids, :self.K]
                    re_trans = re_trans[valid_ids, :self.K, :]
                    re_trans_c = re_trans_c[valid_ids,:self.K,:]
                    pred_dis = pred_dis[valid_ids,:self.K]
                    pred_trans = pred_trans[valid_ids,:self.K,:]
                    init_trans = init_trans[valid_ids,:self.K,:]
                    pred_trans_2 = np.zeros(init_trans.shape)
                    pred_trans_2[:,:,0:3] = pred_trans[:,:,0:3]*self.patch_size/2 + np.around(init_trans[:,:,0:3])
                    pred_trans_2[:,:,3] = init_trans[:,:,3]
                    pred_trans_2 = np.clip(pred_trans_2, -self.patch_size, self.patch_size )

                    self.input_content.append(tmp_input)
                    self.gt_content.append(gt_voxel)
                    self.mask_content.append(tmp_mask)
                    self.partial_content.append(partial_shape)
                    self.partial_mask.append(partial_mask)
                    self.pos_content.append([xmin,xmax,ymin,ymax,zmin,zmax])
                    self.pred_trans.append(pred_trans_2)
                    self.init_trans.append(init_trans)
                    self.refine_trans.append(pred_trans)
                    self.re_pairs.append(re_pairs)
                    self.pred_dis.append(pred_dis)

                    self.shape_names.append(self.dataset_names[self.names[i]])

                    img_y = i//4
                    img_x = (i%4)*3+2
                    if img_y<4:
                        tmpvox = self.recover_voxel(gt_voxel,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

                    img_x = (i%4)*3
                    if img_y<4:
                        tmpvox = self.recover_voxel(tmp_input,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

                    img_x = (i%4)*3+1
                    if img_y<4:
                        tmpvox = self.recover_voxel(partial_shape,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

                    if img_y<4:
                        cv2.imwrite(self.sample_dir+"/a_content_train.png", self.imgout_0)
        else:
            print("ERROR: cannot load dataset txt: "+"splits/"+self.data_content+"_train.txt")
            exit(-1)

        if os.path.exists("splits/"+self.data_content+"_test.txt"):
            #load data

            fin = open("splits/"+self.data_content+"_test.txt")

            self.dataset_names_test = [name.strip() for name in fin.readlines()]
            fin.close()

            self.test_dataset_len = len(self.dataset_names_test)
            self.test_dataset_len = 100
            if self.small_dataset:
                self.test_dataset_len = 4

            self.mask_test  = []
            self.input_test = []
            self.partial_test = []
            self.partial_mask_test = []
            self.gt_test = []
            self.pos_test = []

            self.re_ids_test = []
            self.re_pairs_test = []
            self.re_trans_test = []
            self.pred_trans_test = []
            self.init_trans_test = []
            self.refine_trans_test = []
            self.pred_dis_test = []
            self.shape_names_test = []
            self.gt_pc = []
            self.partial_pc = []
            self.vsizes = []
            self.pre_fix = ''

            self.test_names = range(self.test_dataset_len)

            vsize_file = os.path.join(self.dump_deform_path, 'vsizes.npy')
            self.dump_deform_path = config.dump_deform_path
            loc_file = os.path.join(self.dump_deform_path, 'crop_locs.npy')

            #self.test_names = test_names
            self.test_dataset_len = len(self.test_names)

            self.imgout_0 = np.full([self.real_size*4, self.real_size*4*3], 255, np.uint8)
            if config.train_joint:
                for i in range(self.test_dataset_len):
                    print("preprocessing test - "+str(i+1)+"/"+str(self.test_dataset_len))
                    tmp_raw = get_vox_from_binvox_1over2(os.path.join(self.data_dir,self.dataset_names_test[self.test_names[i]]+vox_name)).astype(np.uint8)

                    xmin,xmax,ymin,ymax,zmin,zmax = self.get_voxel_bbox(tmp_raw)
                    self.vsizes.append(np.array([xmax-xmin, ymax-ymin, zmax-zmin]))
                    pred_files = glob(os.path.join(self.dump_deform_path, self.dataset_names_test[self.test_names[i]], "pred_deform_*.npz"))
                    if len(pred_files)==0:
                        print('not exists')
                        print(os.path.join(self.dump_deform_path, self.dataset_names_test[self.test_names[i]],))
                        self.test_dataset_len = self.test_dataset_len -1
                        continue
                    else:
                        pred_f = pred_files[-1]

                    dump_files = glob(os.path.join(self.dump_deform_path, self.dataset_names_test[self.test_names[i]], 'dump_deform_wc_top*.npz'))
                    if len(dump_files)==0:
                        print('not exist', os.path.join(self.dump_deform_path, self.dataset_names_test[self.test_names[i]],))
                        self.test_dataset_len = self.test_dataset_len -1
                        continue
                    else:
                        dump_f = dump_files[-1]

                    a =  np.load(dump_f,allow_pickle=True)

                    re_ids = a['re_ids']
                    re_dis = a['re_dis']*10.0
                    crop_locs = a['crop_locs']
                    #self.c_locs.append(crop_locs)

                    re_trans = a['re_trans']

                    re_trans_c = a['re_trans_c']
                    coarse_out = a['coarse_shape']
                    #print(coarse_out.shape)
                    b = np.load(pred_f,allow_pickle=True)
                    pred_trans = b['pred_trans']
                    pred_dis = b['pred_dis']
                    init_trans = b['init_trans']
                    pred_dis = np.clip(pred_dis, 0, 1.0)

                    tmp = self.crop_voxel(tmp_raw,xmin,xmax,ymin,ymax,zmin,zmax)
                    gt_voxel = tmp
                    
                    tmp_input, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)
                    partial_shape, partial_mask =  self.random_crop(tmp, c_locs=crop_locs)
                    partial_mask = tmp_mask

                    if self.compute_cd:
                        path1 = os.path.join(self.c3d_data_dir, self.dataset_names_test[self.test_names[i]]+'.h5')
                        if not os.path.exists(path1):
                            path1 = path1.replace('train', 'val')
                        if not os.path.exists(path1):
                            self.test_dataset_len = self.test_dataset_len -1
                            print('pc non_exits')
                            continue
                        fh5_1 = h5py.File(path1, "r")
                        pc1 = fh5_1['data'][:]

                    if self.pred_coarse:
                        tmp_input = coarse_out
                        input_in = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()
                        mask_in = F.max_pool3d((input_in>self.sampling_threshold).float(), kernel_size = self.patch_size, stride = self.stride, padding = 0)
                        tmp_mask = mask_in.cpu().numpy()[0,0]

                        partial_shape2 = gaussian_filter(partial_shape.astype(np.float32), sigma=1)
                        partial_in2 = torch.from_numpy(partial_shape2).to(self.device).unsqueeze(0).unsqueeze(0).float()
                        mask_partial_in = F.max_pool3d((partial_in2>self.sampling_threshold).float(), \
                                kernel_size = self.patch_size, stride = self.stride, padding = 0)
                        partial_mask = mask_partial_in.cpu().numpy()[0,0]

                    re_ids = re_ids.astype(np.int32)
                    total_c_locs, total_d_locs = self.filter_locs(tmp_mask, partial_mask)

                    re_pairs,step_ids = self.get_locs(re_ids, total_c_locs, total_d_locs, step=self.sample_step)

                    valid_ids = step_ids
                    re_dis = re_dis[valid_ids, :self.K]
                    re_trans = re_trans[valid_ids, :self.K, :]
                    re_trans_c = re_trans_c[valid_ids,:self.K,:]
                    try:
                        pred_dis = pred_dis[valid_ids,:self.K]
                    except:
                        print("??")
                        continue
                    pred_trans = pred_trans[valid_ids,:self.K,:]
                    init_trans = init_trans[valid_ids,:self.K,:]
                    pred_trans_2 = np.zeros(init_trans.shape)
                    pred_trans_2[:,:,3] =  init_trans[:,:,3]
                    pred_trans_2[:,:,0:3] = pred_trans[:,:,0:3]*self.patch_size/2 + np.around(init_trans[:,:,0:3])
                    pred_trans_2 = np.clip(pred_trans_2, -self.patch_size, self.patch_size )

                    #if self.gua_filter:
                    #   tmp_input = gaussian_filter(tmp_input.astype(np.float32), sigma=1)
                    #    partial_shape =  gaussian_filter(partial_shape.astype(np.float32), sigma=1)
                    if self.compute_cd:
                        self.gt_pc.append(pc1)
                    if self.dump_mesh:
                        self.gt_meshes.append(new_mesh)

                    self.input_test.append(tmp_input)
                    self.gt_test.append(gt_voxel)
                    self.mask_test.append(tmp_mask)
                    self.partial_test.append(partial_shape)
                    self.partial_mask_test.append(partial_mask)
                    self.pos_test.append([xmin,xmax,ymin,ymax,zmin,zmax])
                    self.re_trans_test.append(re_trans)
                    self.pred_trans_test.append(pred_trans_2)
                    self.init_trans_test.append(init_trans)
                    self.refine_trans_test.append(pred_trans)

                    self.re_pairs_test.append(re_pairs)
                    self.pred_dis_test.append(pred_dis)

                    self.shape_names_test.append(self.dataset_names_test[self.test_names[i]])

                    img_y = i//4
                    img_x = (i%4)*3+2
                    if img_y<4:
                        tmpvox = self.recover_voxel(gt_voxel,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

                    img_x = (i%4)*3
                    if img_y<4:
                        tmpvox = self.recover_voxel(tmp_input,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

                    img_x = (i%4)*3+1
                    if img_y<4:
                        tmpvox = self.reshape_to_patch_size(partial_shape, np.zeros((256,256,256)))
                        #tmpvox = self.recover_voxel(partial_shape,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

                    if img_y<4:
                        cv2.imwrite(self.sample_dir+"/a_content_test.png", self.imgout_0)

        print('done')


        self.deformer = JointDeformer(self.g_dim, self.gw_dim, self.max_wd_num, self.loc_size, \
                    wd_size=self.wd_size, use_mean_x=self.use_mean_x)

        self.deformer.to(self.device)
        self.optimizer_g = torch.optim.Adam(self.deformer.parameters(), lr=config.lr)
        self.scheduler_g = torch.optim.lr_scheduler.StepLR(self.optimizer_g, 2, gamma=0.99)

        self.max_to_keep = 4
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
        if self.input_size==64:
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

    def random_crop(self, vox, crop_size=20, c_range=20, prob=None, c_locs=None, par=False, par2=False, par3=False):
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

        if p<=0.4 and c_locs is None:
            cnt=0
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
                cnt+=1
                if cnt>10:
                    break

        elif c_locs is None:
            if self.data_content=='content_chair':
                cnt=0
                while(edges[loc_starts[0]+csize[0]//2,loc_starts[1]+csize[1]//2,loc_starts[2]+csize[2]//2,]<0.4 \
                    or vox[loc_starts[0]:loc_starts[0]+csize[0], loc_starts[1]:loc_starts[1]+csize[1],\
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
                cnt = 0
                while(edges[loc_starts[0]+csize[0]//2,loc_starts[1]+csize[1]//2,loc_starts[2]+csize[2]//2,]<0.2 \
                    or vox[loc_starts[0]:loc_starts[0]+csize[0], loc_starts[1]:loc_starts[1]+csize[1],\
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

    def get_locs(self, re_ids, c_locs, d_locs, step=1):
        if step==1:
            out_locs = np.zeros((re_ids.shape[0], re_ids.shape[1], 3))
            out_locs[:, 0, :] = c_locs[re_ids[:,0]]
            for i in range(out_locs.shape[1]-1):
                out_locs[:, i+1, :] = d_locs[re_ids[:,i+1]]
            step_ids = range(re_ids.shape[0])
        else:
            re_c_locs = c_locs[re_ids[:,0]]
            c_locs_mask = (re_c_locs % 2).sum(1)

            c_locs_mask2 = (re_c_locs[:,0]!=re_c_locs[:,0].max())*(\
                re_c_locs[:,1]!=re_c_locs[:,1].max())*(re_c_locs[:,2]!=re_c_locs[:,2].max())
            c_locs_mask = c_locs_mask2.astype(np.int32)*c_locs_mask

            c_locs_final = re_c_locs[c_locs_mask==0]
            c_ids = np.where(c_locs_mask==0)[0]
            step_ids = c_ids
            out_locs = np.zeros((c_locs_final.shape[0], re_ids.shape[1], 3))
            out_locs[:,0,:] = c_locs_final
            for i in range(out_locs.shape[1]-1):
                out_locs[:, i+1, :] = d_locs[re_ids[c_ids,i+1]]

        return out_locs.astype(np.int32), step_ids

    def compute_iou_np(self, recons, gt, thresh=0.4):
        x = (recons>thresh).astype(np.float32)

        y = (gt>thresh).astype(np.float32)

        intersection = np.sum(x*y)
        union = np.sum(x) + np.sum(y) - np.sum(x*y)
        iou = intersection/(union+1e-5)
        return iou

    def compute_iou(self, recons, gt, thresh=0.4):
        x = (recons>thresh).float()

        y = (gt>thresh).float()

        intersection = torch.sum(x*y)
        union = torch.sum(x) + torch.sum(y) - torch.sum(x*y)
        iou = intersection/(union+1e-5)
        return iou

    def get_crop_ids(self, re_pairs, loc_starts, loc_ends):
        c_locs = re_pairs[:, 0,:]
        c_starts = c_locs*self.stride
        valid_ids = []
        for i in range(c_starts.shape[0]):
            if (c_starts[i,0]<=loc_ends[0]-self.wd_size and c_starts[i,1]<=loc_ends[1]-self.wd_size and \
                c_starts[i,2]<=loc_ends[2]-self.wd_size) and \
                (c_starts[i,0]+self.patch_size>=loc_starts[0]+self.wd_size and \
                c_starts[i,1]+self.patch_size>=loc_starts[1]+self.wd_size \
                and c_starts[i,2]+self.patch_size>=loc_starts[2]+self.wd_size):

                valid_ids.append(i)
        return valid_ids


    def dump_patches(self, partial_np, coarse_np, re_pairs, re_trans, dp_size=26, pre_fix=''):
        out_path = 'tmp_patch_meshes_fig4'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        pad_size = (34-self.patch_size)//2

        K = self.K
        n_pairs = re_pairs.shape[0]
        select_ids = np.random.randint(0, n_pairs, size=20)
        
        for i in range(n_pairs):
        #for i in select_ids:
            d_start = re_pairs[i, 0]*self.stride
            if d_start[0]>40 or d_start[1]<72 or d_start[2]>40:
                continue
            c_vox_i = coarse_np[max(d_start[0]-pad_size,0):d_start[0]+self.patch_size + pad_size, \
                    max(d_start[1]-pad_size,0):d_start[1]+self.patch_size+pad_size, max(d_start[2]-pad_size,0):d_start[2]+self.patch_size+pad_size]
            #c_vox_i = coarse_np[0:50, 72:, 4:43 ]
            m_vox = -gaussian_filter((c_vox_i>0.4).astype(np.float32), sigma=0.8)
            verts, faces, normals, values = measure.marching_cubes(m_vox, -0.25)
            v = verts-verts.min(0)
            v = v/v.max()
            v = v-(v.max(0)+v.min(0))/2
            mesh = Mesh(v=verts, f=faces)
            mesh.write_ply(os.path.join(out_path, '%s_%d_or_%d_%d_%d.ply'%(pre_fix, i, d_start[0],d_start[1],d_start[2] )))

            for k_i in range(K):
                d_start = re_pairs[i, 1+k_i]*self.stride

                d_vox_i = partial_np[max(d_start[0]-pad_size,0):d_start[0]+self.patch_size + pad_size, \
                    max(d_start[1]-pad_size,0):d_start[1]+self.patch_size+pad_size, max(d_start[2]-pad_size,0):d_start[2]+self.patch_size+pad_size]
                d_vox_i = self.reshape_to_patch_size(d_vox_i, np.zeros((34,34,34)))

                m_vox = -gaussian_filter((d_vox_i>0.4).astype(np.float32), sigma=0.8)  # 0.8, 0.2, 0.35 # 0.7, 0.25
                verts, faces, normals, values = measure.marching_cubes(m_vox, -0.25)
                v = verts-verts.min(0)
                v = v/v.max()
                v = v-(v.max(0)+v.min(0))/2
                mesh = Mesh(v=verts, f=faces)
                mesh.write_ply(os.path.join(out_path, '%s_%d_re%d_%d_%d_%d.ply'%(pre_fix, i, k_i, d_start[0],d_start[1],d_start[2])))


    def get_overlapping_windows_rand_new(self, partial_np, re_pairs, re_trans, refine_trans, re_dis, starts=None, ends=None,):
        K = self.K
        cell_size = self.wd_size*self.loc_size

        vx, vy, vz = partial_np.shape
        pad_x = max(0, cell_size-vx)
        pad_y = max(0, cell_size-vy)
        pad_z = max(0, cell_size-vz)

        if starts is None:
            loc_starts = np.zeros(3)
            valid_ids=[]
            while(len(valid_ids)<20):
                loc_starts[0] = (np.random.randint(0, vx-self.wd_size*self.loc_size)//self.wd_size)*self.wd_size
                loc_starts[1] = (np.random.randint(0, vy-self.wd_size*self.loc_size)//self.wd_size)*self.wd_size
                loc_starts[2] = (np.random.randint(0, vz-self.wd_size*self.loc_size)//self.wd_size)*self.wd_size
                loc_ends = loc_starts + self.wd_size*self.loc_size
                loc_starts = loc_starts.astype(np.int32)
                loc_ends = loc_ends.astype(np.int32)
                valid_ids = self.get_crop_ids(re_pairs, loc_starts, loc_ends)
        else:
            loc_starts = starts
            loc_ends = ends
            valid_ids = self.get_crop_ids(re_pairs, loc_starts, loc_ends)

        re_pairs_i = re_pairs[valid_ids, :,:]
        re_trans_i = re_trans[valid_ids, :,:]
        refine_trans_i = refine_trans[valid_ids,:,:]


        re_dis_i = re_dis[valid_ids, :]
        indices = np.argsort(re_dis_i, axis=1)
        re_dis_i = np.clip(re_dis_i, 0, 1)



        # patch size 18
        # retrival stage: self.K is 8, sliding window stride 4, self.K*(128/4)^3
        # deformation stage: K is flexible, sliding stride is 8, K*(40/8)^3

        c_num = re_pairs_i.shape[0]
        #print(c_num) # c_num = (40/8)^3
        c_locs = re_pairs_i[:,0,:]
        c_starts = c_locs*self.stride
        c_starts = c_starts.astype(np.int32)
        if c_num>0:
            K = min(self.K, self.max_wd_num//c_num+ 1)


        windows = np.zeros((c_num*K, self.wd_size*self.loc_size,self.wd_size*self.loc_size,self.wd_size*self.loc_size))
        centers = np.zeros((c_num*K,3))
        loc_mask = np.zeros((c_num*K, self.loc_size, self.loc_size, self.loc_size))
        or_window = np.zeros((self.wd_size*self.loc_size,self.wd_size*self.loc_size, self.wd_size*self.loc_size))
        init_blended_window = np.zeros((self.wd_size*self.loc_size,self.wd_size*self.loc_size, self.wd_size*self.loc_size))
        init_X = np.zeros((c_num*K, self.loc_size, self.loc_size, self.loc_size))

        mask = np.zeros((vx, vy, vz))
        mask[int(loc_starts[0]):min(vx,int(loc_ends[0])), int(loc_starts[1]):min(vy,int(loc_ends[1])), \
            int(loc_starts[2]):min(vz,int(loc_ends[2]))] = 1.0

        batch_size = c_num*K
        center = torch.Tensor([self.patch_size,self.patch_size,self.patch_size,]).float().cuda().view(1,3).repeat(batch_size, 1)/2.0 -0.5
        angle = torch.Tensor([0, 0, 0]).float().view(1,3).cuda().repeat(batch_size, 1)
        scale = torch.Tensor([1.0]).view(1,1).cuda().repeat(batch_size, 1)
        total_d_patches = np.zeros((c_num*K, self.patch_size,self.patch_size,self.patch_size))
        total_trans = np.zeros((c_num*K, 3))

        for i in range(c_num):
            for k_i in range(K):
                d_start = re_pairs_i[i, 1+indices[i, k_i]]*self.stride
                d_vox_i = partial_np[d_start[0]:d_start[0]+self.patch_size, \
                        d_start[1]:d_start[1]+self.patch_size, d_start[2]:d_start[2]+self.patch_size]
                if re_trans_i[i, indices[i,k_i], 3]==1:
                    d_vox_i = np.flip(d_vox_i, 2)
                # print(re_trans_i[i,indices[i,k_i], 0:3])
                d_vox_i = self.move_vox(d_vox_i, np.around(re_trans_i[i,indices[i,k_i], 0:3]))
                total_d_patches[i*K+k_i] = d_vox_i
                total_trans[i*K+k_i] = refine_trans_i[i,indices[i,k_i],0:3]

        if c_num>0:
            total_trans = torch.from_numpy(total_trans).float().to(self.device)
            total_d_patches = torch.from_numpy(total_d_patches).float().unsqueeze(1).to(self.device)
            aff_matrix = get_affine_matrix3d(total_trans*(self.patch_size/2),center, scale, angle)
            total_d_patches = warp_affine3d(total_d_patches, aff_matrix[:,:3,:], (self.patch_size,self.patch_size,self.patch_size,),)
            total_d_patches = total_d_patches.detach().cpu().numpy()[:,0]

        for i in range(c_num):
            for k_i in range(K):
                #windows[i*self.K+k_i, c_starts[i, 0]:c_starts[i, 0]+self.patch_size, c_starts[i, 1]:c_starts[i, 1]\
                #        +self.patch_size, c_starts[i, 2]:c_starts[i, 2]+self.patch_size] = d_vox_i
                sx = max(c_starts[i, 0]-loc_starts[0], 0)
                sy = max(c_starts[i, 1]-loc_starts[1], 0)
                sz = max(c_starts[i, 2]-loc_starts[2], 0)
                ex = min(loc_ends[0]-loc_starts[0], c_starts[i, 0]-loc_starts[0]+self.patch_size)
                ey = min(loc_ends[1]-loc_starts[1], c_starts[i, 1]-loc_starts[1]+self.patch_size)
                ez = min(loc_ends[2]-loc_starts[2], c_starts[i, 2]-loc_starts[2]+self.patch_size)

                dsx = max(0, loc_starts[0]-c_starts[i, 0])
                dsy = max(0, loc_starts[1]-c_starts[i, 1])
                dsz = max(0, loc_starts[2]-c_starts[i, 2])

                windows[k_i*c_num+i, sx:ex, sy:ey, sz:ez] = total_d_patches[i*K+k_i][dsx:dsx+ex-sx, dsy:dsy+ey-sy, dsz:dsz+ez-sz]
                if k_i==0:
                    or_window[sx:ex, sy:ey, sz:ez] = total_d_patches[i*K+k_i][dsx:dsx+ex-sx, dsy:dsy+ey-sy, dsz:dsz+ez-sz]

                #centers[i*self.K+k_i] = [c_starts[i,0]+self.patch_size/2, c_starts[i,1]+self.patch_size/2, \
                #        c_starts[i,2]+self.patch_size/2,]

                centers[k_i*c_num+i] =[(ex+sx)/2, (ey+sy)/2,(ez+sz)/2]

            for ix in range(c_starts[i,0], c_starts[i,0]+self.patch_size-self.wd_size):
                for iy in range(c_starts[i,1], c_starts[i,1]+self.patch_size-self.wd_size):
                    for iz in range(c_starts[i,2], c_starts[i,2]+self.patch_size-self.wd_size):
                        if ix%self.wd_size!=0 or iy%self.wd_size!=0 or iz%self.wd_size!=0:
                            continue
                        if ix<loc_starts[0] or iy<loc_starts[1] or iz<loc_starts[2] or \
                                ix>loc_ends[0]-self.wd_size or iy>loc_ends[1]-self.wd_size \
                                or iz>loc_ends[2]-self.wd_size:
                            continue
                        wx = (ix-loc_starts[0])//self.wd_size
                        wy = (iy-loc_starts[1])//self.wd_size
                        wz = (iz-loc_starts[2])//self.wd_size
                        for k_id in range(K):
                            loc_mask[k_id*c_num+i, wx, wy, wz] = 1
                            init_X[k_id*c_num+i, wx, wy, wz] = 1-re_dis_i[i, indices[i, k_id]]



        # init_X1 = init_X/(np.expand_dims(np.sum(np.exp(init_X), axis=0),0)+1e-5)
        #init_X = np.exp(init_X)*loc_mask/(np.expand_dims(np.sum(np.exp(init_X)*loc_mask, axis=0),0)+1e-5)
        init_X = init_X*loc_mask/(np.expand_dims(np.sum(init_X*loc_mask, axis=0),0)+1e-5)
        for bx in range(self.loc_size):
           for by in range(self.loc_size):
               for bz in range(self.loc_size):
                   init_blended_window[bx*self.wd_size:(bx+1)*self.wd_size,by*self.wd_size:(by+1)*self.wd_size,\
                       bz*self.wd_size:(bz+1)*self.wd_size,] = np.sum(np.reshape(init_X[:,bx,by,bz], (-1,1,1,1))\
                       *windows[:,bx*self.wd_size:(bx+1)*self.wd_size,by*self.wd_size:(by+1)*self.wd_size,\
                       bz*self.wd_size:(bz+1)*self.wd_size,], axis=0)

        if c_num*K > self.max_wd_num:
            selected_ids = range(self.max_wd_num)
            windows = windows[selected_ids,:,:,:]
            loc_mask = loc_mask[selected_ids,:,:,:]
            centers = centers[selected_ids,:]

        elif c_num*K < self.max_wd_num:
            #pad_windows = np.zeros((self.max_wd_num-c_num*self.K, vx,vy,vz))
            pad_windows = np.zeros((self.max_wd_num-c_num*K, self.wd_size*self.loc_size,self.wd_size*self.loc_size,\
                    self.wd_size*self.loc_size))
            windows = np.concatenate((windows, pad_windows), axis=0)
            pad_loc_mask =  np.zeros((self.max_wd_num-c_num*K, self.loc_size, self.loc_size, self.loc_size))
            loc_mask = np.concatenate((loc_mask, pad_loc_mask), axis=0)
            pad_centers = np.zeros((self.max_wd_num-c_num*K, 3))
            centers = np.concatenate((centers, pad_centers), axis=0)
        if starts is None:
            return windows, mask, loc_mask, centers, or_window,init_blended_window, loc_starts, loc_ends
        else:
            return windows, mask, loc_mask, centers, or_window,init_blended_window, c_num

    def compute_cd_for_vox(self, pc, vox, pos, gt_pt_mean=None):
        xmin, xmax, ymin, ymax, zmin, zmax = pos
        vsize = max(ymax - ymin-1, xmax - xmin-1, zmax-zmin-1)
        #vsize = ymax - ymin-1
        y_range = pc.max(0)[1] - pc.min(0)[1]
        #y_range=1.0
        voxel = y_range/float(vsize)

        vol = np.zeros((128,128,128))
        vol[:vox.shape[0]-16, :vox.shape[1]-16, :vox.shape[2]-16] = vox[8:-8, 8:-8, 8:-8]

        pt_mask = (vol>self.sampling_threshold).astype(np.int32)

        points = self.shape_coord[pt_mask.astype(np.bool)].astype(np.float32)
        points = points*voxel
        # points = points - points.mean(0)
        # pc = pc - pc.mean(0)
        if gt_pt_mean is None:
            pt_mean = (points.max(0)+points.min(0))/2
        else:
            pt_mean = gt_pt_mean

        points = points - pt_mean
        pc = pc - (pc.max(0)+pc.min(0))/2
        # points = points - points.min(0)
        # pc = pc - pc.min(0)
        # points = pc.max(0)*points/points.max(0)

        pc_pred = torch.from_numpy(points).float().to(self.device).unsqueeze(0)
        pc_gt = torch.from_numpy(pc).float().to(self.device).unsqueeze(0)

        dist1, dist2, idx1, idx2 = chamLoss(pc_pred, pc_gt)
        cdis = dist1.mean() + dist2.mean()
        pt_num = (min(pc_pred.shape[1], pc_gt.shape[1])//1024)*1024
        pt_ids = np.random.randint(0, pc_pred.shape[1], size=pt_num)
        pc_pred = pc_pred[:, pt_ids, :]
        pt_ids2 = np.random.randint(0, pc_gt.shape[1], size=pt_num)
        pc_gt = pc_gt[:, pt_ids2, :]

        return cdis.item(), points, pc, pt_mean


    def load(self):
        #load previous checkpoint
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            print('loading from', model_dir)
            fin.close()
            checkpoint = torch.load(model_dir)
            self.deformer.load_state_dict(checkpoint['deformer'])
            self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
            if 'scheduler_g' in checkpoint.keys():
                self.scheduler_g.load_state_dict(checkpoint['scheduler_g'])
            # self.discriminator.load_state_dict(checkpoint['discriminator'])
            # self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
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


        torch.save({
                'deformer': self.deformer.state_dict(),
                # 'discriminator': self.discriminator.state_dict(),
                'optimizer_g': self.optimizer_g.state_dict(),
                # 'optimizer_d': self.optimizer_d.state_dict(),
                'scheduler_g': self.scheduler_g.state_dict(),
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

    @property
    def model_dir(self):
        return "{}_retriv".format(self.data_content)

    def train(self, config):

        if config.continue_train or self.mode=='test':
            self.load()

        if self.mode=='test':
            self.eval_one_epoch(self.start_epoch,config)
            return

        iter_counter = 0
        start_time = time.time()
        training_epoch = config.epoch
        self.dataset_len = len(self.input_content)
        batch_index_list = np.arange(self.dataset_len)

        for epoch in range(self.start_epoch,  training_epoch):
            if self.mode!='test':
                np.random.shuffle(batch_index_list)

            self.deformer.train()
            # self.discriminator.train()
            total_loss_dreal = 0.0
            total_loss_dfake = 0.0
            total_loss_r = 0.0
            total_loss_s = 0.0
            total_loss_g = 0.0

            self.imgout_0 = np.full([self.real_size*4, self.real_size*4*5], 255, np.uint8)
            for idx in range(self.dataset_len):
                # ready a fake image
                dxb = batch_index_list[idx]

                input_in = torch.from_numpy(self.input_content[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                gt_train = torch.from_numpy(self.gt_content[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()

                partial_shape = gaussian_filter(self.partial_content[dxb].astype(np.float32), sigma=1)
                partial_in = torch.from_numpy(partial_shape).to(self.device).unsqueeze(0).unsqueeze(0).float()

                re_pairs = self.re_pairs[dxb]
                pred_dis = self.pred_dis[dxb]
                pred_trans = self.pred_trans[dxb]
                init_trans = self.init_trans[dxb]
                refine_trans = self.refine_trans[dxb]

                vx, vy, vz = self.gt_content[dxb].shape
                cell_size = self.wd_size*self.loc_size

                pad_x = max(0, cell_size-self.gt_content[dxb].shape[0])
                pad_y = max(0, cell_size-self.gt_content[dxb].shape[1])
                pad_z = max(0, cell_size-self.gt_content[dxb].shape[2])

                nsx = (vx+pad_x)//cell_size+1
                nsy = (vy+pad_y)//cell_size+1
                nsz = (vz+pad_z)//cell_size+1
                
                out_shape = np.zeros((vx+pad_x, vy+pad_y, vz+pad_z))
                or_shape = np.zeros((vx+pad_x, vy+pad_y, vz+pad_z))
                init_shape = np.zeros((vx+pad_x, vy+pad_y, vz+pad_z))

                vx1 = vx+ pad_x
                vy1 = vy + pad_y
                vz1 = vz + pad_z
                total_rloss_i = 0.0
                total_sloss_i= 0.0
                total_cnt = 0
                for loc_i in range(nsx):
                    for loc_j in range(nsy):
                        for loc_k in range(nsz):
                            loc_starts = np.array([loc_i*cell_size, loc_j*cell_size, loc_k*cell_size], np.int32)
                            loc_ends = np.array([(loc_i+1)*cell_size, (loc_j+1)*cell_size, (loc_k+1)*cell_size], np.int32)
                            if loc_ends[0]>vx1:
                                loc_ends[0] = (vx1//self.wd_size) * self.wd_size
                                loc_starts[0] = loc_ends[0]-cell_size
                            if loc_ends[1]>vy1:
                                loc_ends[1] = (vy1//self.wd_size) * self.wd_size
                                loc_starts[1] = loc_ends[1]-cell_size
                            if loc_ends[2]>vz1:
                                loc_ends[2] = (vz1//self.wd_size) * self.wd_size
                                loc_starts[2] = loc_ends[2]-cell_size

                            input_windows, window_mask,loc_mask, window_centers,or_window, init_blended_window, c_num\
                               = self.get_overlapping_windows_rand_new(self.partial_content[dxb], re_pairs, init_trans, \
                               refine_trans, pred_dis, starts=loc_starts, ends=loc_ends)

                            loc_mask_in = torch.from_numpy(loc_mask).to(self.device).unsqueeze(0).float()
                            mask_big = F.interpolate(loc_mask_in, scale_factor=self.wd_size, mode='nearest').sum(1)[0]

                            or_window = or_window*(mask_big>0.4).float().detach().cpu().numpy()
                            or_shape[loc_starts[0]:loc_ends[0], loc_starts[1]:loc_ends[1],loc_starts[2]:loc_ends[2],]=or_window
                            init_shape[loc_starts[0]:loc_ends[0], loc_starts[1]:loc_ends[1],loc_starts[2]:loc_ends[2],]=init_blended_window


                            if c_num<20:
                                out_shape[loc_starts[0]:loc_ends[0], loc_starts[1]:loc_ends[1],loc_starts[2]:loc_ends[2],]=or_window
                                continue

                            total_cnt+=1

                            windows_in = torch.from_numpy(input_windows).to(self.device).unsqueeze(0).float()
                            window_mask_in = torch.from_numpy(window_mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
                            window_centers_in = torch.from_numpy(window_centers).to(self.device).float()
                            loc_mask_in = torch.from_numpy(loc_mask).to(self.device).unsqueeze(0).float()
                            # gt_train = self.reshape_to_size_torch(gt_train, (128,128,128))
                            # Dmask_in = self.reshape_to_size_torch(Dmask_in, (56,56,56))
                            partial_in = self.reshape_to_size_torch(partial_in, (128,128,128))
                            input_in = self.reshape_to_size_torch(input_in, (128,128,128))
                            window_mask_in = self.reshape_to_size_torch(window_mask_in, (128,128,128))

                            #self.deformer.zero_grad()
                            self.optimizer_g.zero_grad()

                            # D:[N, 3], X: [N, 3,3,3]
                            D_pred, X_pred = self.deformer(partial_in, input_in, windows_in, window_mask_in, loc_mask_in, is_training=True)

                            voxel_out = self.deform_windows(windows_in,  window_centers_in, D_pred, loc_starts, loc_ends)[:,0] # cropped
                            X_pred_big = F.interpolate(X_pred, scale_factor=self.wd_size, mode='nearest')

                            blended_out = torch.sum(voxel_out*X_pred_big[0], dim=0)
                            gt_wd = gt_train[0,0,loc_starts[0]:loc_ends[0], loc_starts[1]:loc_ends[1], loc_starts[2]:loc_ends[2]]
                            gt_wd = F.pad(gt_wd, (0, max(0, loc_ends[2]-vz), 0, max(0, loc_ends[1]-vy), 0, max(0, loc_ends[0]-vx)))
                            loss_recons = torch.sum(torch.abs(gt_wd-blended_out))/(torch.sum(gt_wd) + torch.sum(blended_out)+1e-5)

                            loss_smooth = self.get_smooth_loss(voxel_out, X_pred[0], loc_mask, loc_starts, loc_ends)
                            loss = loss_recons + self.w_s*loss_smooth
                            #loss = loss_recons
                            loss.backward()
                            self.optimizer_g.step()

                            out_shape[loc_starts[0]:loc_ends[0], loc_starts[1]:loc_ends[1],loc_starts[2]:loc_ends[2],]= \
                                blended_out.detach().cpu().numpy()

                            total_rloss_i += loss_recons.item()
                            total_sloss_i += loss_smooth.item()
                out_shape = out_shape[:vx, :vy, :vz]
                or_shape = or_shape[:vx, :vy, :vz]
                init_shape = init_shape[:vx, :vy, :vz]

                if epoch%self.img_epoch==0:
                    img_y = dxb//4
                    img_x = (dxb%4)*5

                    if img_y<4:
                        tmp_voxel_fake = partial_shape
                        xmin,xmax,ymin,ymax,zmin,zmax = self.pos_content[dxb]
                        tmpvox = self.recover_voxel(tmp_voxel_fake,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

                    img_y = dxb//4
                    img_x = (dxb%4)*5+4
                    if img_y<4:
                        # tmp_mask_exact = self.get_voxel_mask_exact(gt_train.cpu().numpy()[0,0])
                        tmp_mask_exact= self.gt_content[dxb]
                        xmin,xmax,ymin,ymax,zmin,zmax = self.pos_content[dxb]
                        tmpvox = self.recover_voxel(tmp_mask_exact,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)


                    img_x = (dxb%4)*5+3
                    if img_y<4:
                        tmp_out = out_shape
                        tmpvox = self.recover_voxel(tmp_out,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

                    img_x = (dxb%4)*5+2
                    if img_y<4:
                        tmp_out = or_shape
                        tmpvox = self.recover_voxel(tmp_out,xmin,xmax,ymin,ymax,zmin,zmax)
                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)


                    img_x = (dxb%4)*5+1
                    if img_y<4:
                        tmp_out = self.input_content[dxb]
                        tmpvox = self.recover_voxel(tmp_out,xmin,xmax,ymin,ymax,zmin,zmax)

                        self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

            if epoch%self.img_epoch==0:
                cv2.imwrite(self.sample_dir+"/train_"+str(epoch)+"_0.png", self.imgout_0)
            self.log_string("Epoch: [%d/%d] time: %.0f, loss_r: %.5f, loss_s: %.5f" % \
                        (epoch, training_epoch, time.time() - start_time, loss_recons.item(),loss_smooth.item()))

            if epoch%self.eval_epoch==0:
                self.eval_one_epoch(epoch,config)
            if epoch%self.save_epoch==0:
                self.save(epoch)
        self.save(epoch)


    def eval_one_epoch(self, epoch, config):

        start_time = time.time()
        total_loss = 0.0
        total_iou = 0.0
        total_cdis = 0.0
        total_cdis_c = 0.0
        total_num=0
        self.imgout_0 = np.full([self.real_size*4, self.real_size*4*5], 255, np.uint8)

        #self.imgout_0 = np.full([self.real_size, self.real_size*5], 255, np.uint8)
        self.deformer.eval()
        if self.test_dataset_len==0:
            return
        total_feat_fake = np.zeros((self.test_dataset_len, 4096))
        total_feat_real = np.zeros((self.test_dataset_len, 4096))

        for idx in range(self.test_dataset_len):

            tt0 = time.time()
            # ready a fake image
            dxb = idx
            input_in = torch.from_numpy(self.input_test[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
            # mask_in =  torch.from_numpy(self.mask_test[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
            gt_test = torch.from_numpy(self.gt_test[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
            partial_shape = gaussian_filter(self.partial_test[dxb].astype(np.float32), sigma=1)
            partial_in = torch.from_numpy(partial_shape).to(self.device).unsqueeze(0).unsqueeze(0).float()

            partial_in = self.reshape_to_size_torch(partial_in, (128,128,128))
            input_in = self.reshape_to_size_torch(input_in, (128,128,128))

            re_pairs = self.re_pairs_test[dxb]
            pred_dis = self.pred_dis_test[dxb]
            pred_trans = self.pred_trans_test[dxb]
            init_trans = self.init_trans_test[dxb]
            re_trans = self.re_trans_test[dxb]
            refine_trans = self.refine_trans_test[dxb]

            vx, vy, vz = self.gt_test[dxb].shape
            cell_size = self.wd_size*self.loc_size
            cell_stride = cell_size

            overlap = 8
            cell_stride = cell_size-overlap

            pad_x = max(0, cell_size-self.gt_test[dxb].shape[0])
            pad_y = max(0, cell_size-self.gt_test[dxb].shape[1])
            pad_z = max(0, cell_size-self.gt_test[dxb].shape[2])

            vx1 = vx + pad_x
            vy1 = vy + pad_y
            vz1 = vz + pad_z
            nsx = vx1//cell_stride+2
            nsy = vy1//cell_stride+2
            nsz = vz1//cell_stride+2

            out_shape = np.zeros((vx+pad_x, vy+pad_y, vz+pad_z))
            or_shape = np.zeros((vx+pad_x, vy+pad_y, vz+pad_z))

            total_rloss_i = 0.0
            total_sloss_i= 0.0
            total_cnt = 0


            for loc_i in range(nsx):
                for loc_j in range(nsy):
                    for loc_k in range(nsz):
                        loc_starts = np.array([loc_i*cell_stride, loc_j*cell_stride, loc_k*cell_stride], np.int32)
                        loc_ends = loc_starts + cell_size

                        if loc_ends[0]>vx:
                            loc_ends[0] = vx1//self.wd_size * self.wd_size
                            loc_starts[0] = loc_ends[0]-cell_size

                        if loc_ends[1]>vy:
                            loc_ends[1] = vy1//self.wd_size * self.wd_size
                            loc_starts[1] = loc_ends[1]-cell_size
                        if loc_ends[2]>vz:
                            loc_ends[2] = vz1//self.wd_size * self.wd_size
                            loc_starts[2] = loc_ends[2]-cell_size


                        input_windows, window_mask,loc_mask, window_centers,or_window, init_blended_window, c_num\
                           = self.get_overlapping_windows_rand_new(self.partial_test[dxb], re_pairs, init_trans, \
                           refine_trans, pred_dis, starts=loc_starts, ends=loc_ends)

                        or_shape[loc_starts[0]:loc_ends[0], loc_starts[1]:loc_ends[1],loc_starts[2]:loc_ends[2],]=or_window

                        if c_num<20:
                            #print(c_num)
                            out_shape[loc_starts[0]:loc_ends[0], loc_starts[1]:loc_ends[1],loc_starts[2]:loc_ends[2],]= or_window
                            continue

                        total_cnt+=1

                        windows_in = torch.from_numpy(input_windows).to(self.device).unsqueeze(0).float()
                        window_mask_in = torch.from_numpy(window_mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
                        window_centers_in = torch.from_numpy(window_centers).to(self.device).float()
                        loc_mask_in = torch.from_numpy(loc_mask).to(self.device).unsqueeze(0).float()
                        window_mask_in = self.reshape_to_size_torch(window_mask_in, (128,128,128))

                        with torch.no_grad():
                            # D:[N, 3], X: [N, 3,3,3]
                            D_pred, X_pred = self.deformer(partial_in,input_in, windows_in, window_mask_in, loc_mask_in, is_training=False)
                            # print('pred', D_pred.min(), D_pred.max(), X_pred.min(), X_pred.max())
                            voxel_out = self.deform_windows(windows_in,  window_centers_in, D_pred, loc_starts, loc_ends)[:,0] # cropped
                            X_pred_big = F.interpolate(X_pred, scale_factor=self.wd_size, mode='nearest')
                            blended_out = torch.sum(voxel_out*X_pred_big[0], dim=0)

                            gt_wd = gt_test[0,0,loc_starts[0]:loc_ends[0], loc_starts[1]:loc_ends[1], loc_starts[2]:loc_ends[2]]
                            gt_wd = F.pad(gt_wd, (0, max(0, loc_ends[2]-vz), 0, max(0, loc_ends[1]-vy), 0, max(0, loc_ends[0]-vx)))
                            #loss_recons = torch.sum((gt_wd-blended_out)**2)/(torch.sum(gt_wd**2)+torch.sum(blended_out)+1e-5)
                            loss_recons = torch.sum(torch.abs(gt_wd-blended_out))/(torch.sum(gt_wd) + torch.sum(blended_out)+1e-5)
                            # loss_recons = self.get_recons_loss(blended_out, gt_train, loc_starts, loc_ends)
                            loss_smooth = self.get_smooth_loss(voxel_out, X_pred[0], loc_mask, loc_starts, loc_ends)

                            out_shape[loc_starts[0]:loc_ends[0], loc_starts[1]:loc_ends[1],loc_starts[2]:loc_ends[2],]= \
                                blended_out.detach().cpu().numpy()

                            total_rloss_i += loss_recons.item()
                            total_sloss_i += loss_smooth.item()

            out_shape = out_shape[:vx, :vy, :vz]
            or_shape = or_shape[:vx, :vy, :vz]

            if self.compute_cd:
                cdis_c, coa_pc, gt_pc_c, gt_pt_mean = self.compute_cd_for_vox(self.gt_pc[dxb], self.gt_test[dxb], self.pos_test[dxb],)
                cdis, pred_pc, gt_pc, _ = self.compute_cd_for_vox(self.gt_pc[dxb], out_shape, self.pos_test[dxb], )
                
                if cdis_c<=0.0002:
                    total_cdis+= cdis
                    total_cdis_c += cdis_c
                    total_num+=1

                print(dxb, self.shape_names_test[dxb], 'cdis', cdis,total_cdis/(total_num+1e-5),'cdis_gt', cdis_c, \
                        total_cdis_c/(total_num+1e-5),)

                m_vox = -gaussian_filter((out_shape>0.35).astype(np.float32), sigma=0.8)  # 0.8, 0.2, 0.35 # 0.7, 0.25
                verts, faces, normals, values = measure.marching_cubes(m_vox, -0.2)

                v = verts-verts.min(0)
                v = v/v.max()
                v = v-(v.max(0)+v.min(0))/2

                mesh = Mesh(v=verts, f=faces)
                mesh.write_ply(os.path.join(self.sample_dir, 'eval_%s_%d_%d_pred.ply'%(self.pre_fix, epoch,dxb)))

                m_vox =-gaussian_filter(self.partial_test[dxb].astype(np.float32), sigma=0.8)
                #verts, faces, normals, values = measure.marching_cubes(m_vox, -self.sampling_threshold)
                verts, faces, normals, values = measure.marching_cubes(m_vox, -0.2)
                mesh = Mesh(v=verts, f=faces)
                mesh.write_ply(os.path.join(self.sample_dir, 'eval_%s_%d_%d_input.ply'%(self.pre_fix, epoch,dxb)))

                m_vox =-gaussian_filter(self.gt_test[dxb].astype(np.float32), sigma=0.8)
                #verts, faces, normals, values = measure.marching_cubes(m_vox, -self.sampling_threshold)
                verts, faces, normals, values = measure.marching_cubes(m_vox, -0.2)
                mesh = Mesh(v=verts, f=faces)
                mesh.write_ply(os.path.join(self.sample_dir, 'eval_%s_%d_%d_gt..ply'%(self.pre_fix, epoch,dxb)))


            imgout =  np.full([self.real_size, self.real_size*5], 255, np.uint8)
            tmp_voxel_fake = partial_shape
            tmpvox = self.reshape_to_patch_size(partial_shape, np.zeros((256,256,256)))
            imgout[:, 0:256]=self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

            tmp_voxel_fake = input_in.cpu().numpy()[0,0]
            tmpvox = np.zeros((256, 256,256))
            tmpvox[64:-64, 64:-64, 64:-64] = tmp_voxel_fake
            imgout[:, 256:2*256] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

            tmp_mask_exact= self.gt_test[dxb]
            xmin,xmax,ymin,ymax,zmin,zmax = self.pos_test[dxb]
            tmpvox = self.recover_voxel(tmp_mask_exact,xmin,xmax,ymin,ymax,zmin,zmax)
            imgout[:, 4*256:5*256] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

            tmp_out = or_shape
            tmpvox = self.recover_voxel(tmp_out,xmin,xmax,ymin,ymax,zmin,zmax)
            imgout[:,2*256:3*256] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

            tmp_out = out_shape
            tmpvox = self.recover_voxel(tmp_out,xmin,xmax,ymin,ymax,zmin,zmax)
            imgout[:,3*256:4*256]  = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)

            cv2.imwrite(self.sample_dir+"/eval_%s_"%(self.pre_fix)+str(epoch)+"_%d_.png"%(dxb), imgout)

        self.log_string("[eval] Epoch: [%d/%d] time: %.0f, eval_loss_r: %.6f, eval_loss_s: %.5f," % (epoch, config.epoch, time.time() - start_time,  total_rloss_i/total_cnt, total_sloss_i/total_cnt,))


    def deform_windows(self, windows_in, window_centers_in, D_pred,loc_starts, loc_ends):
        batch_size = windows_in.shape[1]

        # crop_wds = windows_in[0,:,loc_starts[0]:loc_ends[0], loc_starts[1]:loc_ends[1], loc_starts[2]:loc_ends[2]]
        crop_wds = windows_in[0]
        # print("D_pred", D_pred.shape, D_pred.min(), D_pred.max())
        angle = torch.Tensor([0, 0, 0]).float().view(1,3).cuda().repeat(batch_size, 1)

        scale = torch.Tensor([1.0]).view(1,1).cuda().repeat(batch_size, 1)
        aff_matrix = get_affine_matrix3d(D_pred[:,0:3]*self.trans_limit, window_centers_in, scale, angle)
        crop_wds_out = warp_affine3d(crop_wds.unsqueeze(1), aff_matrix[:,:3,:], (loc_ends[0]-loc_starts[0], loc_ends[1]-loc_starts[1],\
            loc_ends[2]-loc_starts[2]))
        return crop_wds_out

    def get_recons_loss(self, blended_out, gt_train,loc_starts, loc_ends):
        gt_wd = gt_train[0,0,loc_starts[0]:loc_ends[0], loc_starts[1]:loc_ends[1], loc_starts[2]:loc_ends[2]]
        loss_r = torch.sum((gt_wd-blended_out)**2)/(torch.sum(gt_wd**2)+1e-5)
        return loss_r


    def get_smooth_loss(self, voxel_out, X_pred_big, loc_mask, loc_starts, loc_ends):
        # voxel out: [M, 24,24,24], X_pred_big: [M, 24,24,24]
        # loc_mask: [M, 3,3,3]

        # x direction
        total_pairwise_e = torch.Tensor([0.0]).to(self.device)
        for ix in range(self.loc_size-1):
            for iy in range(self.loc_size-1):
                for iz in range(self.loc_size-1):
                    # x-left

                    mask_i = loc_mask[:, ix, iy, iz]*loc_mask[:, ix+1, iy,iz]
                    ids = np.where(mask_i==1)[0]
                    if len(ids)>0:
                        X1 = X_pred_big[ids][:, ix, iy, iz].view(len(ids), 1)
                        X2 = X_pred_big[ids][:, ix+1, iy, iz].view(1, len(ids))
                        #print("X1, X2",X1, X2,)
                        X_cross = torch.matmul(X1, X2)
                        voxel_i = voxel_out[ids][:, ix*self.wd_size+self.wd_size//2:(ix+1)*self.wd_size+self.wd_size//2,\
                            iy*self.wd_size:(iy+1)*self.wd_size, iz*self.wd_size:(iz+1)*self.wd_size]
                        voxel_cross1 = voxel_i.view(len(ids), 1,self.wd_size, self.wd_size, \
                            self.wd_size).repeat(1, len(ids), 1,1,1)
                        voxel_cross2 = voxel_i.view(1, len(ids),self.wd_size, self.wd_size, \
                            self.wd_size).repeat(len(ids),1,1,1,1)
                        voxel_cross = torch.sum(torch.abs(voxel_cross2-voxel_cross1),dim=(2,3,4))/(torch.sum(voxel_cross2,\
                            dim=(2,3,4))+torch.sum(voxel_cross1, dim=(2,3,4))+1e-5)
                        total_pairwise_e += torch.mean(voxel_cross*X_cross)


                    mask_i = loc_mask[:, ix, iy, iz]*loc_mask[:, ix, iy+1,iz]
                    ids = np.where(mask_i==1)[0]
                    if len(ids)>0:
                        X1 = X_pred_big[ids][:, ix, iy, iz].view(len(ids), 1)
                        X2 = X_pred_big[ids][:, ix, iy+1, iz].view(1, len(ids))
                        X_cross = torch.matmul(X1, X2)
                        voxel_i = voxel_out[ids][:, ix*self.wd_size:(ix+1)*self.wd_size,\
                            iy*self.wd_size+self.wd_size//2:(iy+1)*self.wd_size+self.wd_size//2, \
                                iz*self.wd_size:(iz+1)*self.wd_size]
                        voxel_cross1 = voxel_i.view(len(ids), 1,self.wd_size, self.wd_size, \
                            self.wd_size).repeat(1, len(ids), 1,1,1)
                        voxel_cross2 = voxel_i.view(1, len(ids),self.wd_size, self.wd_size, \
                            self.wd_size).repeat(len(ids),1,1,1,1)
                        voxel_cross = torch.sum(torch.abs(voxel_cross2-voxel_cross1),dim=(2,3,4))/(torch.sum(voxel_cross2,\
                            dim=(2,3,4))+torch.sum(voxel_cross1, dim=(2,3,4))+1e-5)
                        total_pairwise_e += torch.mean(voxel_cross*X_cross)

                    mask_i = loc_mask[:, ix, iy, iz]*loc_mask[:, ix, iy,iz+1]
                    ids = np.where(mask_i==1)[0]
                    if len(ids)>0:
                        X1 = X_pred_big[ids][:, ix, iy, iz].view(len(ids), 1)
                        X2 = X_pred_big[ids][:, ix, iy, iz+1].view(1, len(ids))
                        X_cross = torch.matmul(X1, X2)
                        voxel_i = voxel_out[ids][:, ix*self.wd_size:(ix+1)*self.wd_size,iy*self.wd_size:(iy+\
                            1)*self.wd_size, iz*self.wd_size+self.wd_size//2:(iz+1)*self.wd_size+self.wd_size//2]
                        voxel_cross1 = voxel_i.view(len(ids), 1,self.wd_size, self.wd_size, \
                            self.wd_size).repeat(1, len(ids), 1,1,1)
                        voxel_cross2 = voxel_i.view(1, len(ids),self.wd_size, self.wd_size, \
                            self.wd_size).repeat(len(ids),1,1,1,1)
                        voxel_cross = torch.sum(torch.abs(voxel_cross2-voxel_cross1),dim=(2,3,4))/(torch.sum(voxel_cross2,\
                            dim=(2,3,4))+torch.sum(voxel_cross1, dim=(2,3,4))+1e-5)
                        total_pairwise_e += torch.mean(voxel_cross*X_cross)
        return total_pairwise_e


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

