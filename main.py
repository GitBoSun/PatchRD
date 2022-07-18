import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()

# training options
parser.add_argument("--train_complete", action="store_true", dest="train_complete", default=False, help="True for training the coarse completor")
parser.add_argument("--train_patch", action="store_true", dest="train_patch", default=False, help="True for training feature embedding for retrieval")
parser.add_argument("--train_deform", action="store_true", dest="train_deform", default=False, help="True for training initial deformation")
parser.add_argument("--train_joint", action="store_true", dest="train_joint", default=False, help="True for jointly training deformation and blending")
parser.add_argument("--mode", action="store", dest="mode", default="train", help="mode: train or test")
parser.add_argument("--gpu", action="store", dest="gpu", default="0", help="to use which GPU")

# training parameters 
parser.add_argument("--epoch", action="store", dest="epoch", default=20, type=int, help="Epoch to train")
parser.add_argument("--batch_size", action="store", dest="batch_size", default=1, type=int, help="Batch size [1]")
parser.add_argument("--lr", action="store", dest="lr", default=1e-4, type=float, help="Learning rate")
parser.add_argument("--decay_step", action="store", dest="decay_step", default=1, type=int, help="lr decay step")
parser.add_argument("--lr_decay", action="store", dest="lr_decay", default=0.99, type=float, help="lr decay rate")
parser.add_argument("--continue_train", action="store_true", dest="continue_train", default=False, help="If True, continue training, otherwise train from scratch")

# dump and eval options 
parser.add_argument("--dump_deform", action="store_true", dest="dump_deform", default=False, help="True for dumping intermediate retrieval and  deformation results in mode --train_patch and --train_deform")
parser.add_argument("--compute_cd", action="store_true", dest="compute_cd", default=False, help="True for evaluating chamfer distance and dumping mesh resulst in mode --train_joint")
parser.add_argument("--small_dataset", action="store_true", dest="small_dataset", default=False, help="True for only use four samples in the testing mode")


# model parameters 
parser.add_argument("--g_dim", action="store", dest="g_dim", default=32, type=int, help="Channel dimension for models")
parser.add_argument("--z_dim", action="store", dest="z_dim", default=8, type=int, help="Dimension of the latent feature embedding for retrieval learning")

# data parameters 
parser.add_argument("--input_size", action="store", dest="input_size", default=32, type=int, help="Input voxel size")
parser.add_argument("--output_size", action="store", dest="output_size", default=128, type=int, help="Output voxel size")
parser.add_argument("--patch_size", action="store", dest="patch_size", default=18, type=int, help="Patch size for retrieval and deformation")
parser.add_argument("--csize", action="store", dest="csize", default=26, type=int, help="Average size of cropped area")
parser.add_argument("--c_range", action="store", dest="c_range", default=26, type=int, help="Devation of cropped size")
parser.add_argument("--K", action="store", dest="K", default=2, type=int, help="ratio of randomly sampled patch pairs VS similiar pairs in retrieval learning")
parser.add_argument("--max_sample_num", action="store", dest="max_sample_num", default=200, type=int, help="Maximum number of patches in retrieval learning")


# paths  
parser.add_argument("--data_content", action="store", dest="data_content", help="Data category. See ./splits for all categories")
parser.add_argument("--data_dir", action="store", dest="data_dir", help="Root directory of dataset")
parser.add_argument("--dump_deform_path", action="store", dest="dump_deform_path", help="Directory to dump intermidiate retrieval and initial deformation results.")
parser.add_argument("--model_name", action="store", dest="model_name", default="checkpoint", help="Model name.")
parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint", help="Directory name to save the checkpoints")
parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="./samples/", help="Directory name to save the image samples")
parser.add_argument("--log_dir", action="store", dest="log_dir", default="./logs/", help="Directory name to save the training logs")


# loss weights 
parser.add_argument("--w_mask", action="store", dest="w_mask", default=1, type=float, help="[coarse completion]: weight for the occupied area in the reconstruction loss")
parser.add_argument("--w_posi", action="store", dest="w_posi", default=1, type=float, help="[coarse completion]: weight for the occupied area in the cross entropy loss")
parser.add_argument("--w_ident", action="store", dest="w_ident", default=0, type=float, help="[retrieval learning]: weight for similar pairs")
parser.add_argument("--w_dis", action="store", dest="w_dis", default=1, type=float, help="[initial deformation]: weight for the patch distance prediction")
parser.add_argument("--w_r", action="store", dest="w_r", default=1, type=float, help="[deformation and blending]: weight for reconstruction term")
parser.add_argument("--w_s", action="store", dest="w_s", default=0, type=float, help="[deformation and blending]: weight for smoothness term")

# parameters for training deformation and blending (--train_joint) 
parser.add_argument("--max_wd_num", action="store", dest="max_wd_num", default=32, type=int, help="Maximum number of windows")
parser.add_argument("--sample_step", action="store", dest="sample_step", default=32, type=int, help="Sample stride of retrieved patches")
parser.add_argument("--loc_size", action="store", dest="loc_size", default=32, type=int, help="Number of windows in one subvolume")
parser.add_argument("--wd_size", action="store", dest="wd_size", default=8, type=int, help="Window size")
parser.add_argument("--trans_limit", action="store", dest="trans_limit", default=1e-4, type=float, help="Translation limit upon the initial deformation")
parser.add_argument("--gw_dim", action="store", dest="gw_dim", default=32, type=int, help="Channel dimension for the window encoding branch")

FLAGS = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu


if FLAGS.train_patch:
    from train_patch import MODEL_PATCH
    model = MODEL_PATCH(FLAGS)
    model.train(FLAGS)

elif FLAGS.train_complete:
    from train_complete import MODEL_COMPLETE
    model = MODEL_COMPLETE(FLAGS)
    model.train(FLAGS)

elif FLAGS.train_deform:
    print('***********train deform')
    from train_deform import MODEL_DEFORM
    model = MODEL_DEFORM(FLAGS)
    model.train(FLAGS)

elif FLAGS.train_joint:
    print('***********train joint')
    from train_joint import MODEL_JOINT
    model = MODEL_JOINT(FLAGS)
    model.train(FLAGS)
else:
    print('Invalid training options!')
