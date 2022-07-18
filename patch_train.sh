python main.py \
--data_content content_chair \
--data_dir ./data/03001627/ \
--input_size 32 \
--output_size 128 \
--sample_dir samples_patch \
--checkpoint_dir checkpoint_patch \
--gpu 0 \
--epoch 200 \
--lr 1e-4 \
--z_dim 128 \
--g_dim 32 \
--K 2 \
--w_ident 8 \
--csize 26 \
--c_range 26  \
--max_sample_num 400 \
--train_patch \
--model_name retrieval \
--dump_deform_path ./dump_deform/chair \
#--dump_deform \
#--small_dataset \
#--continue_train \