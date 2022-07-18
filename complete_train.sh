python main.py \
--data_content content_chair \
--data_dir ./data/03001627/ \
--input_size 32 \
--output_size 128 \
--sample_dir samples_complete \
--checkpoint_dir checkpoint_complete \
--gpu 0 \
--epoch 101 \
--lr 1e-4 \
--g_dim 32 \
--w_posi 1.5 \
--w_mask 0.5 \
--csize 26 \
--c_range 26 \
--train_complete \
--model_name coarse_comp \
#--mode test \
#--continue_train \
