# single GPU training
#python train.py -opt train_option.yml

# distributed training
# 8 GPUs
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 train.py -opt train_option.yml --launcher pytorch

