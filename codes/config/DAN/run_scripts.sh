# single GPU training
#python train.py -opt options/train/train_SRResNet.yml

# distributed training
# 4 GPUs
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 train_EM.py -opt options/train/train_EM.yml --launcher pytorch

