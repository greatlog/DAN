# distributed training
# 2 GPUs
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 train_IKC.py --launcher pytorch
