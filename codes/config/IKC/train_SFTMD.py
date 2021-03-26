import argparse
import logging
import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import options as option
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler
from models import create_model
from utils import util


def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    if (
        mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # Initializes the default distributed process group


def main():
    ###### SFTMD train ######
    #### setup options
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt_F",
        type=str,
        default="options/train/SFTMD/train_SFTMD_x4.yml",
        help="Path to option YMAL file of SFTMD_Net.",
    )
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt_F = option.parse(args.opt_F, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt_F = option.dict_to_nonedict(opt_F)

    #### random seed
    seed = opt_F["train"]["manual_seed"]
    if seed is None:
        seed = random.randint(1, 10000)
    util.set_random_seed(seed)

    # load PCA matrix of enough kernel
    print("load PCA matrix")
    pca_matrix = torch.load(
        "../../../pca_matrix/IKC/pca_matrix.pth",
        map_location=lambda storage, loc: storage,
    )
    print("PCA matrix shape: {}".format(pca_matrix.shape))

    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training
        opt_F["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt_F["dist"] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### loading resume state if exists
    if opt_F["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt_F["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt_F, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:
        if resume_state is None:
            util.mkdir_and_rename(
                opt_F["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt_F["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )
            os.system("rm ./log")
            os.symlink(os.path.join(opt_F["path"]["experiments_root"], ".."), "./log")

        # config loggers. Before it, the log will not work
        util.setup_logger(
            "base",
            opt_F["path"]["log"],
            "train_" + opt_F["name"],
            level=logging.INFO,
            screen=True,
            tofile=True,
        )
        util.setup_logger(
            "val",
            opt_F["path"]["log"],
            "val_" + opt_F["name"],
            level=logging.INFO,
            screen=True,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt_F))
        # tensorboard logger
        if opt_F["use_tb_logger"] and "debug" not in opt_F["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt_F["name"]))
    else:
        util.setup_logger(
            "base", opt_F["path"]["log"], "train", level=logging.INFO, screen=True
        )
        logger = logging.getLogger("base")

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt_F["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt_F["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt_F["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(
                train_set, dataset_opt, opt_F, train_sampler
            )
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt_F, None)
            if rank <= 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    assert val_loader is not None

    #### create model
    model_F = create_model(opt_F)

    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model_F.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )
    for epoch in range(start_epoch, total_epochs + 1):
        if opt_F["dist"]:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### preprocessing for LR_img and kernel map
            prepro = util.SRMDPreprocessing(
                opt_F["scale"],
                pca_matrix,
                random=True,
                para_input=opt_F["code_length"],
                kernel=opt_F["kernel_size"],
                noise=False,
                cuda=True,
                sig=opt_F["sig"],
                sig_min=opt_F["sig_min"],
                sig_max=opt_F["sig_max"],
                rate_iso=1.0,
                scaling=3,
                rate_cln=0.2,
                noise_high=0.0,
            )
            LR_img, ker_map = prepro(train_data["GT"])

            #### update learning rate, schedulers
            model_F.update_learning_rate(
                current_step, warmup_iter=opt_F["train"]["warmup_iter"]
            )

            #### training
            model_F.feed_data(train_data, LR_img, ker_map)
            model_F.optimize_parameters(current_step)

            #### log
            if current_step % opt_F["logger"]["print_freq"] == 0:
                logs = model_F.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model_F.get_current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    # tensorboard logger
                    if opt_F["use_tb_logger"] and "debug" not in opt_F["name"]:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            # validation
            if current_step % opt_F["train"]["val_freq"] == 0 and rank <= 0:
                avg_psnr = 0.0
                idx = 0
                for _, val_data in enumerate(val_loader):
                    idx += 1
                    #### preprocessing for LR_img and kernel map
                    prepro = util.SRMDPreprocessing(
                        opt_F["scale"],
                        pca_matrix,
                        random=True,
                        para_input=opt_F["code_length"],
                        kernel=opt_F["kernel_size"],
                        noise=False,
                        cuda=True,
                        sig=opt_F["sig"],
                        sig_min=opt_F["sig_min"],
                        sig_max=opt_F["sig_max"],
                        rate_iso=1.0,
                        scaling=3,
                        rate_cln=0.2,
                        noise_high=0.0,
                    )
                    LR_img, ker_map = prepro(val_data["GT"])

                    model_F.feed_data(val_data, LR_img, ker_map)
                    model_F.test()

                    visuals = model_F.get_current_visuals()
                    sr_img = util.tensor2img(visuals["SR"])  # uint8
                    gt_img = util.tensor2img(visuals["GT"])  # uint8

                    # Save SR images for reference
                    img_name = os.path.splitext(
                        os.path.basename(val_data["LQ_path"][0])
                    )[0]
                    # img_dir = os.path.join(opt_F['path']['val_images'], img_name)
                    img_dir = os.path.join(
                        opt_F["path"]["val_images"], str(current_step)
                    )
                    util.mkdir(img_dir)

                    save_img_path = os.path.join(
                        img_dir, "{:s}_{:d}.png".format(img_name, current_step)
                    )
                    util.save_img(sr_img, save_img_path)

                    # calculate PSNR
                    crop_size = opt_F["scale"]
                    gt_img = gt_img / 255.0
                    sr_img = sr_img / 255.0
                    cropped_sr_img = sr_img[
                        crop_size:-crop_size, crop_size:-crop_size, :
                    ]
                    cropped_gt_img = gt_img[
                        crop_size:-crop_size, crop_size:-crop_size, :
                    ]
                    avg_psnr += util.calculate_psnr(
                        cropped_sr_img * 255, cropped_gt_img * 255
                    )

                avg_psnr = avg_psnr / idx

                # log
                logger.info("# Validation # PSNR: {:.4e}".format(avg_psnr))
                logger_val = logging.getLogger("val")  # validation logger
                logger_val.info(
                    "<epoch:{:3d}, iter:{:8,d}> psnr: {:.6f}".format(
                        epoch, current_step, avg_psnr
                    )
                )
                # tensorboard logger
                if opt_F["use_tb_logger"] and "debug" not in opt_F["name"]:
                    tb_logger.add_scalar("psnr", avg_psnr, current_step)

            #### save models and training states
            if current_step % opt_F["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    logger.info("Saving models and training states.")
                    model_F.save(current_step)
                    model_F.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info("Saving the final model.")
        model_F.save("latest")
        logger.info("End of SFTMD training.")


if __name__ == "__main__":
    main()
