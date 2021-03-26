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
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt_P",
        type=str,
        default="options/train/Predictor/train_Predictor_x4.yml",
        help="Path to option YMAL file of Predictor.",
    )
    parser.add_argument(
        "-opt_C",
        type=str,
        default="options/train/Corrector/train_Corrector_x4.yml",
        help="Path to option YMAL file of Corrector.",
    )
    parser.add_argument(
        "-opt_F",
        type=str,
        default="options/test/SFTMD/test_SFTMD_x4.yml",
        help="Path to option YMAL file of SFTMD_Net.",
    )
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt_P = option.parse(args.opt_P, is_train=True)
    opt_C = option.parse(args.opt_C, is_train=True)
    opt_F = option.parse(args.opt_F, is_train=False)

    # convert to NoneDict, which returns None for missing keys
    opt_P = option.dict_to_nonedict(opt_P)
    opt_C = option.dict_to_nonedict(opt_C)
    opt_F = option.dict_to_nonedict(opt_F)

    #### set random seed
    seed = opt_P["train"]["manual_seed"]
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
        opt_P["dist"] = False
        opt_C["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt_P["dist"] = True
        opt_C["dist"] = True
        init_dist()
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    if opt_P["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt_P["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt_P, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            util.mkdir_and_rename(
                opt_P["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt_P["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )
            # Corrector path
            util.mkdir_and_rename(
                opt_C["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt_C["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )

            os.system("rm ./log")
            os.symlink(os.path.join(opt_P["path"]["experiments_root"], ".."), "./log")

        # config loggers. Before it, the log will not work
        util.setup_logger(
            "base",
            opt_P["path"]["log"],
            "train_" + opt_P["name"],
            level=logging.INFO,
            screen=True,
            tofile=True,
        )
        util.setup_logger(
            "val",
            opt_P["path"]["log"],
            "val_" + opt_P["name"],
            level=logging.INFO,
            screen=True,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt_P))
        logger.info(option.dict2str(opt_C))
        # tensorboard logger
        if opt_P["use_tb_logger"] and "debug" not in opt_P["name"]:
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
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt_P["name"]))
    else:
        util.setup_logger(
            "base", opt_P["path"]["log"], "train", level=logging.INFO, screen=True
        )
        logger = logging.getLogger("base")

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt_P["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt_P["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt_P["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(
                train_set, dataset_opt, opt_P, train_sampler
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
            val_loader = create_dataloader(val_set, dataset_opt, opt_P, None)
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
    model_F = create_model(opt_F)  # load pretrained model of SFTMD
    model_P = create_model(opt_P)
    model_C = create_model(opt_C)

    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model_P.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )
    for epoch in range(start_epoch, total_epochs + 1):
        if opt_P["dist"]:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### update learning rate, schedulers
            # model.update_learning_rate(current_step, warmup_iter=opt_P['train']['warmup_iter'])

            #### preprocessing for LR_img and kernel map
            prepro = util.SRMDPreprocessing(
                opt_P["scale"],
                pca_matrix,
                random=True,
                para_input=opt_P["code_length"],
                kernel=opt_P["kernel_size"],
                noise=False,
                cuda=True,
                sig=opt_P["sig"],
                sig_min=opt_P["sig_min"],
                sig_max=opt_P["sig_max"],
                rate_iso=1.0,
                scaling=3,
                rate_cln=0.2,
                noise_high=0.0,
            )
            LR_img, ker_map = prepro(train_data["GT"])

            #### training Predictor
            model_P.feed_data(LR_img, ker_map)
            model_P.optimize_parameters(current_step)
            P_visuals = model_P.get_current_visuals()
            est_ker_map = P_visuals["Batch_est_ker_map"]

            #### log of model_P
            if current_step % opt_P["logger"]["print_freq"] == 0:
                logs = model_P.get_current_log()
                message = "Predictor <epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model_P.get_current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    # tensorboard logger
                    if opt_P["use_tb_logger"] and "debug" not in opt_P["name"]:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            #### training Corrector
            for step in range(opt_C["step"]):
                # test SFTMD for corresponding SR image
                model_F.feed_data(train_data, LR_img, est_ker_map)
                model_F.test()
                F_visuals = model_F.get_current_visuals()
                SR_img = F_visuals["Batch_SR"]
                # Test SFTMD to produce SR images

                # train corrector given SR image and estimated kernel map
                model_C.feed_data(SR_img, est_ker_map, ker_map)
                model_C.optimize_parameters(current_step)
                C_visuals = model_C.get_current_visuals()
                est_ker_map = C_visuals["Batch_est_ker_map"]

                #### log of model_C
                if current_step % opt_C["logger"]["print_freq"] == 0:
                    logs = model_C.get_current_log()
                    message = "Corrector <epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                        epoch, current_step, model_C.get_current_learning_rate()
                    )
                    for k, v in logs.items():
                        message += "{:s}: {:.4e} ".format(k, v)
                        # tensorboard logger
                        if opt_C["use_tb_logger"] and "debug" not in opt_C["name"]:
                            if rank <= 0:
                                tb_logger.add_scalar(k, v, current_step)
                    if rank <= 0:
                        logger.info(message)

            # validation, to produce ker_map_list(fake)
            if current_step % opt_P["train"]["val_freq"] == 0 and rank <= 0:
                avg_psnr = 0.0
                idx = 0
                for _, val_data in enumerate(val_loader):
                    prepro = util.SRMDPreprocessing(
                        opt_P["scale"],
                        pca_matrix,
                        random=True,
                        para_input=opt_P["code_length"],
                        kernel=opt_P["kernel_size"],
                        noise=False,
                        cuda=True,
                        sig=opt_P["sig"],
                        sig_min=opt_P["sig_min"],
                        sig_max=opt_P["sig_max"],
                        rate_iso=1.0,
                        scaling=3,
                        rate_cln=0.2,
                        noise_high=0.0,
                    )
                    LR_img, ker_map = prepro(val_data["GT"])
                    single_img_psnr = 0.0
                    lr_img = util.tensor2img(LR_img)  # save LR image for reference

                    # valid Predictor
                    model_P.feed_data(LR_img, ker_map)
                    model_P.test()
                    P_visuals = model_P.get_current_visuals()
                    est_ker_map = P_visuals["Batch_est_ker_map"]

                    # Save images for reference
                    img_name = os.path.splitext(
                        os.path.basename(val_data["LQ_path"][0])
                    )[0]
                    img_dir = os.path.join(opt_P["path"]["val_images"], img_name)
                    # img_dir = os.path.join(opt_F['path']['val_images'], str(current_step), '_', str(step))
                    util.mkdir(img_dir)
                    save_lr_path = os.path.join(img_dir, "{:s}_LR.png".format(img_name))
                    util.save_img(lr_img, save_lr_path)

                    for step in range(opt_C["step"]):
                        step += 1
                        idx += 1
                        model_F.feed_data(val_data, LR_img, est_ker_map)
                        model_F.test()
                        F_visuals = model_F.get_current_visuals()
                        SR_img = F_visuals["Batch_SR"]
                        # Test SFTMD to produce SR images

                        model_C.feed_data(SR_img, est_ker_map, ker_map)
                        model_C.test()
                        C_visuals = model_C.get_current_visuals()
                        est_ker_map = C_visuals["Batch_est_ker_map"]

                        sr_img = util.tensor2img(F_visuals["SR"])  # uint8
                        gt_img = util.tensor2img(F_visuals["GT"])  # uint8

                        save_img_path = os.path.join(
                            img_dir,
                            "{:s}_{:d}_{:d}.png".format(img_name, current_step, step),
                        )
                        util.save_img(sr_img, save_img_path)

                        # calculate PSNR
                        crop_size = opt_P["scale"]
                        gt_img = gt_img / 255.0
                        sr_img = sr_img / 255.0
                        cropped_sr_img = sr_img[
                            crop_size:-crop_size, crop_size:-crop_size, :
                        ]
                        cropped_gt_img = gt_img[
                            crop_size:-crop_size, crop_size:-crop_size, :
                        ]
                        step_psnr = util.calculate_psnr(
                            cropped_sr_img * 255, cropped_gt_img * 255
                        )
                        logger.info(
                            "<epoch:{:3d}, iter:{:8,d}, step:{:3d}> img:{:s}, psnr: {:.6f}".format(
                                epoch, current_step, step, img_name, step_psnr
                            )
                        )
                        single_img_psnr += step_psnr
                        avg_psnr += util.calculate_psnr(
                            cropped_sr_img * 255, cropped_gt_img * 255
                        )

                    avg_signle_img_psnr = single_img_psnr / step
                    logger.info(
                        "<epoch:{:3d}, iter:{:8,d}, step:{:3d}> img:{:s}, average psnr: {:.6f}".format(
                            epoch, current_step, step, img_name, avg_signle_img_psnr
                        )
                    )

                avg_psnr = avg_psnr / idx

                # log
                logger.info("# Validation # PSNR: {:.6f}".format(avg_psnr))
                logger_val = logging.getLogger("val")  # validation logger
                logger_val.info(
                    "<epoch:{:3d}, iter:{:8,d}, step:{:3d}> psnr: {:.6f}".format(
                        epoch, current_step, step, avg_psnr
                    )
                )
                # tensorboard logger
                if opt_P["use_tb_logger"] and "debug" not in opt_P["name"]:
                    tb_logger.add_scalar("psnr", avg_psnr, current_step)

            #### save models and training states
            if current_step % opt_P["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    logger.info("Saving models and training states.")
                    model_P.save(current_step)
                    model_P.save_training_state(epoch, current_step)
                    model_C.save(current_step)
                    model_C.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info("Saving the final model.")
        model_P.save("latest")
        model_C.save("latest")
        logger.info("End of Predictor and Corrector training.")
    tb_logger.close()


if __name__ == "__main__":
    main()
