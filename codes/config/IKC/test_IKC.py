import argparse
import logging
import os.path
import time
from collections import OrderedDict

import numpy as np
import torch

import options as option
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler
from data.util import bgr2ycbcr
from models import create_model
from utils import util

#### options
parser = argparse.ArgumentParser()
parser.add_argument(
    "-opt_F",
    type=str,
    default="options/test/SFTMD/test_SFTMD_x4.yml",
    help="Path to options YMAL file.",
)
parser.add_argument(
    "-opt_P",
    type=str,
    default="options/test/Predictor/test_Predictor_x4.yml",
    help="Path to options YMAL file.",
)
parser.add_argument(
    "-opt_C",
    type=str,
    default="options/test/Corrector/test_Corrector_x4.yml",
    help="Path to options YMAL file.",
)
opt_F = option.parse(parser.parse_args().opt_F, is_train=False)
opt_P = option.parse(parser.parse_args().opt_P, is_train=False)
opt_C = option.parse(parser.parse_args().opt_C, is_train=False)

opt_F = option.dict_to_nonedict(opt_F)
opt_P = option.dict_to_nonedict(opt_P)
opt_C = option.dict_to_nonedict(opt_C)

#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt_P["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)
util.mkdirs(
    (
        path
        for key, path in opt_C["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)

os.system("rm ./result")
os.symlink(os.path.join(opt_P["path"]["results_root"], ".."), "./result")

util.setup_logger(
    "base",
    opt_P["path"]["log"],
    "test_" + opt_P["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt_P))
logger.info(option.dict2str(opt_C))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt_P["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model by default
model_F = create_model(opt_F)
model_P = create_model(opt_P)
model_C = create_model(opt_C)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]  # path opt['']
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt_P["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results["psnr"] = []
    test_results["ssim"] = []
    test_results["psnr_y"] = []
    test_results["ssim_y"] = []

    for test_data in test_loader:
        single_img_psnr = []
        single_img_ssim = []
        single_img_psnr_y = []
        single_img_ssim_y = []
        need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
        img_path = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
        img_name = img_path

        #### input dataset_LQ
        LR_img = test_data["LQ"]

        # Predictor test
        model_P.feed_data(LR_img)
        model_P.test()
        P_visuals = model_P.get_current_visuals()
        est_ker_map = P_visuals["Batch_est_ker_map"]

        # Corrector test
        for step in range(opt_C["step"]):
            step += 1
            # Test SFTMD to produce SR images
            model_F.feed_data(test_data, LR_img, est_ker_map)
            model_F.test()
            F_visuals = model_F.get_current_visuals()
            SR_img = F_visuals["Batch_SR"]

            model_C.feed_data(SR_img, est_ker_map)
            model_C.test()
            C_visuals = model_C.get_current_visuals()
            est_ker_map = C_visuals["Batch_est_ker_map"]

            sr_img = util.tensor2img(F_visuals["SR"])  # uint8

            # save images
            suffix = opt_P["suffix"]
            if suffix:
                save_img_path = os.path.join(
                    dataset_dir, img_name + suffix + "_" + str(step) + ".png"
                )
            else:
                save_img_path = os.path.join(
                    dataset_dir, img_name + "_" + str(step) + ".png"
                )
            util.save_img(sr_img, save_img_path)

            # calculate PSNR and SSIM
            if need_GT:
                gt_img = util.tensor2img(F_visuals["GT"])
                gt_img = gt_img / 255.0
                sr_img = sr_img / 255.0

                crop_border = (
                    opt_P["crop_border"] if opt_P["crop_border"] else opt_P["scale"]
                )
                if crop_border == 0:
                    cropped_sr_img = sr_img
                    cropped_gt_img = gt_img
                else:
                    cropped_sr_img = sr_img[
                        crop_border:-crop_border, crop_border:-crop_border, :
                    ]
                    cropped_gt_img = gt_img[
                        crop_border:-crop_border, crop_border:-crop_border, :
                    ]

                psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
                # test_results['psnr'].append(psnr)
                # test_results['ssim'].append(ssim)

                if gt_img.shape[2] == 3:  # RGB image
                    sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                    gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                    if crop_border == 0:
                        cropped_sr_img_y = sr_img_y
                        cropped_gt_img_y = gt_img_y
                    else:
                        cropped_sr_img_y = sr_img_y[
                            crop_border:-crop_border, crop_border:-crop_border
                        ]
                        cropped_gt_img_y = gt_img_y[
                            crop_border:-crop_border, crop_border:-crop_border
                        ]
                    psnr_y = util.calculate_psnr(
                        cropped_sr_img_y * 255, cropped_gt_img_y * 255
                    )
                    ssim_y = util.calculate_ssim(
                        cropped_sr_img_y * 255, cropped_gt_img_y * 255
                    )
                    # test_results['psnr_y'].append(psnr_y)
                    # test_results['ssim_y'].append(ssim_y)
                    single_img_psnr += [psnr]
                    single_img_ssim += [ssim]
                    single_img_psnr_y += [psnr_y]
                    single_img_ssim_y += [ssim_y]
                    logger.info(
                        "step:{:3d}, img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.".format(
                            step, img_name, psnr, ssim, psnr_y, ssim_y
                        )
                    )
                else:
                    logger.info(
                        "step:{:3d}, img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}.".format(
                            step, img_name, psnr, ssim
                        )
                    )
            else:
                logger.info(img_name)

        if need_GT:
            max_img_index = np.argmax(single_img_psnr)
            test_results["psnr"].append(single_img_psnr[max_img_index])
            test_results["ssim"].append(single_img_ssim[max_img_index])
            test_results["psnr_y"].append(single_img_psnr_y[max_img_index])
            test_results["ssim_y"].append(single_img_ssim_y[max_img_index])

            avg_signle_img_psnr = np.mean(single_img_psnr)
            avg_signle_img_ssim = np.mean(single_img_ssim)
            avg_signle_img_psnr_y = np.mean(single_img_psnr_y)
            avg_signle_img_ssim_y = np.mean(single_img_ssim_y)
            logger.info(
                "step:{:3d}, img:{:15s} - average PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.".format(
                    step,
                    img_name,
                    avg_signle_img_psnr,
                    avg_signle_img_ssim,
                    avg_signle_img_psnr_y,
                    avg_signle_img_ssim_y,
                )
            )
            max_signle_img_psnr = single_img_psnr[max_img_index]
            max_signle_img_ssim = single_img_ssim[max_img_index]
            max_signle_img_psnr_y = single_img_psnr_y[max_img_index]
            max_signle_img_ssim_y = single_img_ssim_y[max_img_index]
            logger.info(
                "step:{:3d}, img:{:15s} - max PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.".format(
                    step,
                    img_name,
                    max_signle_img_psnr,
                    max_signle_img_ssim,
                    max_signle_img_psnr_y,
                    max_signle_img_ssim_y,
                )
            )

    if need_GT:  # metrics
        # Average PSNR/SSIM results
        ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
        ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
        logger.info(
            "----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n".format(
                test_set_name, ave_psnr, ave_ssim
            )
        )
        if test_results["psnr_y"] and test_results["ssim_y"]:
            ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"])
            ave_ssim_y = sum(test_results["ssim_y"]) / len(test_results["ssim_y"])
            logger.info(
                "----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n".format(
                    ave_psnr_y, ave_ssim_y
                )
            )
