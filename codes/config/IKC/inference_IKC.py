import argparse
import logging
import os.path as osp
import os
import time
from collections import OrderedDict
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

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
parser.add_argument("-input_dir", type=str, default="../../../data_samples/LR")
parser.add_argument("-output_dir", type=str, default="../../../data_samples/IKC_SR")

args = parser.parse_args()

opt_F = option.parse(parser.parse_args().opt_F, is_train=False)
opt_P = option.parse(parser.parse_args().opt_P, is_train=False)
opt_C = option.parse(parser.parse_args().opt_C, is_train=False)

opt_F = option.dict_to_nonedict(opt_F)
opt_P = option.dict_to_nonedict(opt_P)
opt_C = option.dict_to_nonedict(opt_C)

# load pretrained model by default
model_F = create_model(opt_F)
model_P = create_model(opt_P)
model_C = create_model(opt_C)

if not osp.exists(args.output_dir):
    os.makedirs(args.output_dir)

test_files = glob(osp.join(args.input_dir, "*png"))
for inx, path in tqdm(enumerate(test_files)):
    name = path.split("/")[-1].split(".")[0]

    img = cv2.imread(path)[:, :, [2, 1, 0]]
    img = img.transpose(2, 0, 1)[None] / 255
    LR_img = torch.as_tensor(np.ascontiguousarray(img)).float()

    model_P.feed_data(LR_img)
    model_P.test()
    P_visuals = model_P.get_current_visuals()
    est_ker_map = P_visuals["Batch_est_ker_map"]

    for step in range(opt_C["step"]):
        step += 1
        # Test SFTMD to produce SR images
        model_F.feed_data(LR_img = LR_img, ker_map = est_ker_map)
        model_F.test()
        F_visuals = model_F.get_current_visuals()
        SR_img = F_visuals["Batch_SR"]

        model_C.feed_data(SR_img, est_ker_map)
        model_C.test()
        C_visuals = model_C.get_current_visuals()
        est_ker_map = C_visuals["Batch_est_ker_map"]

        sr_img = util.tensor2img(F_visuals["SR"])  # uint8

        save_path = osp.join(args.output_dir, "{}_x{}.png".format(name, opt_P["scale"]))
        cv2.imwrite(save_path, sr_img)