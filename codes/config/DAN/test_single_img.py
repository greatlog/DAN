import os.path
import logging
import time
import argparse
from collections import OrderedDict
import numpy as np
import torch
import options.options as option
from models import create_model
import sys

import cv2

sys.path.insert(0, '../../')
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, default='options/test/test_EM.yml', help='Path to options YMAL file.')
parser.add_argument('-img_path', type=str, default='../../../data_samples/chip.png')
args = parser.parse_args()
opt = option.parse(args.opt, is_train=False)

opt = option.dict_to_nonedict(opt)


# load pretrained model by default
model = create_model(opt)
img = cv2.imread(args.img_path)[:,:,[2,1,0]]
img = img.transpose(2, 0, 1)[None] / 255
img_t = torch.as_tensor(np.ascontiguousarray(img)).float()
model.feed_data(img_t)
model.test()
sr = model.fake_SR.detach().float().cpu()[0]
sr_im = util.tensor2img(sr)
cv2.imwrite('sr_img.png', sr_im)


