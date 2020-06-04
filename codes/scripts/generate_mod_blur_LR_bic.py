import os
import sys
import cv2
import numpy as np
import torch

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.util import imresize_np
    from utils import util
except ImportError:
    pass


def generate_mod_LR_bic():
    # set parameters
    up_scale = 4
    mod_scale = 4
    # set data dir
    sourcedir = '/data/DIV2K_public/gt_k_x4' #'/mnt/yjchai/SR_data/DIV2K_test_HR' #'/mnt/yjchai/SR_data/Flickr2K/Flickr2K_HR'
    savedir = '/data/DIV2KRK_public/x4HRblur.lmdb' #'/mnt/yjchai/SR_data/DIV2K_test' #'/mnt/yjchai/SR_data/Flickr2K_train'

    # set random seed
    util.set_random_seed(0)

    # load PCA matrix of enough kernel
    print('load PCA matrix')
    pca_matrix = torch.load('/data/IKC/pca_aniso_matrix.pth', map_location=lambda storage, loc: storage)
    print('PCA matrix shape: {}'.format(pca_matrix.shape))

    saveHRpath = os.path.join(savedir, 'HR', 'x' + str(mod_scale))
    saveLRpath = os.path.join(savedir, 'LR', 'x' + str(up_scale))
    saveBicpath = os.path.join(savedir, 'Bic', 'x' + str(up_scale))
    saveLRblurpath = os.path.join(savedir, 'LRblur', 'x' + str(up_scale))

    if not os.path.isdir(sourcedir):
        print('Error: No source data found')
        exit(0)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    if not os.path.isdir(os.path.join(savedir, 'HR')):
        os.mkdir(os.path.join(savedir, 'HR'))
    if not os.path.isdir(os.path.join(savedir, 'LR')):
        os.mkdir(os.path.join(savedir, 'LR'))
    if not os.path.isdir(os.path.join(savedir, 'Bic')):
        os.mkdir(os.path.join(savedir, 'Bic'))
    if not os.path.isdir(os.path.join(savedir, 'LRblur')):
        os.mkdir(os.path.join(savedir, 'LRblur'))

    if not os.path.isdir(saveHRpath):
        os.mkdir(saveHRpath)
    else:
        print('It will cover ' + str(saveHRpath))

    if not os.path.isdir(saveLRpath):
        os.mkdir(saveLRpath)
    else:
        print('It will cover ' + str(saveLRpath))

    if not os.path.isdir(saveBicpath):
        os.mkdir(saveBicpath)
    else:
        print('It will cover ' + str(saveBicpath))

    if not os.path.isdir(saveLRblurpath):
        os.mkdir(saveLRblurpath)
    else:
        print('It will cover '+ str(saveLRblurpath))

    filepaths = sorted([f for f in os.listdir(sourcedir) if f.endswith('.png')])
    print(filepaths)
    num_files = len(filepaths)

    # kernel_map_tensor = torch.zeros((num_files, 1, 10)) # each kernel map: 1*10

    # prepare data with augementation
    for i in range(num_files):
        filename = filepaths[i]
        print('No.{} -- Processing {}'.format(i, filename))
        # read image
        image = cv2.imread(os.path.join(sourcedir, filename))

        width = int(np.floor(image.shape[1] / mod_scale))
        height = int(np.floor(image.shape[0] / mod_scale))
        # modcrop
        if len(image.shape) == 3:
            image_HR = image[0:mod_scale * height, 0:mod_scale * width, :]
        else:
            image_HR = image[0:mod_scale * height, 0:mod_scale * width]
        # LR_blur, by random gaussian kernel
        img_HR = util.img2tensor(image_HR)
        C, H, W = img_HR.size()
        # sig_list = [1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2]
        # # sig = 2.6
        # for sig in np.linspace(1.8, 3.2, 8):
        prepro = util.SRMDPreprocessing(up_scale, pca_matrix, random=True, para_input=10, kernel=11, noise=False,
                                        cuda=True, sig=0, sig_min=0.6, sig_max=5, rate_iso=0, scaling=3,
                                        rate_cln=0.2, noise_high=0.0) #random(sig_min, sig_max) | stable kernel(sig)
        LR_img, ker_map = prepro(img_HR.view(1, C, H, W))
        image_LR_blur = util.tensor2img(LR_img)
        cv2.imwrite(os.path.join(saveLRblurpath, filename), image_LR_blur)
        cv2.imwrite(os.path.join(saveHRpath, filename), image_HR)
        # LR
        image_LR = imresize_np(image_HR, 1 / up_scale, True)
        # bic
        image_Bic = imresize_np(image_LR, up_scale, True)

        # cv2.imwrite(os.path.join(saveHRpath, filename), image_HR)
        cv2.imwrite(os.path.join(saveLRpath, filename), image_LR)
        cv2.imwrite(os.path.join(saveBicpath, filename), image_Bic)

        # kernel_map_tensor[i] = ker_map
    # save dataset corresponding kernel maps
    # torch.save(kernel_map_tensor, './Set5_sig2.6_kermap.pth')
    print("Image Blurring & Down smaple Done: X"+str(up_scale))

if __name__ == "__main__":
    generate_mod_LR_bic()
