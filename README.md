This is an official implementation of [Unfolding the Alternating Optimization for Blind Super Resolution](https://arxiv.org/abs/2010.02631)

If this repo works for you, please cite our paper:
```
@article{luo2020unfolding,
  title={Unfolding the Alternating Optimization for Blind Super Resolution},
  author={Luo, Zhengxiong and Huang, Yan and Li, Shang and Wang, Liang and Tan, Tieniu},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={33},
  year={2020}
}
```

This repo is build on the basis of [[MMSR]](https://github.com/open-mmlab/mmsr) and [[IKC]](https://github.com/yuanjunchai/IKC)

## News
[x] Add more pretrained weights and update the results of DANv1 \!
[x] Add pretrained weights and update the results of about [[IKC]](https://github.com/yuanjunchai/IKC)\!
[]  Add DANv2

## Main Results

### Results about Setting 1
| Method | Scale |   Set5  |   Set5   | Set14   |  Set14   | B100   |  B100    | Urban100 | Urban100  | Mangan109 | Manga109  |
|:------:|:-----:|:-------:|:--------:|:-------:|---------|:-------:|----------|:-------:|-----------|:-------:|:-------:|
|        |       | PSNR    | SSIM     | PSNR    | SSIM     | PSNR    | SSIM     | PSNR     | SSIM     | PSNR      | SSIM     |
| IKC    | x2    |  37.19  |  0.9526  |  32.94  |  0.9024  |  31.51  |  0.8790  |  29.85   |  0.8928  |  36.93    |  0.9667  |
| DANv1  | x2    |  37.34  |  0.9526  |  33.08  |  0.9041  |  31.76  |  0.8858  |  30.60   |  0.9060  |  37.23    |  0.9710  |
| IKC    | x3    |  33.06  |  0.9146  |  29.38  |  0.8233  |  28.53  |  0.7899  |  27.43   |  0.8302  |  32.43    |  0.9316  |
| DANv1  | x3    |  34.04  |  0.9199  |  30.09  |  0.8287  |  28.94  |  0.7919  |  27.65   |  0.8352  |  33.16    |  0.9382  |
| IKC    | x4    |  31.67  |  0.8829  |  28.31  |  0.7643  |  27.37  |  0.7192  |  25.33   |  0.7504  |  28.91    |  0.8782  |
| DANv1  | x4    |  31.89  |  0.8864  |  28.42  |  0.7687  |  27.51  |  0.7248  |  25.86   |  0.7721  |  30.50    |  0.9037  |

### Results about Setting 2 (DIV2KRK)

|Method          |  x2   |  x2    |  x4   |  x4    
|:--------------:|:-----:|:------:|:-----:|:------:|
|                | PSNR  | SSIM   | PSNR  |   SSIM |
|KernelGAN + ZSSR| 30.36 | 0.8669 | 26.81 | 0.7316 |
|DANv1           | 32.56 | 0.8997 | 27.55 | 0.7582 |


## Dependenices

* python3
* pytorch >= 1.5
* NVIDIA GPU + CUDA
* Python packages: pip3 install numpy opencv-python lmdb pyyaml

## Pretrained Weights
Pretrained weights of DANv1 and IKC are available at [BaiduYun](https://pan.baidu.com/s/15843FMaiGoREB-8sdmJc4g)(Password: cbjv) or [GoogleDrive](https://drive.google.com/drive/folders/1aOaxXFfMAM6GvvPz56lzUwVmg7BMM_mR?usp=sharing). Download the weights to [checkpoints](./checkpoints)

```
.
|-- checkpoints
`-- |-- DANv1
    |   |-- ...
    `--IKC
        |-- ... 
```

The pretrained models and visual results on DIV2KRK are available at [BaiduYun](https://pan.baidu.com/s/1KOnoIplZmF9XheARW_tM-w)(Password: u9ah) [GoogleDrive](https://drive.google.com/drive/folders/1wdlVOcM8faEoTUZhFFsRVKVPiaZjfBf5?usp=sharing)


## Dataset Preparation
We use [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) as our training datasets. 

For evaluation of Setting 1, we use five datasets, i.e., [Set5](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip), [Set14](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip), [Urban100](https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip), [BSD100](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip) and [Manga109](http://www.manga109.org/en/).

We use [DIV2KRK](http://www.wisdom.weizmann.ac.il/~vision/kernelgan/DIV2KRK_public.zip) for evaluation of Setting 2.

To train a model on the full dataset(DIV2K+Flickr2K, totally 3450 images), download datasets from official websites. 
After download, run [`codes/scripts/generate_mod_blur_LR_bic.py`](codes/scripts/generate_mod_blur_LR_bic.py) to generate LRblur/LR/HR/Bicubic datasets paths. (You need to modify the file paths by yourself.)

```bash
python3 codes/scripts/generate_mod_blur_LR_bic.py
```

For efficient IO, run [`codes/scripts/create_lmdb.py`](codes/scripts/create_lmdb.py) to transform datasets to binary files. (You need to modify the file paths by yourself.)

```bash
python3 codes/scripts/create_lmdb.py
```

## Train

For single GPU:
```bash
cd codes/config/DANv1
python3 train.py -opt=train_setting1.yml
```

For distributed training
```bash
cd codes/config/DANv1
bash run_scripts.sh
```


## Test on Synthetic Images
```bash
cd codes/config/DANv1
python3 test.py -opt=test_setting1.yml
```

## Test on Real Images
```bash
cd codes/config/DANv1
python3 test_single_img.py -opt=test_option.yml -input_dir=/path/to/real/images/ -output_dir=/path/to/save/sr/results/
```
