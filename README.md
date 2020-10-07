This is an offical implementation of [Unfolding the Alternating Optimization for Blind Super Resolution](https://arxiv.org/abs/2010.02631)

This repo is build on the basis of mmsr [[MMSR]](https://github.com/open-mmlab/mmsr) and [[IKC]](https://github.com/yuanjunchai/IKC)

## Dependenices

* python3
* pytorch > 1.0
* NVIDIA GPU + CUDA
* Python packages: pip3 install numpy opencv-python lmdb pyyaml

## Models

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

For efficient IO, run  run [`codes/scripts/create_lmdb.py`](codes/scripts/create_lmdb.py) to transform datasets to binary files. (You need to modify the file paths by yourself.)

```bash
python3 codes/scripts/create_lmdb.py
```

## Train

For single GPU:
```bash
cd codes/config/DAN
python3 train.py -opt=train_option.yml
```

For distributed training
```bash
cd codes/config/DAN
bash run.sh
```


## Test on Synthetic Images
```bash
cd codes/config/DAN
python3 test.py -opt=test_option.yml
```

## Test on Real Images
```bash
cd codes/config/DAN
python3 test_single_image.py -opt=test_option.yml -input_dir=/path/to/real/images/ -output_dir=/path/to/save/sr/results/
```
