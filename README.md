This repo is build on the basis of mmsr [[MMSR]](https://github.com/open-mmlab/mmsr) and [[IKC]](https://github.com/yuanjunchai/IKC)

# Dependenices

* python3
* pytorch > 1.0
* NVIDIA GPU + CUDA
* Python packages: pip3 install numpy opencv-python lmdb pyyaml


## Dataset Preparation
We use [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/), [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar), [Set5](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip), [Set14](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip), [Urban100](https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip), [BSD100](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip) datasets. 
To train a model on the full dataset(DIV2K+Flickr2K, totally 3450 images), download datasets from official websites. 
After download, run [`codes/scripts/generate_mod_LR_bic.py`](codes/scripts/generate_mod_LR_bic.py) to generate LRblur/LR/HR/Bicubic datasets paths. 
```bash
python3 codes/scripts/generate_mod_LR_bic.py
```

For efficient IO, run  run [`codes/scripts/create_lmdb.py`](codes/scripts/create_lmdb.py) to transform datasets to binary files.

```bash
python3 codes/scripts/create_lmdb.py
```

## Train

```bash
cd codes/config/DAN
python3 train.py -opt=train_option.yml
```

For distributed training
```bash
cd codes/config/DAN
bash run.sh
```


## Test
```bash
cd codes/config/DAN
python3 test.py -opt=test_option.yml
```
The path of pretrained model in [`codes/config/DAN/test_option.yml`](codes/config/DAN/test_option.yml) needs to be modified to the tested model.

## Test Real image
```bash
cd codes/config/DAN
python3 test_single_image.py -opt=test_option.yml
```
