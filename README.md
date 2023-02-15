# HAT [[Paper Link]](https://arxiv.org/abs/2205.04437) [![Replicate](https://replicate.com/cjwbw/hat/badge)](https://replicate.com/cjwbw/hat)

### Activating More Pixels in Image Super-Resolution Transformer
[Xiangyu Chen](https://chxy95.github.io/), [Xintao Wang](https://xinntao.github.io/), [Jiantao Zhou](https://scholar.google.com/citations?hl=zh-CN&user=mcROAxAAAAAJ) and [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=zh-CN)

#### BibTeX

    @article{chen2022activating,
      title={Activating More Pixels in Image Super-Resolution Transformer},
      author={Chen, Xiangyu and Wang, Xintao and Zhou, Jiantao and Dong, Chao},
      journal={arXiv preprint arXiv:2205.04437},
      year={2022}
    }
**Google Scholar has unknown bugs for indexing this paper recently, while it can still be cited by the above BibTeX.**

## Updates
- ✅ Upload a GAN-based HAT models for Real-World Super-Resolution (Real_HAT_GAN_SRx4.pth).
- **(To do)** Add the tile mode for Replicate demo.
- **(To do)** Update the Replicate demo for Real-World Super-Resolution.
- **(To do)** Upate the training configs for the Real-World GAN-based HAT model.
- **(To do)** Add HAT models for Multiple Image Restoration tasks.

## Overview
<img src="https://raw.githubusercontent.com/chxy95/HAT/master/figures/Performance_comparison.png" width="600"/>

## Environment
- [PyTorch >= 1.7](https://pytorch.org/) **(Recommend **NOT** using torch 1.8!!! It would cause abnormal performance.)**
- [BasicSR == 1.3.4.9](https://github.com/XPixelGroup/BasicSR/blob/master/INSTALL.md) 
### Installation
```
pip install -r requirements.txt
python setup.py develop
```

## How To Test
- Refer to `./options/test` for the configuration file of the model to be tested, and prepare the testing data and pretrained model.  
- The pretrained models are available at
[Google Drive](https://drive.google.com/drive/folders/1HpmReFfoUqUbnAOQ7rvOeNU3uf_m69w0?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1u2r4Lc2_EEeQqra2-w85Xg) (access code: qyrl).  
- Then run the follwing codes (taking `HAT_SRx4_ImageNet-pretrain.pth` as an example):
```
python hat/test.py -opt options/test/HAT_SRx4_ImageNet-pretrain.yml
```
The testing results will be saved in the `./results` folder.  
  
**Note that the tile mode is also provided for limited GPU memory when testing. You can modify the specific settings of the tile mode in your custom testing option by referring to `./options/test/HAT_tile_example.yml`.**

## How To Train
- Refer to `./options/train` for the configuration file of the model to train.
- Preparation of training data can refer to [this page](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md). ImageNet dataset can be downloaded at the [official website](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php).
- The training command is like
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 hat/train.py -opt options/train/train_HAT_SRx2_from_scratch.yml --launcher pytorch
```
- Note that the default batch size per gpu is 4, which will cost about 20G memory for each GPU.  

The training logs and weights will be saved in the `./experiments` folder.

## Results
The inference results on benchmark datasets are available at
[Google Drive](https://drive.google.com/drive/folders/1t2RdesqRVN7L6vCptneNRcpwZAo-Ub3L?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1CQtLpty-KyZuqcSznHT_Zw) (access code: 63p5).


## Contact
If you have any question, please email chxy95@gmail.com or join in the [Wechat group of BasicSR](https://github.com/XPixelGroup/BasicSR#-contact) to discuss with the authors.
