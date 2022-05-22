# HAT [[Paper Link]](https://arxiv.org/abs/2205.04437)

### Activating More Pixels in Image Super-Resolution Transformer
Xiangyu Chen, [Xintao Wang](https://scholar.google.com.hk/citations?user=FQgZpQoAAAAJ&hl=en), [Jiantao Zhou](https://scholar.google.com/citations?hl=zh-CN&user=mcROAxAAAAAJ) and [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=zh-CN)

<img src="https://raw.githubusercontent.com/chxy95/HAT/master/figures/Performance_comparison.png" width="600"/>

#### BibTeX

    @article{chen2022activating,
      title={Activating More Pixels in Image Super-Resolution Transformer},
      author={Chen, Xiangyu and Wang, Xintao and Zhou, Jiantao and Dong, Chao},
      journal={arXiv preprint arXiv:2205.04437},
      year={2022}
    }

## Environment
- [PyTorch >= 1.7](https://pytorch.org/)
- [BasicSR >= 1.3.4.9](https://github.com/XPixelGroup/BasicSR/blob/master/INSTALL.md) 
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

## Results
The inference results on benchmark datasets are available at
[Google Drive](https://drive.google.com/drive/folders/1t2RdesqRVN7L6vCptneNRcpwZAo-Ub3L?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1CQtLpty-KyZuqcSznHT_Zw) (access code: 63p5).


### This repo is still being updated. The training codes will be released soon.
