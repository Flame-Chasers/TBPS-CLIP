<div>

# 【AAAI 2024 🔥】An Empirical Study of CLIP for Text-based Person Search
[![Paper](http://img.shields.io/badge/Paper-AAAI_2024-Green.svg)](https://ojs.aaai.org/index.php/AAAI/article/view/27801)
[![Paper](http://img.shields.io/badge/Paper-arxiv.2308.10045-FF6B6B.svg)](https://arxiv.org/abs/2308.10045)
</div>

This repository offers the official implementation of [TBPS-CLIP](https://arxiv.org/abs/2308.10045) in PyTorch.

In the meantime, check out our related papers if you are interested:
+ 【ACM MM 2023】 [Text-based Person Search without Parallel Image-Text Data](https://arxiv.org/abs/2305.12964)
+ 【IJCAI 2023】 [RaSa: Relation and Sensitivity Aware Representation Learning for Text-based Person Search](https://arxiv.org/abs/2305.13653)
+ 【ICASSP 2022】 [Learning Semantic-Aligned Feature Representation for Text-based Person Search](https://arxiv.org/abs/2112.06714)

## Note 
More experiments and implementation details are attached on the Appendix of the [arXiv](https://arxiv.org/abs/2308.10045) version.


## Overview
By revisiting the critical design of data augmentation and loss function in [CLIP](https://arxiv.org/abs/2103.00020),
we provide a strong baseline [TBPS-CLIP](https://arxiv.org/abs/2308.10045) for text-based person search.

<img src="image/intro.png" width="550">


## Environment

All the experiments are conducted on 4 Nvidia A40 (48GB) GPUs. The CUDA version is 11.7.

The required packages are listed in `requirements.txt`. You can install them using:

```sh
pip install -r requirements.txt
```

## Download
1. Download CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description), ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN) and RSTPReid dataset from [here](https://github.com/NjtechCVLab/RSTPReid-Dataset).
2. Download the annotation json files from [here](https://drive.google.com/file/d/1C5bgGCABtuzZMaa2n4Sc0qclUvZ-mqG9/view?usp=drive_link).
3. Download the pretrained CLIP checkpoint from [here](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt).

## Configuration
In `config/config.yaml` and `config/s.config.yaml`, set the paths for the annotation file, image path and the CLIP checkpoint path.


## Training

You can start the training using PyTorch's torchrun with ease:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --rdzv_id=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=4 \
main.py
```

You can also easily run simplified version using:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --rdzv_id=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=4 \
main.py --simplified
```


## Model Checkpoints
|                                     |                                          **CUHK-PEDES**                                           | **ICFG-PEDES** | **RSTPReid** |
|:-----------------------------------:|:-------------------------------------------------------------------------------------------------:|:--------------:|:------------:|
| **TBPS-CLIP (ViT-B/16)**            | [Download](https://drive.google.com/file/d/1m_3pKanUWHQHeJ-zt-QeRXs7bmay-U5P/view?usp=drive_link) |  [Download](https://drive.google.com/file/d/1az4z5b_ADXR7DcysPB5giOl52LjWDCSu/view?usp=drive_link)  | [Download](https://drive.google.com/file/d/1qMUAsH-1lzkWUFQsUvUKTY0J6ZuGkYd6/view?usp=drive_link)         |
| **Simplified TBPS-CLIP (ViT-B/16)** | [Download](https://drive.google.com/file/d/1W5oFZK9WNHMfy0OOaYQBzPsP1LZR80bT/view?usp=drive_link) |  [Download](https://drive.google.com/file/d/1UoLd-MQ8tYJ7YPgCbh3nVSVYnJ9a_TG5/view?usp=drive_link)  | [Download](https://drive.google.com/file/d/18zlc3q3Sze5rx3TqcfEeZEjrQXUTpcQF/view?usp=drive_link)         |


## Acknowledgement
+ [CLIP](https://arxiv.org/abs/2103.00020) The model architecture of TBPS-CLIP

## Citation
If you find this paper useful, please consider staring 🌟 this repo and citing 📑 our paper:
```
@inproceedings{cao2024empirical,
  title={An Empirical Study of CLIP for Text-Based Person Search},
  author={Cao, Min and Bai, Yang and Zeng, Ziyin and Ye, Mang and Zhang, Min},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={1},
  pages={465--473},
  year={2024}
}
```


## License
This code is distributed under an MIT LICENSE.
