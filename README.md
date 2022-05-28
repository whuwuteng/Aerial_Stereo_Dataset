# A Very Large Aerial Stereo Dense Matching Benchmark Open Dataset

This is the Github repository for the stereo dense matching benchmark hosted at [AI4GEO project](http://ai4geo.eu/index.php). 

In order to discuss the transferability of deep learning methods on aerial dataset, we produce **6** aerial dataset covers **4** different area. 

This work is an extension of our [previous work](https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLIII-B2-2021/405/2021/), and the [old version](https://github.com/whuwuteng/benchmark_ISPRS2021) dataset is already published.

## ISPRS Congress 2022

In the [ISPRS Conress 2022](https://www.isprs2022-nice.com/), we will present this work, and the [slide](Poster_ISPRS2022.pdf) and the [poster](Poster_ISPRS2022.pdf) is provided.


## Introduction

For stereo dense matching, there are many famous benchmark dataset in Robust Vision, for example, [KITTI stereo](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) and [middlebury stereo](https://vision.middlebury.edu/stereo/).
With the development of machine learning, especially deep learning, these methods usually need a lot of training data(or ground truth). 
For photogrammetry community, as far as we know, it is not easy to find these training data. We will publish our data as ground truth. The data is produced from original image and LiDAR dataset. To be noticed, the image and LiDAR should be well-registered.

### Global information of dataset

For each dataset, the global information of the dataset is listed follow:

<!-- check refer to http://wfeii.com/2021/10/14/markdown-code.html -->
<!-- comment refer to https://www.w3cschool.cn/lme/q92a1srq.html -->

|     Dataset      | Color | GSD(cm) | LiDAR(<img src="https://render.githubusercontent.com/render/math?math=\large pt/m^2">) |Origin orientation|ICP refined|
| :----------: | :-----------: | :-----------: | :-----------: |:-----------: | :----------: |
|ISPRS-Vaihingen|IR-R-G|      8      |      6.7      | &#10004; |`x`|
|EuroSDR-Vaihingen|R-G-B|      20      |      6.7      | &#10004; |`x`|
|Toulouse-UMBRA|R-G-B|      12.5      |      2-4      | `x` |&#9745;|
|Toulouse-Métropole|R-G-B|      5      |      8     | &#10004; |`x`|
|Enschede|R-G-B|      10      |      10      |`x` |&#9745;|
|DublinCity|R-G-B|      3.4      |      250-348      |`x` |&#9745;|

In the table, the origin orientation accuracy  influence the data accuracy, in order to improve the quality of the dataset, an ICP based Image-LiDAR is proposed to refine the orientation. 

### Dataset structure

The training and evluation dataset is also provided, the structure of the folder is same with the [old version](https://github.com/whuwuteng/benchmark_ISPRS2021).


## TODO

- [x] Image-LiDAR process
- [ ] Publish dataset V1
- [ ] Five cameras process


### Citation

If you think you have any problem, contact [Teng Wu]<whuwuteng@gmail.com>

