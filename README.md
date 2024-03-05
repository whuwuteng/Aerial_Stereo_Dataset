# An evaluation of Deep Learning based stereo dense matching dataset shift from aerial images and a large scale stereo dataset

This Github will provide the detail information of our paper [An evaluation of Deep Learning based stereo dense matching dataset shift from aerial images and a large scale stereo dataset](https://www.sciencedirect.com/science/article/pii/S1569843224000694).

Because the length limit of [International Journal of Applied Earth Observation and Geoinformation](https://www.sciencedirect.com/journal/international-journal-of-applied-earth-observation-and-geoinformation) is 8000 words, the original paper can be found on [Arxiv](https://arxiv.org/abs/2402.12522), if you have any questions about this long version paper, please contact me.

This is the Github repository for the stereo dense matching benchmark for [AI4GEO project](http://ai4geo.eu/index.php). 

In order to discuss the transferability of deep learning methods on aerial dataset, we produce **6** aerial dataset covers **4** different area. 

## History

This work is an extension of our [previous work](https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLIII-B2-2021/405/2021/), and the [old version](https://github.com/whuwuteng/benchmark_ISPRS2021) dataset is already published. In the [ISPRS Conress 2022 in Nice](https://www.isprs2022-nice.com/), we presented an extension work  as a poster, and the [slide](congress_ISPRS2022/Slide_ISPRS2022.pdf) and the [poster](congress_ISPRS2022/Poster_ISPRS2022.pdf) is provided.


## Introduction

For stereo dense matching, there are many famous benchmark dataset in Robust Vision, for example, [KITTI stereo](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) and [middlebury stereo](https://vision.middlebury.edu/stereo/).
With the development of machine learning, especially deep learning, these methods usually need a lot of training data(or ground truth). 
For photogrammetry community, as far as we know, it is not easy to find these training data. We will publish our data as ground truth. The data is produced from original image and LiDAR dataset. To be noticed, the image and LiDAR should be well-registered.

### Global information of dataset

For each dataset, the global information of the dataset is listed follow:

|     Dataset      | Color | GSD(cm) | LiDAR($pt/m^2$) |Origin orientation|ICP refined|Outlier remove|
| :----------: | :-----------: | :-----------: | :-----------: |:-----------: | :----------: | :----------: |
|ISPRS-Vaihingen|IR-R-G|      8      |      6.7      | &#10004; |`x`|`x`|
|EuroSDR-Vaihingen|R-G-B|      20      |      6.7      | &#10004; |`x`|`x`|
|Toulouse-UMBRA|R-G-B|      12.5      |      2-4      | `x` |&#9745;|&#9745;|
|Toulouse-Métropole|R-G-B|      5      |      8     | &#10004; |`x`|`x`|
|Enschede|R-G-B|      10      |      10      |`x` |&#9745;|&#9745;|
|DublinCity|R-G-B|      3.4      |      250-348      |`x` |&#9745;|`x`|

In the table, the origin orientation accuracy  influence the data accuracy, in order to improve the quality of the dataset, an ICP based Image-LiDAR is proposed to refine the orientation. 

### Dataset structure

The training and evaluation dataset is also provided, the structure of the folder is same with the [old version](https://github.com/whuwuteng/benchmark_ISPRS2021).

Because the whole dataset is  too large, so only the used in the paper is uploaded.

### ISPRS-Vaihingen

All the training and testing data can be found on [Google Drive](https://drive.google.com/file/d/1Gcap1_p13QJoF7ShLP8sEBsSY1SNRgk4/view?usp=sharing), this is a newer version compare to the [old version](https://github.com/whuwuteng/benchmark_ISPRS2021), the using origin image and LiDAR is same with [old version](https://github.com/whuwuteng/benchmark_ISPRS2021).

### EuroSDR-Vaihingen

This dataset is collected nearly same time with ISPRS-Vaihingen, the difference is that the resolution.



## DublinCity 

[DublinCity](https://v-sense.scss.tcd.ie/dublincity/) is an open dataset, the original aerial and LiDAR point cloud can be [downloaded](https://geo.nyu.edu/catalog/nyu-2451-38684), the origin dataset is very large.

You can find the training and testing dataset from [another paper](https://openaccess.thecvf.com/content/CVPR2023W/PCV/html/Wu_PSMNet-FusionX3_LiDAR-Guided_Deep_Learning_Stereo_Dense_Matching_on_Aerial_Images_CVPRW_2023_paper.html). To save the disk, we do not upload this time, more information can be found on [Github](https://github.com/whuwuteng/PSMNet-FusionX3) also.

###  Method

In the paper, we evaluate the state of the art methods of deep learning on stereo dense matching before 2020. 

###  Pretrained models

The pretrained models are important in the paper, so we will also share the pretrained models and training setting in the paper.


## TODO

- [x] Image-LiDAR process
- [ ] Publish dataset V1 (use in the paper)
- [x] Publish the long paper on Arxiv
- [ ] Publish pretrained models
- [ ] Publish full dataset

## Stereo-LiDAR fusion

Based on the data generation,  we also generate the Toulouse2020 data from IGN, and this data can be found in our CVPR photogrammetry and computer vision workshop paper. The Github site can be found [here](https://github.com/whuwuteng/PSMNet-FusionX3).

### Citation

If you think you have any problem, contact [Teng Wu]<whuwuteng@gmail.com>

