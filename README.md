# An evaluation of Deep Learning based stereo dense matching dataset shift from aerial images and a large scale stereo dataset

This Github will provide the detail information of our paper (**JAG-D-23-00976**) under review on  [International Journal of Applied Earth Observation and Geoinformation](https://track.authorhub.elsevier.com/?uuid=fc239731-c1a9-4c4c-9104-bc1f55105626).

This is the Github repository for the stereo dense matching benchmark hosted at [AI4GEO project](http://ai4geo.eu/index.php). 

In order to discuss the transferability of deep learning methods on aerial dataset, we produce **6** aerial dataset covers **4** different area. 

## History

This work is an extension of our [previous work](https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLIII-B2-2021/405/2021/), and the [old version](https://github.com/whuwuteng/benchmark_ISPRS2021) dataset is already published. In the [ISPRS Conress 2022 in Nice](https://www.isprs2022-nice.com/), we presented an extension work  as a poster, and the [slide](congress_ISPRS2022/Slide_ISPRS2022.pdf) and the [poster](congress_ISPRS2022/Poster_ISPRS2022.pdf) is provided.


## Introduction

For stereo dense matching, there are many famous benchmark dataset in Robust Vision, for example, [KITTI stereo](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) and [middlebury stereo](https://vision.middlebury.edu/stereo/).
With the development of machine learning, especially deep learning, these methods usually need a lot of training data(or ground truth). 
For photogrammetry community, as far as we know, it is not easy to find these training data. We will publish our data as ground truth. The data is produced from original image and LiDAR dataset. To be noticed, the image and LiDAR should be well-registered.

### Global information of dataset

For each dataset, the global information of the dataset is listed follow:

|     Dataset      | Color | GSD(cm) | LiDAR($pt/m^2$) |Origin orientation|ICP refined|
| :----------: | :-----------: | :-----------: | :-----------: |:-----------: | :----------: |
|ISPRS-Vaihingen|IR-R-G|      8      |      6.7      | &#10004; |`x`|
|EuroSDR-Vaihingen|R-G-B|      20      |      6.7      | &#10004; |`x`|
|Toulouse-UMBRA|R-G-B|      12.5      |      2-4      | `x` |&#9745;|
|Toulouse-Métropole|R-G-B|      5      |      8     | &#10004; |`x`|
|Enschede|R-G-B|      10      |      10      |`x` |&#9745;|
|DublinCity|R-G-B|      3.4      |      250-348      |`x` |&#9745;|

In the table, the origin orientation accuracy  influence the data accuracy, in order to improve the quality of the dataset, an ICP based Image-LiDAR is proposed to refine the orientation. 

### Dataset structure

The training and evaluation dataset is also provided, the structure of the folder is same with the [old version](https://github.com/whuwuteng/benchmark_ISPRS2021).




## TODO

- [x] Image-LiDAR process
- [ ] Publish dataset V1
- [ ] Publish the long paper on Arxiv


### Citation

If you think you have any problem, contact [Teng Wu]<whuwuteng@gmail.com>

