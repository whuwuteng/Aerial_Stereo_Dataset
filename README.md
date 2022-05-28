# A Very Large Aerial Stereo Dense Matching Benchmark Open Dataset

This is the Github repository for the stereo dense matching benchmark hosted at [AI4GEO project](http://ai4geo.eu/index.php). 

In order to discuss the transferability of deep learning methods on aerial dataset, we produce **6** aerial dataset covers **4** different area.

## Introduction

For stereo dense matching, there are many famous benchmark dataset in Robust Vision, for example, [KITTI stereo](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) and [middlebury stereo](https://vision.middlebury.edu/stereo/).
With the development of machine learning, especially deep learning, these methods usually need a lot of training data(or ground truth). 
For photogrammetry community, as far as we know, it is not easy to find these training data. We will publish our data as ground truth. The data is produced from original image and LiDAR dataset. To be noticed, the image and LiDAR should be well-registered.

### Global information of dataset

For each dataset, the global information of the dataset is listed follow:

<!-- check refer to http://wfeii.com/2021/10/14/markdown-code.html -->

|     Dataset      | Color | GSD(cm) | LiDAR(<img src="https://render.githubusercontent.com/render/math?math=\large pt/m^2">) |Origin orientation|ICP refined|
| :----------: | :-----------: | :-----------: | :-----------: |:-----------: | :----------: |
|ISPRS-Vaihingen|IR-R-G|      8      |      6.7      | &#10004; |x|
|EuroSDR-Vaihingen|R-G-B|      20      |      6.7      | &#10004; |x|
|Toulouse-UMBRA|R-G-B|      12.5      |      2-4      | x |&#10004;|
|Toulouse-Métropole|R-G-B|      5      |      8     | &#10004; |x|
|Enschede|R-G-B|      10      |      10      |x |&#10004;|
|DublinCity|R-G-B|      3.4      |      250-348      |x |&#10004;|

In the table, the origin orientation quality is according to the data accuracy, dataset size. 

### Dataset structure

The training and evluation dataset is also provided, the structure of the folder is 


## TODO

- [x] Image-LiDAR process
- [ ] Publish dataset V1
- [ ] Five cameras process


### Citation

If you think you have any problem, contact [Teng Wu]<whuwuteng@gmail.com>

