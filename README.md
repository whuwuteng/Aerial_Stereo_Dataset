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

|     Dataset      | Color | GSD(cm) | LiDAR($pt/m^2$) |Origin orientation|ICP refined|Outlier remove| Difficulty|
| :----------: | :-----------: | :-----------: | :-----------: |:-----------: | :----------: | :----------: | :----------: |
|ISPRS-Vaihingen|IR-R-G|      8      |      6.7      | &#10004; |`x`|`x`|++|
|EuroSDR-Vaihingen|R-G-B|      20      |      6.7      | &#10004; |`x`|`x`|++|
|Toulouse-UMBRA|R-G-B|      12.5      |      2-4      | `x` |&#9745;|&#9745;|++++|
|Toulouse-Métropole|R-G-B|      5      |      8     | &#10004; |`x`|`x`|+|
|Enschede|R-G-B|      10      |      10      |`x` |&#9745;|&#9745;|+++|
|DublinCity|R-G-B|      3.4      |      250-348      |`x` |&#9745;|`x`|++|

In the table, the origin orientation accuracy  influence the data accuracy, in order to improve the quality of the dataset, an ICP based Image-LiDAR is proposed to refine the orientation. 

### Dataset structure

The training and evaluation dataset is also provided, the structure of the folder is same with the [old version](https://github.com/whuwuteng/benchmark_ISPRS2021).

Because the whole dataset is  too large, so only the used in the paper is uploaded. The data will be hosted by [Zenodo](https://zenodo.org/), because the limitation is **50GB**, so the **dataset used** be in the paper and the **pre-trained models** are available on Zenodo. 

### ISPRS-Vaihingen

All the training and testing data can be found on [Google Drive](https://drive.google.com/file/d/1Gcap1_p13QJoF7ShLP8sEBsSY1SNRgk4/view?usp=sharing), this is a newer version compare to the [old version](https://github.com/whuwuteng/benchmark_ISPRS2021), the origin image and LiDAR used are same with [old version](https://github.com/whuwuteng/benchmark_ISPRS2021).

An example is show here :

| <img src="/ISPRS-Vaihingen/10030062_10030063_0014_left.png" width="160"  height="160" alt="*Left image*" /> <img src="/ISPRS-Vaihingen/10030062_10030063_0014_show.png" width="160"  height="160" alt="*Left image*" /> <img src="/ISPRS-Vaihingen/10250130_10250131_0001_left.png" width="160"  height="160" alt="*Left image*" /> <img src="/ISPRS-Vaihingen/10250130_10250131_0001_show.png" width="160"  height="160" alt="*Left image*" /> <img src="/ISPRS-Vaihingen/colorbar.png" width="28"  height="160" alt="*Left image*" />|
| :----------------------------------------------------------: |
|                *Example for ISPRS-Vaihingen*                |

### EuroSDR-Vaihingen

This dataset is collected nearly same time with ISPRS-Vaihingen, the difference is that the resolution, the dataset can be found [here](https://ifpwww.ifp.uni-stuttgart.de/ISPRS-EuroSDR/ImageMatching/default.aspx), because the LiDAR is from **ISPRS-Vaihingen**, so only a small part of the data is used.

An example is show here :

| <img src="/EuroSDR-Vaihingen/07_29_0003_left.png" width="160"  height="160" alt="*Left image*" /> <img src="/EuroSDR-Vaihingen/07_29_0003_show.png" width="160"  height="160" alt="*Left image*" /> <img src="/EuroSDR-Vaihingen/20_31_0003_left.png" width="160"  height="160" alt="*Left image*" /> <img src="/EuroSDR-Vaihingen/20_31_0003_show.png" width="160"  height="160" alt="*Left image*" /> <img src="/EuroSDR-Vaihingen/colorbar.png" width="28"  height="160" alt="*Left image*" />|
| :----------------------------------------------------------: |
|                *Example for EuroSDR-Vaihingen*                |

 ### Toulouse-UMBRA

This dataset is collected by IGN(French map agency) in **2012**, and the camera is produced by IGN, the origin image is **16bit**, this dataset is for remote sensing use, to make the training data, we use auto just. And in the experiment, we found that the image is quite different from other dataset. 

| <img src="/figures/Toulouse_umbra.png" width="700" alt="*Origin Toulouse-UMBRA coverage*" /> |
| :----------------------------------------------------------: |
|                *Origin Toulouse-UMBRA coverage*                |

An example is show here :

| <img src="/ Toulouse-UMBRA/ESS301_fx00004_00212_ESS301_fx00005_00195_0007_left.png" width="160"  height="160" alt="*Left image*" /> <img src="/ Toulouse-UMBRA/ESS301_fx00004_00212_ESS301_fx00005_00195_0007_show.png" width="160"  height="160" alt="*Left image*" /> <img src="/ Toulouse-UMBRA/ESS301_fx00004_00212_ESS301_fx00005_00195_0042_left.png" width="160"  height="160" alt="*Left image*" /> <img src="/ Toulouse-UMBRA/ESS301_fx00004_00212_ESS301_fx00005_00195_0042_show.png" width="160"  height="160" alt="*Left image*" /> <img src="/ Toulouse-UMBRA/colorbar.png" width="28"  height="160" alt="*Left image*" />|
| :----------------------------------------------------------: |
|                *Example for Toulouse-UMBRA*                |

### Toulouse-Métropole

This dataset is collect by [AI4GEO](http://ai4geo.eu/index.php) in **2019**, the camera is **UltraCam Osprey Prime M3**, and the LiDAR is **ALS70**. The origin dataset is too large, only the area same with the **Toulous-UMBRA** is used in the paper for produce the dataset.

| <img src="/figures/Toulouse_metropole.png" width="700" alt="*Origin Toulouse-Métropole coverage*" /> |
| :----------------------------------------------------------: |
|                *Origin Toulouse-Métropole coverage*                |

An example is show here :

| <img src="/Toulouse-Metropole/21_14339_Lvl02-Color_21_14340_Lvl02-Color_0026_left.png" width="160"  height="160" alt="*Left image*" /> <img src="/Toulouse-Metropole/21_14339_Lvl02-Color_21_14340_Lvl02-Color_0026_show.png" width="160"  height="160" alt="*Left image*" /> <img src="/Toulouse-Metropole/24_14564_Lvl02-Color_24_14565_Lvl02-Color_0025_left.png" width="160"  height="160" alt="*Left image*" /> <img src="/Toulouse-Metropole/24_14564_Lvl02-Color_24_14565_Lvl02-Color_0025_show.png" width="160"  height="160" alt="*Left image*" /> <img src="/Toulouse-Metropole/colorbar.png" width="28"  height="160" alt="*Left image*" />|
| :----------------------------------------------------------: |
|                *Example for  Toulouse-Métropole*                |

### Enschede

This dataset is a dataset collected from [ITC Faculty Geo-Information Science and Earth Observation](https://www.itc.nl/education/study-finder/geo-information-science-earth-observation/) in **2011**, the LiDAR is **AHN2** in **2012**. The origin device has **5** cameras, only the nadir camera is used in the experiment.

An example is show here :

| <img src="/Enschede/0021775_1_0021776_1_0009_left.png" width="160"  height="160" alt="*Left image*" /> <img src="/Enschede/0021775_1_0021776_1_0009_show.png" width="160"  height="160" alt="*Left image*" /> <img src="/Enschede/0021777_1_0021778_1_0007_left.png" width="160"  height="160" alt="*Left image*" /> <img src="/Enschede/0021777_1_0021778_1_0007_show.png" width="160"  height="160" alt="*Left image*" /> <img src="/Enschede/colorbar.png" width="28"  height="160" alt="*Left image*" />|
| :----------------------------------------------------------: |
|                *Example for Enschede*                |

### DublinCity 

[DublinCity](https://v-sense.scss.tcd.ie/dublincity/) is an open dataset, the original aerial and LiDAR point cloud can be [downloaded](https://geo.nyu.edu/catalog/nyu-2451-38684), the origin dataset is very large.

You can find the training and testing dataset from [another paper](https://openaccess.thecvf.com/content/CVPR2023W/PCV/html/Wu_PSMNet-FusionX3_LiDAR-Guided_Deep_Learning_Stereo_Dense_Matching_on_Aerial_Images_CVPRW_2023_paper.html). To save the disk, we do not upload this time, more information can be found on [Github](https://github.com/whuwuteng/PSMNet-FusionX3) also.

An example is show here :

| <img src="/DublinCity/3489_DUBLIN_AREA_2KM2_rgb_124885_id278c1_20150326120951_3489_DUBLIN_AREA_2KM2_rgb_124888_id281c1_20150326120954_0005_left.png" width="160"  height="160" alt="*Left image*" /> <img src="/DublinCity/3489_DUBLIN_AREA_2KM2_rgb_124885_id278c1_20150326120951_3489_DUBLIN_AREA_2KM2_rgb_124888_id281c1_20150326120954_0005_show.png" width="160"  height="160" alt="*Left image*" /> <img src="/DublinCity/3489_DUBLIN_AREA_2KM2_rgb_124791_id184c1_20150326120501_3489_DUBLIN_AREA_2KM2_rgb_124792_id185c1_20150326120502_0009_left.png" width="160"  height="160" alt="*Left image*" /> <img src="/DublinCity/3489_DUBLIN_AREA_2KM2_rgb_124791_id184c1_20150326120501_3489_DUBLIN_AREA_2KM2_rgb_124792_id185c1_20150326120502_0009_show.png" width="160"  height="160" alt="*Left image*" /> <img src="/DublinCity/colorbar.png" width="28"  height="160" alt="*Left image*" />|
| :----------------------------------------------------------: |
|                *Example for Enschede*                |

##  Method

In the paper, we evaluate the state of the art methods of deep learning on stereo dense matching before 2020. 

### MicMac

MicMac can is a open source code,  the code can be found from [Github](https://github.com/micmacIGN/micmac). In the experiment, the command line is :

```
mm3d MM1P left.tif right.tif NONE DefCor=0.2 HasSBG=false HasVeg=true 
```

### SGM(GPU)

This method is revised during the experiment, because the origin disparity range is too small, i.e **128**, in our experiment, **256** is used. The code can be found in [folder](sgm_cuda_256).

### GraphCuts

This origin code can be found from [Github](https://github.com/t-taniai/LocalExpStereo), to make the code run in Linux, a new version can be found [here](https://github.com/whuwuteng/LocalExpStereo).

### CBMV

The code can be found from [Github](https://github.com/kbatsos/CBMV).

### MC-CNN

The code can be found from [Github](https://github.com/jzbontar/mc-cnn).

### DeepFeature

The code can be found [here](https://bitbucket.org/saakuraa/cvpr16_stereo_public/src/master/).

### PSM net

The origin code can be found from [Github](https://github.com/JiaRenChang/PSMNet).

### HRS net

The origin code can be found from [Github](https://github.com/gengshan-y/high-res-stereo).

### DeepPruner

The origin code can be found from [Github](https://github.com/uber-research/DeepPruner).

### GANet

The origin code can be found from [Github](https://github.com/feihuzhang/GANet).

### LEAStereo

The origin code can be found from [Github](https://github.com/XuelianCheng/LEAStereo).

##  Pretrained models

The pretrained models are important in the paper, so we will also share the pretrained models and training setting in the paper.

### CBMV

The pre-trained model for the **6** dataset are provide in CBMV_Model.zip :
|     Model Name      | training data | images |
| :----------: | :-----------: | :-----------: |
| CBMV_model_ISPRS-Vaihingen.rf | ISPRS-Vaihingen  | 200 |
| CBMV_model_EuroSDR-Vaihingen.rf | EuroSDR-Vaihingen | 200 |
| CBMV_model_Toulouse-UMBRA.rf | Toulouse-UMBRA | 200 |
| CBMV_model_Toulouse-Metropole.rf | Toulouse-Metropole | 200 |
| CBMV_model_Enschede.rf | Enschede | 200 |
| CBMV_model_DublinCity.rf | DublinCity | 200 |

### MC-CNN

The pre-trained model for the **6** dataset are provide in MC-CNN_Model.zip, and a model trained on all the image :

|     Model Name      | training data | images |
| :----------: | :-----------: | :-----------: |
| MC-CNN_model_ISPRS-Vaihingen.t7 | ISPRS-Vaihingen  | 1200 |
| MC-CNN_model_EuroSDR-Vaihingen.t7 | EuroSDR-Vaihingen | 1200 |
| MC-CNN_model_Toulouse-UMBRA.t7 | Toulouse-UMBRA | 1200 |
| MC-CNN_model_Toulouse-Metropole.t7 | Toulouse-Metropole | 1200 |
| MC-CNN_model_Enschede.t7 | Enschede | 1200 |
| MC-CNN_model_DublinCity.t7 | DublinCity | 1200 |
| MC-CNN_model_All.t7 | 6 dataset | 1200 |

### DeepFeature

The pre-trained model for the **6** dataset are provide in DeepFeature_Model.zip, and a model trained on all the image :

|     Model Name      | training data | images |
| :----------: | :-----------: | :-----------: |
| bn_meanvar_ISPRS-Vaihingen.t7 and param_ISPRS-Vaihingen.t7 | ISPRS-Vaihingen  | 1200 |
| bn_meanvar_EuroSDR-Vaihingen.t7 and param_EuroSDR-Vaihingen.t7 | EuroSDR-Vaihingen  | images1200 |
| bn_meanvar_Toulouse-UMBRA.t7 and param_Toulouse-UMBRA.t7 | Toulouse-UMBRA | 1200 |
| bn_meanvar_Toulouse-Metropole.t7 and param_Toulouse-Metropole.t7 | Toulouse-Metropole | 1200 |
| bn_meanvar_Enschede.t7 and param_Enschede.t7 | Enschede | 1200 |
| bn_meanvar_DublinCity.t7 and param_DublinCity.t7 | DublinCity | 1200 |
| bn_meanvar_all.t7 and param_all.t7 | 6 dataset | 1200 |

### PSM net

The pre-trained model for the **6** dataset are provide in PSMNet_Model.zip, and a model trained on all the image :
|     Model Name      | training data | images |
| :----------: | :-----------: | :-----------: |
| PSMNet_Model_ISPRS-Vaihingen.tar | ISPRS-Vaihingen  | 1200 |
| PSMNet_Model_EuroSDR-Vaihingen.tar | EuroSDR-Vaihingen | 1200 |
| PSMNet_Model_Toulouse-UMBRA.tar | Toulouse-UMBRA | 1200 |
| PSMNet_Model_Toulouse-Metropole.tar | Toulouse-Metropole | 1200 |
| PSMNet_Model_Enschede.tar | Enschede | 1200 |
| PSMNet_Model_DublinCity.tar | DublinCity | 1200 |
| PSMNet_Model_All.tar | 6 dataset | 1200 |

### HRS net

The pre-trained model for the **6** dataset are provide in HRSNet_Model.zip, and a model trained on all the image :
|     Model Name      | training data | images |
| :----------: | :-----------: | :-----------: |
| HRSNet_Model_ISPRS-Vaihingen.tar | ISPRS-Vaihingen  | 1200 |
| HRSNet_Model_EuroSDR-Vaihingen.tar | EuroSDR-Vaihingen | 1200 |
| HRSNet_Model_Toulouse-UMBRA.tar | Toulouse-UMBRA | 1200 |
| HRSNet_Model_Toulouse-Metropole.tar | Toulouse-Metropole | 1200 |
| HRSNet_Model_Enschede.tar | Enschede | 1200 |
| HRSNet_Model_DublinCity.tar | DublinCity | 1200 |
| HRSNet_Model_All.tar | 6 dataset | 1200 |

### DeepPruner

The pre-trained model for the **6** dataset are provide in DeepPruner_Model.zip, and a model trained on all the image :
|     Model Name      | training data | images |
| :----------: | :-----------: | :-----------: |
| DeepPruner_model_ISPRS-Vaihingen.tar | ISPRS-Vaihingen  | 1200 |
| DeepPruner_model_EuroSDR-Vaihingen.tar | EuroSDR-Vaihingen | 1200 |
| DeepPruner_model_Toulouse-UMBRA.tar | Toulouse-UMBRA | 1200 |
| DeepPruner_model_Toulouse-Metropole.tar | Toulouse-Metropole | 1200 |
| DeepPruner_model_Enschede.tar | Enschede | 1200 |
| DeepPruner_model_DublinCity.tar | DublinCity | 1200 |
| DeepPruner_model_All.tar | 6 dataset | 1200 |

### GANet

The pre-trained model for the **6** dataset are provide in GANet_Model.zip, and a model trained on all the image :
|     Model Name      | training data | images |
| :----------: | :-----------: | :-----------: |
| GANet_model_ISPRS-Vaihingen.pth | ISPRS-Vaihingen  | 1200 |
| GANet_model_EuroSDR-Vaihingen.pth | EuroSDR-Vaihingen | 1200 |
| GANet_model_Toulouse-UMBRA.pth | Toulouse-UMBRA | 1200 |
| GANet_model_Toulouse-Metropole.pth | Toulouse-Metropole | 1200 |
| GANet_model_Enschede.pth | Enschede | 1200 |
| GANet_model_DublinCity.pth | DublinCity | 1200 |
| GANet_model_All.pth | 6 dataset | 1200 |

### LEAStereo

The pre-trained model for the **6** dataset are provide in LEAStereo_Model.zip, and a model trained on all the image :
|     Model Name      | training data | images |
| :----------: | :-----------: | :-----------: |
| LEAStereo_model_ISPRS-Vaihingen.pth | ISPRS-Vaihingen  | 1200 |
| LEAStereo_model_EuroSDR-Vaihingen.pth | EuroSDR-Vaihingen | 1200 |
| LEAStereo_model_Toulouse-UMBRA.pth | Toulouse-UMBRA | 1200 |
| LEAStereo_model_Toulouse-Metropole.pth | Toulouse-Metropole | 1200 |
| LEAStereo_model_Enschede.pth | Enschede | 1200 |
| LEAStereo_model_DublinCity.pth | DublinCity | 1200 |
| LEAStereo_model_All.pth | 6 dataset | 1200 |

## TODO

- [x] Image-LiDAR process
- [x] Publish dataset V1 (use in the paper)
- [x] Publish the long paper on Arxiv
- [x] Publish pretrained models
- [ ] Publish full dataset (we don't have the host, the full dataset can be provided after required)

## Stereo-LiDAR fusion

Based on the data generation,  we also generate the Toulouse2020 data from IGN, and this data can be found in our CVPR photogrammetry and computer vision workshop paper. The Github site can be found [here](https://github.com/whuwuteng/PSMNet-FusionX3).

### Citation

If you think you have any problem, contact [Teng Wu]<whuwuteng@gmail.com>

