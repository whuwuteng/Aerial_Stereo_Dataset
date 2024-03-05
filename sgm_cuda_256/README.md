# Semi-Global Matching on the GPU in the paper

## Introduction

A GPU base SGM, use as a base-line method.<mark>The orginal code is clone from [origin code](https://github.com/dhernandez0/sgm).</mark> Because for the origin code, the disparity range is **128**, and this is too small for the application, so here we will provide the code with a disparity range is **256**.

## Update code for Cluster(CNES)

Because there is no OpenCV in the Cluster, and I do not want to install OpenCV(too big), so this code is modified with OpenCV files, so that the code doesn't depend on OpenCV.

### Dependency library
1. remove the [OpenCV](https://opencv.org/) library.
2. use [png++](https://www.nongnu.org/pngpp/).

### Disparity range
In the code, the **PATH_AGGREGATION** is 8 for defualt.
From the vesion **1.0** to now, the **MAX_DISPARITY** is 256 now. 

From the code, the **WARP_SIZE** is 32(fixed). So for 32 thread, run one piexl, so that the batch size is **MAX_DISPARITY**/**WARP_SIZE**.
Considering that if the **MAX_DISPARITY** is 256, the batch size is 8. Shortage is that batch size is 4 is more clear.
The **Algorithm 3** is the main idea for the code, the parameters are shared in the 32 thread. 

## TODO
- [ ] check the label **recompute**

**recompute** means calculating the cost again, now **recompute** is not used in **DIR_LEFTRIGHT** and **DIR_RIGHTLEFT**.
If so, the code can be much cleaner. (T *rp0, T *rp1, T *rp2, T *rp3, T *rp4, T *rp5, T *rp6, T *rp7) can be removed.
The iterations in **DIR_LEFTRIGHT** and **DIR_RIGHTLEFT** can be removed.

- [ ] make the code clean

- [ ] left-right check

## Feed Back
If you think you have any problem, contact [Teng Wu]<Teng.Wu@ign.fr>

