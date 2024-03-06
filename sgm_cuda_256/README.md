# Semi-Global Matching on the GPU in the paper

## Introduction

A GPU base SGM, use as a base-line method.<mark>The orginal code is clone from [origin code](https://github.com/dhernandez0/sgm).</mark> Because for the origin code, the disparity range is **128**, and this is too small for the application, so here we will provide the code with a disparity range is **256**.

If you want to use the origin code setting (depends on OpenCV), we will try provide it, the code can be found in [folder WithOpenCV](/WithOpenCV).

## Update code for Cluster(CNES)

Because there is no [OpenCV](https://opencv.org/) in the Cluster, and I do not want to install OpenCV (too big, and no sudo), so this code is modified with OpenCV files, so that the code doesn't depend on OpenCV.

On the other hand, because the compared methods have sub-pixel accuracy, the output file is **16bit, i.e.  unsigned short**.

### Dependency library
1. remove the [OpenCV](https://opencv.org/) library, so all the needed OpenCV files are in [folder](opencv).
2. use [png++](https://www.nongnu.org/pngpp/), this is used to read and write file, so that this code only support **PNG** files, the input and output are both **PNG** file.

### Disparity range
In the code, the **PATH_AGGREGATION** is 8 for defualt,  and the **MAX_DISPARITY** is 256 now. 

From the code, the **WARP_SIZE** is 32(fixed). So for 32 thread, run one piexl, so that the batch size is **MAX_DISPARITY**/**WARP_SIZE**.
Considering that if the **MAX_DISPARITY** is 256, the batch size is 8. Shortage is that batch size is 4 is more clear.
The **Algorithm 3** is the main idea for the code, the parameters are shared in the 32 thread. 

## Compile

To compile the CUDA code, the [compatibility is important](https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version), you can set the compile gcc/g++ before make, for my computer, the version is **Cuda compilation tools, release 11.5, V11.5.119**, so we can run like this :

```
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

mkdir build
cd build
cmake ..
make
```

## Exmaple

we give an example in the [folder](example) for the Toulouse Métropole dataset.

```
./example.sh
```

## TODO
- [ ] check the label **recompute**

**recompute** means calculating the cost again, now **recompute** is not used in **DIR_LEFTRIGHT** and **DIR_RIGHTLEFT**.
If so, the code can be much cleaner. (T *rp0, T *rp1, T *rp2, T *rp3, T *rp4, T *rp5, T *rp6, T *rp7) can be removed.
The iterations in **DIR_LEFTRIGHT** and **DIR_RIGHTLEFT** can be removed.

- [ ] make the code clean

- [ ] sub-pixel accuracy

- [ ] left-right check

## Feed Back
If you think you have any problem, contact [Teng Wu]<whuwuteng@gmail.com>

