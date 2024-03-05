/**
    This file is part of sgm. (https://github.com/dhernandez0/sgm).

    Copyright (c) 2016 Daniel Hernandez Juarez.

    sgm is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    sgm is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with sgm.  If not, see <http://www.gnu.org/licenses/>.

**/

#include <stdio.h>
#include <iostream>
#include <numeric>
#include <sys/time.h>
#include <vector>
#include <stdlib.h>
#include <typeinfo>

//#include <opencv2/opencv.hpp>
#include "opencv/mat.hpp"
#include "opencv/cvtColor.hpp"
#include "opencv/types.hpp"
#include "png++/image.hpp"

#include <numeric>
#include <stdlib.h>
#include <ctime>
#include <sys/types.h>
#include <stdint.h>
#include <linux/limits.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include "disparity_method.h"

//typedef Vec<unsigned char, 3> Vec3b;
Mat_<unsigned char, 3> readPNG_RGB_Header(const char * pszName)
{
	png::image< png::rgb_pixel > image(pszName);
	int nCols = image.get_width();
	int nRows = image.get_height();

	Mat_<unsigned char, 3> cvImg = Mat_<unsigned char, 3>(nRows, nCols, Scalar(0, 0, 0));

	unsigned char * pImg = (unsigned char * )cvImg.ptr();
	for (int i = 0; i < nRows; i++) {
		for (int j = 0; j < nCols; j++) {
			pImg[(i * nCols + j) * 3 + 0] = image.get_pixel(j, i).red;
			pImg[(i * nCols + j) * 3 + 1] = image.get_pixel(j, i).green;
			pImg[(i * nCols + j) * 3 + 2] = image.get_pixel(j, i).blue;
		}
	}
	return cvImg;
}

int writePNG(const char * pszName, Mat_<unsigned char, 1> & cvImg)
{
	int height = cvImg.rows;
	int width = cvImg.cols;

	png::image<png::gray_pixel> image(width, height);
	unsigned char * pImg = (unsigned char * )cvImg.ptr();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			unsigned char val = (unsigned char)(pImg[i * width + j]);
			image.set_pixel(j, i, val);
		}
	}
	image.write(pszName);
	return 0;
}

int writePNG16(const char * pszName, Mat_<unsigned char, 1> & cvImg, int scale  = 256)
{
	int height = cvImg.rows;
	int width = cvImg.cols;

	png::image<png::gray_pixel_16> image(width, height);
	unsigned char * pImg = (unsigned char * )cvImg.ptr();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			unsigned short val = (unsigned short)(pImg[i * width + j] * scale);
			image.set_pixel(j, i, val);
		}
	}
	image.write(pszName);
	return 0;
}

bool directory_exists(const char* dir) {
	DIR* d = opendir(dir);
	bool ok = false;
	if(d) {
	    closedir(d);
	    ok = true;
	}
	return ok;
}

bool check_directories_exist(const char* directory, const char* left_dir, const char* right_dir, const char* disparity_dir) {
	char left_dir_sub[PATH_MAX];
	char right_dir_sub[PATH_MAX];
	char disparity_dir_sub[PATH_MAX];
	sprintf(left_dir_sub, "%s/%s", directory, left_dir);
	sprintf(right_dir_sub, "%s/%s", directory, right_dir);
	sprintf(disparity_dir_sub, "%s/%s", directory, disparity_dir);

	return directory_exists(left_dir_sub) && directory_exists(right_dir_sub) && directory_exists(disparity_dir_sub);
}

int main(int argc, char *argv[]) {
	if(argc < 5) {
		std::cerr << "Usage: cuda_sgm dir p1 p2 scale" << std::endl;
		return -1;
	}
	/*if(MAX_DISPARITY != 128) {
		std::cerr << "Due to implementation limitations MAX_DISPARITY must be 128" << std::endl;
		return -1;
	}*/
	if(PATH_AGGREGATION != 4 && PATH_AGGREGATION != 8) {
        std::cerr << "Due to implementation limitations PATH_AGGREGATION must be 4 or 8" << std::endl;
        return -1;
    }
	const char* directory = argv[1];
	uint8_t p1, p2;
	p1 = atoi(argv[2]);
	p2 = atoi(argv[3]);
	
	int scale = atoi(argv[4]);

	DIR *dp;
	struct dirent *ep;

	// Directories
	const char* left_dir = "colored_0";
	const char* disparity_dir = "disparities";
	const char* right_dir = "colored_1";
	const char* gt_dir = "gt";

	if(!check_directories_exist(directory, left_dir, right_dir, disparity_dir)) {
		std::cerr << "We need <left>, <right> and <disparities> directories" << std::endl;
		exit(EXIT_FAILURE);
	}
	char abs_left_dir[PATH_MAX];
    sprintf(abs_left_dir, "%s/%s", directory, left_dir);
	dp = opendir(abs_left_dir);
	if (dp == NULL) {
		std::cerr << "Invalid directory: " << abs_left_dir << std::endl;
		exit(EXIT_FAILURE);
	}
	char left_file[PATH_MAX];
	char right_file[PATH_MAX];
	char dis_file[PATH_MAX];
	char gt_file[PATH_MAX];
	char gt_dir_sub[PATH_MAX];

	sprintf(gt_dir_sub, "%s/%s", directory, gt_dir);
	const bool has_gt = directory_exists(gt_dir_sub);
	int n = 0;
	int n_err = 0;
	std::vector<float> times;

	init_disparity_method(p1, p2);
	while ((ep = readdir(dp)) != NULL) {
		// Skip directories
		if (!strcmp (ep->d_name, "."))
			continue;
		if (!strcmp (ep->d_name, ".."))
			continue;

		sprintf(left_file, "%s/%s/%s", directory, left_dir, ep->d_name);
		sprintf(right_file, "%s/%s/%s", directory, right_dir, ep->d_name);
		sprintf(dis_file, "%s/%s/%s", directory, disparity_dir, ep->d_name);
		sprintf(gt_file, "%s/%s/%s", directory, gt_dir, ep->d_name);
		int gt_len = strlen(gt_file);

		Mat_<unsigned char, 3> h_im0 = readPNG_RGB_Header(left_file);
		if(!h_im0.data) {
			std::cerr << "Couldn't read the file " << left_file << std::endl;
			return EXIT_FAILURE;
		}
		Mat_<unsigned char, 3> h_im1 = readPNG_RGB_Header(right_file);
		if(!h_im1.data) {
			std::cerr << "Couldn't read the file " << right_file << std::endl;
			return EXIT_FAILURE;
		}

		// Convert images to grayscale
		Mat_<unsigned char, 1> gray_im0(h_im0.rows, h_im0.cols, Scalar(0));
		if (h_im0.channels > 1) {
			cvtColor(h_im0, gray_im0, CV_RGB2GRAY);
		}
		
		Mat_<unsigned char, 1> gray_im1(h_im1.rows, h_im1.cols, Scalar(0));
		if (h_im1.channels > 1) {
			cvtColor(h_im1, gray_im1, CV_RGB2GRAY);
		}
		
		if(h_im0.rows != h_im1.rows || h_im0.cols != h_im1.cols) {
			std::cerr << "Both images must have the same dimensions" << std::endl;
			return EXIT_FAILURE;
		}
		// result
		Mat_<unsigned char, 1> disparity_im;
		if(h_im0.rows % 4 != 0 || h_im0.cols % 4 != 0) {
			std::cerr << "Due to implementation limitations image width and height must be a divisible by 4" << std::endl;
			// return EXIT_FAILURE;
			// crop image
            int tar_rows = h_im0.rows;
            if (h_im0.rows % 4 != 0){
            	tar_rows = (h_im0.rows/4 + 1) * 4;
            }
            int tar_cols = h_im0.cols;
            if (h_im0.cols % 4 != 0){
            	tar_cols = (h_im0.cols/4 + 1) * 4;          
            }

			//std::cout << tar_rows << std::endl;
			//std::cout << tar_cols << std::endl;

            Mat_<unsigned char, 1> fill_left(tar_rows, tar_cols, Scalar(0));
			Mat_<unsigned char, 1> fill_right(tar_rows, tar_cols, Scalar(0));

			//std::cout << fill_left.rows << std::endl;
			//std::cout << fill_left.cols << std::endl;

			gray_im0.copyTo(fill_left, Rect_<int>(0, 0, h_im0.cols, h_im0.rows));
			gray_im1.copyTo(fill_right, Rect_<int>(0, 0, h_im0.cols, h_im0.rows));

			float elapsed_time_ms;
			Mat_<unsigned char, 1> disparity_tar = compute_disparity_method(fill_left, fill_right, &elapsed_time_ms, directory, ep->d_name);
#if LOG
			std::cout << "done" << std::endl;
#endif
			times.push_back(elapsed_time_ms);

			disparity_tar.copyTo(disparity_im, Rect_<int>(0, 0, h_im0.cols, h_im0.rows));
		}
		else{
#if LOG
			std::cout << "processing: " << left_file << std::endl;
#endif
			// Compute
			float elapsed_time_ms;
			//writePNG("left.png", gray_im0);
			//writePNG("right.png", gray_im1);
			disparity_im = compute_disparity_method(gray_im0, gray_im1, &elapsed_time_ms, directory, ep->d_name);

#if LOG
			std::cout << "done" << std::endl;
#endif
			times.push_back(elapsed_time_ms);
		}
#if WRITE_FILES
	writePNG16(dis_file, disparity_im, scale);
	//writePNG(dis_file, disparity_im);
#endif
	}
	closedir(dp);
	finish_disparity_method();

	double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
	if(has_gt) {
		printf("%f\n", (float) n_err/n);
	} else {
		std::cout << "It took an average of " << mean << " miliseconds, " << 1000.0f/mean << " fps" << std::endl;
	}

	return 0;
}
