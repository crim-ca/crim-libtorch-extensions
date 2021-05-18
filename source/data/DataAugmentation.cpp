/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//
// Copyright (C) 2014 Takuya MINAGAWA.
// Third party copyrights are property of their respective owners.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//M*/

#include <iostream>

#include <opencv2/highgui/highgui.hpp>

#include "data/DataAugmentation.h"
#include "data/RandomRotation.h"
#include "data/Util.h"


cv::Rect RandomDeformRect(
	const cv::Rect& input_rect,
	double x_slide_sigma,
	double y_slide_sigma,
	double aspect_range,
	cv::RNG& rng
) {
	double x_mv_r = rng.gaussian(x_slide_sigma);
	double y_mv_r = rng.gaussian(y_slide_sigma);
	double aspect_change = rng.gaussian(aspect_range);

	cv::Rect dst_rect;
	double deform = aspect_change / (2.0 + aspect_change);
	dst_rect.width = input_rect.width * (1.0 + deform);
	dst_rect.height = input_rect.height * (1.0 - deform);
	dst_rect.x = input_rect.x + (input_rect.width - dst_rect.width) / 2;
	dst_rect.y = input_rect.y + (input_rect.height - dst_rect.height) / 2;
	dst_rect.x += x_mv_r * dst_rect.width;
	dst_rect.y += y_mv_r * dst_rect.height;

	return dst_rect;
}


cv::Mat ImageTransform(
	const cv::Mat& img,
	int64_t image_size,
	double yaw_sigma,
	double pitch_sigma,
	double roll_sigma,
	double blur_max_sigma,
	double noise_max_sigma,
	double x_slide_sigma,
	double y_slide_sigma,
	double aspect_range,
	double hflip_ratio,
	double vflip_ratio,
	cv::RNG& rng
) {
	assert(img.type() == CV_8UC1 || img.type() == CV_8UC3);

	// Deform Rect Randomly
	cv::Rect area(img.cols / 2 - image_size / 2, img.rows / 2 - image_size / 2, image_size, image_size);

	cv::Rect rect = RandomDeformRect(area, x_slide_sigma, y_slide_sigma, aspect_range, rng);

	rect = util::TruncateRect(rect, img.size());

	// Random Rotation
	cv::Mat dst;
	RandomRotateImage(img, dst, yaw_sigma, pitch_sigma, roll_sigma, rect, rng);

	// Random Noise
	double noise_sigma = rng.uniform(0.0, noise_max_sigma);
	if (noise_sigma > 0){
		cv::Mat gauss_noise(dst.size(), CV_32FC(dst.channels()));
		cv::randn(gauss_noise, 0.0, noise_sigma);
		int num = dst.cols * dst.rows * dst.channels();
		unsigned char* dst_ptr = dst.data;
		float* noise_ptr = (float*)gauss_noise.data;
		for (int i = 0; i < num; i++){
			int val = *dst_ptr + *noise_ptr;
			*dst_ptr = (val > 255) ? 255 : (val < 0) ? 0 : val;
			dst_ptr++, noise_ptr++;
		}
	}

	// Random Blur
	cv::Mat dst2;
	double blur_sigma = rng.uniform(0.0, blur_max_sigma);
	int size = blur_sigma * 2.5 + 0.5;
	size += (1 - size % 2);
	if (blur_sigma > 0 && size >= 3){
		cv::Size ksize(size, size);
		cv::GaussianBlur(dst, dst2, ksize, blur_sigma);
	}
	else{
		dst2 = dst;
	}

	// Rondom Flip (horizontal)
	cv::Mat dst3;
	double flip_prob = rng.uniform(0.0, 1.0);
	if (hflip_ratio > flip_prob) {
		cv::flip(dst2, dst3, 1);
	}
	else {
		dst3 = dst2;
	}

	// Rondom Flip (vertical)
	cv::Mat dst4;
	flip_prob = rng.uniform(0.0, 1.0);
	if (vflip_ratio > flip_prob) {
		cv::flip(dst3, dst4, 0);
	}
	else {
		dst4 = dst3;
	}
	return dst4;
}

