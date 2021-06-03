#include "stdafx.h"
#pragma hdrstop
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

#include "data/Util.h"

namespace util{

	//! �͂ݏo��̈���J�b�g
	cv::Rect TruncateRect(const cv::Rect& obj_rect, const cv::Size& img_size)
	{
		cv::Rect resize_rect = obj_rect;
		if (obj_rect.x < 0){
			resize_rect.x = 0;
			resize_rect.width += obj_rect.x;
		}
		if (obj_rect.y < 0){
			resize_rect.y = 0;
			resize_rect.height += obj_rect.y;
		}
		if (resize_rect.x + resize_rect.width > img_size.width){
			resize_rect.width = img_size.width - resize_rect.x;
		}
		if (resize_rect.y + resize_rect.height > img_size.height){
			resize_rect.height = img_size.height - resize_rect.y;
		}

		return resize_rect;
	}


	//! ���S�𓮂������ɁA�͂ݏo��̈���J�b�g
	cv::Rect TruncateRectKeepCenter(const cv::Rect& obj_rect, const cv::Size& max_size)
	{
		cv::Rect exp_rect = obj_rect;
		if (exp_rect.x < 0){
			exp_rect.width += 2 * exp_rect.x;
			exp_rect.x = 0;
		}
		if (exp_rect.y < 0){
			exp_rect.height += 2 * exp_rect.y;
			exp_rect.y = 0;
		}
		if (exp_rect.x + exp_rect.width > max_size.width){
			exp_rect.x += (exp_rect.x + exp_rect.width - max_size.width) / 2;
			exp_rect.width = max_size.width - exp_rect.x;
		}
		if (exp_rect.y + exp_rect.height > max_size.height){
			exp_rect.y += (exp_rect.y + exp_rect.height - max_size.height) / 2;
			exp_rect.height = max_size.height - exp_rect.y;
		}
		return exp_rect;
	}



}
