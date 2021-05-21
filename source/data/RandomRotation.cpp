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

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "data/RandomRotation.h"
#include "data/Util.h"


//!���[�^�s�b�`�^���[���ƕ��s�ړ���������A�J�����O���s��i��]�{���i�j�����߂�
/*!
\param[in] yaw ���[
\param[in] pitch �s�b�`
\param[in] roll ���[��
\param[in] trans_x ���i�ړ�X����
\param[in] trans_y ���i�ړ�Y����
\param[in] trans_z ���i�ړ�Z����
\param[out] external_matrix �O���s��
*/
void composeExternalMatrix(float yaw, float pitch, float roll, float trans_x, float trans_y, float trans_z, cv::Mat& external_matrix)
{
	external_matrix.release();
	external_matrix.create(3, 4, CV_64FC1);

	double sin_yaw = sin((double)yaw * CV_PI / 180);
	double cos_yaw = cos((double)yaw * CV_PI / 180);
	double sin_pitch = sin((double)pitch * CV_PI / 180);
	double cos_pitch = cos((double)pitch * CV_PI / 180);
	double sin_roll = sin((double)roll * CV_PI / 180);
	double cos_roll = cos((double)roll * CV_PI / 180);

	external_matrix.at<double>(0, 0) = cos_pitch * cos_yaw;
	external_matrix.at<double>(0, 1) = -cos_pitch * sin_yaw;
	external_matrix.at<double>(0, 2) = sin_pitch;
	external_matrix.at<double>(1, 0) = cos_roll * sin_yaw + sin_roll * sin_pitch * cos_yaw;
	external_matrix.at<double>(1, 1) = cos_roll * cos_yaw - sin_roll * sin_pitch * sin_yaw;
	external_matrix.at<double>(1, 2) = -sin_roll * cos_pitch;
	external_matrix.at<double>(2, 0) = sin_roll * sin_yaw - cos_roll * sin_pitch * cos_yaw;
	external_matrix.at<double>(2, 1) = sin_roll * cos_yaw + cos_roll * sin_pitch * sin_yaw;
	external_matrix.at<double>(2, 2) = cos_roll * cos_pitch;

	external_matrix.at<double>(0, 3) = trans_x;
	external_matrix.at<double>(1, 3) = trans_y;
	external_matrix.at<double>(2, 3) = trans_z;
}


//! ��`�̎l���̍��W�����ꂼ��Ď����W�n�֕ϊ�
cv::Mat Rect2Mat(const cv::Rect& img_rect)
{
	// �摜�v���[�g�̎l���̍��W
	cv::Mat srcCoord(3, 4, CV_64FC1);
	srcCoord.at<double>(0, 0) = img_rect.x;
	srcCoord.at<double>(1, 0) = img_rect.y;
	srcCoord.at<double>(2, 0) = 1;
	srcCoord.at<double>(0, 1) = img_rect.x + img_rect.width;
	srcCoord.at<double>(1, 1) = img_rect.y;
	srcCoord.at<double>(2, 1) = 1;
	srcCoord.at<double>(0, 2) = img_rect.x + img_rect.width;
	srcCoord.at<double>(1, 2) = img_rect.y + img_rect.height;
	srcCoord.at<double>(2, 2) = 1;
	srcCoord.at<double>(0, 3) = img_rect.x;
	srcCoord.at<double>(1, 3) = img_rect.y + img_rect.height;
	srcCoord.at<double>(2, 3) = 1;

	return srcCoord;
}


//! ���͉摜�̎l����transM�ɉ����ē����ϊ����A�o�͉摜�̊O�ڒ����`�����߂�
/*!
\param[in] img_size ���͉摜�T�C�Y
\param[in] transM 3x3�̓����ϊ��s��(CV_64FC1)
\param[out] CircumRect �o�͉摜�̊O�ڒ����`
*/
void CircumTransImgRect(const cv::Size& img_size, const cv::Mat& transM, cv::Rect_<double>& CircumRect)
{
	// ���͉摜�̎l����Ď����W�֕ϊ�
	cv::Mat cornersMat = Rect2Mat(cv::Rect(0, 0, img_size.width, img_size.height));

	// ���W�ϊ����A�͈͂��擾
	cv::Mat dstCoord = transM * cornersMat;
	double min_x = std::min(dstCoord.at<double>(0, 0) / dstCoord.at<double>(2, 0), dstCoord.at<double>(0, 3) / dstCoord.at<double>(2, 3));
	double max_x = std::max(dstCoord.at<double>(0, 1) / dstCoord.at<double>(2, 1), dstCoord.at<double>(0, 2) / dstCoord.at<double>(2, 2));
	double min_y = std::min(dstCoord.at<double>(1, 0) / dstCoord.at<double>(2, 0), dstCoord.at<double>(1, 1) / dstCoord.at<double>(2, 1));
	double max_y = std::max(dstCoord.at<double>(1, 2) / dstCoord.at<double>(2, 2), dstCoord.at<double>(1, 3) / dstCoord.at<double>(2, 3));

	CircumRect.x = min_x;
	CircumRect.y = min_y;
	CircumRect.width = max_x - min_x;
	CircumRect.height = max_y - min_y;
}



//! ���͉摜�Əo�͉摜�̍��W�̑Ή��֌W���v�Z
/*!
\param[in] src_size ���͉摜�T�C�Y
\param[in] dst_rect ���͉摜�𓧎��ϊ��������̏o�͉摜�̊O�ڒ����`
\param[in] transMat 4x4�̉�]/���s�ړ��s��(CV_64FC1)�B���_�ŉ�]�����āAZ�������ɕ��s�ړ���������
\param[out] map_x �o�͉摜�̊e���W�ɑ΂�����͉摜��x���W
\param[out] map_y �o�͉摜�̊e���W�ɑ΂�����͉摜��y���W

transMat�͓��͉摜���R�����I�ɉ�]���A���̒��S��(0,0,Z)�ɒu���悤�ɕϊ�����s��B
�o�͉摜�͏œ_������1�̃J������z�肵�A���͉摜�������ɓ����ϊ�����B
�������A�X�P�[�������킹�邽�߂ɏo�͉摜��X,Y���W��1/Z����B
�o�͉摜��̍��W��(dx, dy)�ŗ^����ꂽ���A���_�Ƃ��̓_�����Ԓ�����(dx*r, dy*r, Z*r)�ŕ\�����B
���͉摜��̍��W(sx,sy)���R�������W�ŕ\����transMat*(sx, sy, 0, 1)^T �ƂȂ�̂ŁA(sx, sy)��(dx, dy)�̊֌W��
(sx, sy, 0, 1)^T = transMat^(-1) * (dx*r, dy*r, Z*r)
�ƂȂ�B
��������Ar���������Ƃ�dx��dy�ɑΉ�����sx��sy�����܂�B
*/
void CreateMap(const cv::Size& src_size, const cv::Rect_<double>& dst_rect, const cv::Mat& transMat, cv::Mat& map_x, cv::Mat& map_y)
{
	map_x.create(dst_rect.size(), CV_32FC1);
	map_y.create(dst_rect.size(), CV_32FC1);

	double Z = transMat.at<double>(2, 3);

	cv::Mat invTransMat = transMat.inv();	// �t�s��
	cv::Mat dst_pos(3, 1, CV_64FC1);	// �o�͉摜��̍��W
	dst_pos.at<double>(2, 0) = Z;
	for (int dy = 0; dy<map_x.rows; dy++){
		dst_pos.at<double>(1, 0) = dst_rect.y + dy;
		for (int dx = 0; dx<map_x.cols; dx++){
			dst_pos.at<double>(0, 0) = dst_rect.x + dx;
			cv::Mat rMat = -invTransMat(cv::Rect(3, 2, 1, 1)) / (invTransMat(cv::Rect(0, 2, 3, 1)) * dst_pos);
			cv::Mat src_pos = invTransMat(cv::Rect(0, 0, 3, 2)) * dst_pos * rMat + invTransMat(cv::Rect(3, 0, 1, 2));
			map_x.at<float>(dy, dx) = src_pos.at<double>(0, 0) + (float)src_size.width / 2;
			map_y.at<float>(dy, dx) = src_pos.at<double>(1, 0) + (float)src_size.height / 2;
		}
	}
}


void RotateImage(const cv::Mat& src, cv::Mat& dst, float yaw, float pitch, float roll,
	float Z = 1000, int interpolation = cv::INTER_LINEAR, int boarder_mode = cv::BORDER_CONSTANT, const cv::Scalar& border_color = cv::Scalar(0, 0, 0))
{
	// rotation matrix
	cv::Mat rotMat_3x4;
	composeExternalMatrix(yaw, pitch, roll, 0, 0, Z, rotMat_3x4);

	cv::Mat rotMat = cv::Mat::eye(4, 4, rotMat_3x4.type());
	rotMat_3x4.copyTo(rotMat(cv::Rect(0, 0, 4, 3)));

	// From 2D coordinates to 3D coordinates
	// The center of image is (0,0,0)
	cv::Mat invPerspMat = cv::Mat::zeros(4, 3, CV_64FC1);
	invPerspMat.at<double>(0, 0) = 1;
	invPerspMat.at<double>(1, 1) = 1;
	invPerspMat.at<double>(3, 2) = 1;
	invPerspMat.at<double>(0, 2) = -(double)src.cols / 2;
	invPerspMat.at<double>(1, 2) = -(double)src.rows / 2;

	// �R�������W����Q�������W�֓����ϊ�
	cv::Mat perspMat = cv::Mat::zeros(3, 4, CV_64FC1);
	perspMat.at<double>(0, 0) = Z;
	perspMat.at<double>(1, 1) = Z;
	perspMat.at<double>(2, 2) = 1;

	// ���W�ϊ����A�o�͉摜�̍��W�͈͂��擾
	cv::Mat transMat = perspMat * rotMat * invPerspMat;
	cv::Rect_<double> CircumRect;
	CircumTransImgRect(src.size(), transMat, CircumRect);

	// �o�͉摜�Ɠ��͉摜�̑Ή��}�b�v���쐬
	cv::Mat map_x, map_y;
	CreateMap(src.size(), CircumRect, rotMat, map_x, map_y);
	cv::remap(src, dst, map_x, map_y, interpolation, boarder_mode, border_color);
}


// Keep center and expand rectangle for rotation
cv::Rect ExpandRectForRotate(const cv::Rect& area)
{
	cv::Rect exp_rect;

	int w = cvRound(std::sqrt((double)(area.width * area.width + area.height * area.height)));

	exp_rect.width = w;
	exp_rect.height = w;
	exp_rect.x = area.x - (exp_rect.width - area.width) / 2;
	exp_rect.y = area.y - (exp_rect.height - area.height) / 2;

	return exp_rect;
}


void RandomRotateImage(
	const cv::Mat& src,
	/*[out]*/cv::Mat& dst,
	float yaw_sigma,
	float pitch_sigma,
	float roll_sigma,
	cv::RNG& rng,
	const cv::Rect& area,
	float Z,
	int interpolation,
	int boarder_mode,
	const cv::Scalar& boarder_color
) {
	double yaw = rng.gaussian(yaw_sigma);
	double pitch = rng.gaussian(pitch_sigma);
	double roll = rng.gaussian(roll_sigma);
	//double yaw = rng.uniform(-yaw_range / 2, yaw_range / 2);
	//double pitch = rng.uniform(-pitch_range / 2, pitch_range / 2);
	//double roll = rng.uniform(-roll_range / 2, roll_range / 2);

	cv::Rect rect = (area.width <= 0 || area.height <= 0) ? cv::Rect(0, 0, src.cols, src.rows) :
		ExpandRectForRotate(area);
	rect = util::TruncateRectKeepCenter(rect, src.size());

	cv::Mat rot_img;
	RotateImage(src(rect).clone(), rot_img, yaw, pitch, roll, Z, interpolation, boarder_mode, boarder_color);

	cv::Rect dst_area((rot_img.cols - area.width) / 2, (rot_img.rows - area.height) / 2, area.width, area.height);
	dst_area = util::TruncateRectKeepCenter(dst_area, rot_img.size());
	dst = rot_img(dst_area).clone();
}

