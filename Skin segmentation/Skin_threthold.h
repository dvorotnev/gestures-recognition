/*
    Алгоритмы цветовой сегментации.
*/


#ifndef __SKIN_THRETHOLD_H__
#define __SKIN_THRETHOLD_H__

#include <core.hpp>

void thresholdRGB(const cv::Mat& image, cv::Mat& binary_mask, uchar min_R, uchar max_R, uchar min_G, uchar max_G, uchar min_B, uchar max_B);
void thresholdYCrCb(const cv::Mat& image, cv::Mat& binary_mask, uchar min_Y, uchar max_Y, uchar min_Cr, uchar max_Cr, uchar min_Cb, uchar max_Cb);
void thresholdHSV(const cv::Mat& image, cv::Mat& binary_mask, uchar min_H, uchar max_H, uchar min_S, uchar max_S, uchar min_V, uchar max_V);
void thresholdHSL(const cv::Mat& image, cv::Mat& binary_mask, uchar min_H, uchar max_H, uchar min_S, uchar max_S, uchar min_L, uchar max_L);

#endif // __SKIN_THRETHOLD_H__
