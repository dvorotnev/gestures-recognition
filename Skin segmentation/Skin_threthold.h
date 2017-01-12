/*
    Алгоритмы цветовой сегментации.
*/


#ifndef __SKIN_THRETHOLD_H__
#define __SKIN_THRETHOLD_H__

#include <core.hpp>

void thresholdRGB(const cv::Mat& image, cv::Mat& binary_mask);
void thresholdYCrCb(const cv::Mat& image, cv::Mat& binary_mask);
void thresholdHSV(const cv::Mat& image, cv::Mat& binary_mask);
void thresholdHSL(const cv::Mat& image, cv::Mat& binary_mask);

#endif // __SKIN_THRETHOLD_H__
