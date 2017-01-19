/*
    Реализация алгоритмов цветовой сегментации.
*/

#include "Skin_threthold.h"

#include <imgproc.hpp>

using namespace cv;

const uchar Skin = 255;
const uchar NonSkin = 0;

static int maxComponent(Point3i& pixel)
{
    int result = max(pixel.x, pixel.y);
    result = max(result, pixel.z);
    return result;
}

static int minComponent(Point3i& pixel)
{
    int result = min(pixel.x, pixel.y);
    result = min(result, pixel.z);
    return result;
}

void thresholdRGB(const Mat& image, Mat& binary_mask, uchar min_R, uchar max_R,
                  uchar min_G, uchar max_G, uchar min_B, uchar max_B)
{
    binary_mask.create(image.size(), CV_8U);

    for (int y = 0; y < image.rows; ++y)
    {
        const uchar* src_ptr = image.ptr(y);
        uchar* dst_ptr = binary_mask.ptr(y);
        for (int x = 0; x < image.cols; ++x)
        {
            Point3i pixel(src_ptr[3 * x + 2], src_ptr[3 * x + 1], src_ptr[3 * x]);
            int max = maxComponent(pixel);
            int min = minComponent(pixel);

            if (min_R < pixel.x && pixel.x < max_R &&
                min_G < pixel.y && pixel.y < max_G &&
                min_B < pixel.z && pixel.z < max_B &&
                (max - min) > 15 && abs(pixel.x - pixel.y) > 15 &&
                pixel.x > pixel.y && pixel.x > pixel.z)
            {
                dst_ptr[x] = Skin;
            }
            else
            {
                dst_ptr[x] = NonSkin;
            }
        }
    }

    return;
}

void thresholdYCrCb(const Mat& image, Mat& binary_mask, uchar min_Y, uchar max_Y,
                    uchar min_Cr, uchar max_Cr, uchar min_Cb, uchar max_Cb)
{
    binary_mask.create(image.size(), CV_8U);
    Mat converted_image;
    cvtColor(image, converted_image, COLOR_BGR2YCrCb);

    for (int y = 0; y < image.rows; ++y)
    {
        const uchar* src_ptr = converted_image.ptr(y);
        uchar* dst_ptr = binary_mask.ptr(y);
        for (int x = 0; x < image.cols; ++x)
        {
            Point3i pixel(src_ptr[3 * x], src_ptr[3 * x + 1], src_ptr[3 * x + 2]);

            if (min_Y < pixel.x && pixel.x < max_Y &&
                min_Cr < pixel.y && pixel.y < max_Cr &&
                min_Cb < pixel.z && pixel.z < max_Cb)
            {
                dst_ptr[x] = Skin;
            }
            else
            {
                dst_ptr[x] = NonSkin;
            }
        }
    }

    return;
}

void thresholdHSV(const Mat& image, Mat& binary_mask, uchar min_H, uchar max_H,
                  uchar min_S, uchar max_S, uchar min_V, uchar max_V)
{
    binary_mask.create(image.size(), CV_8U);
    Mat converted_image;
    cvtColor(image, converted_image, COLOR_BGR2HSV);

    for (int y = 0; y < image.rows; ++y)
    {
        const uchar* src_ptr = converted_image.ptr(y);
        uchar* dst_ptr = binary_mask.ptr(y);
        for (int x = 0; x < image.cols; ++x)
        {
            Point3i pixel(src_ptr[3 * x], src_ptr[3 * x + 1], src_ptr[3 * x + 2]);

            if (min_H < pixel.x && pixel.x < max_H &&
                min_S < pixel.y && pixel.y < max_S &&
                min_V < pixel.z && pixel.z < max_V)
            {
                dst_ptr[x] = Skin;
            }
            else
            {
                dst_ptr[x] = NonSkin;
            }
        }
    }

    return;
}

void thresholdHSL(const Mat& image, Mat& binary_mask, uchar min_H, uchar max_H,
                  uchar min_S, uchar max_S, uchar min_L, uchar max_L)
{
    binary_mask.create(image.size(), CV_8U);
    Mat converted_image;
    cvtColor(image, converted_image, COLOR_BGR2HLS);

    for (int y = 0; y < image.rows; ++y)
    {
        const uchar* src_ptr = converted_image.ptr(y);
        uchar* dst_ptr = binary_mask.ptr(y);
        for (int x = 0; x < image.cols; ++x)
        {
            Point3i pixel(src_ptr[3 * x], src_ptr[3 * x + 2], src_ptr[3 * x + 1]);

            if (min_H < pixel.x && pixel.x < max_H &&
                min_S < pixel.y && pixel.y < max_S &&
                min_L < pixel.z && pixel.z < max_L)
            {
                dst_ptr[x] = Skin;
            }
            else
            {
                dst_ptr[x] = NonSkin;
            }
        }
    }

    return;
}
