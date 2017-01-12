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

void thresholdRGB(const Mat& image, Mat& binary_mask)
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

            if (pixel.x > 95 && pixel.y > 40 && pixel.z > 20 &&
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

void thresholdYCrCb(const Mat& image, Mat& binary_mask)
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

            if (pixel.x >= 50 && pixel.y >= 131 && pixel.y <= 185 &&
                pixel.z >= 80 && pixel.z <= 135)
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

void thresholdHSV(const Mat& image, Mat& binary_mask)
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
            Point2i pixel(src_ptr[3 * x], src_ptr[3 * x + 1]);

            if (pixel.x <= 20 && pixel.y >= 45)
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

void thresholdHSL(const Mat& image, Mat& binary_mask)
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

            if (pixel.x <= 20 && pixel.y >= 45 && pixel.z >= 50)
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
