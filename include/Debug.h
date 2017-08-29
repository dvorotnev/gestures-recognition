/*
    Дополнительный функционал для отладки.
*/

#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <opencv2\core.hpp>

#define ___DEBUG___ 0

// Каталог отладки.
#if ___DEBUG___
static const char debug_directory[] = "..\\result\\";
#endif // ___DEBUG___

// Установка метки в заданной точке изображения в зависимости от режима работы программы.
inline void setLabel(cv::Mat& image, const cv::Point2i& point, uchar label, uchar debug_label)
{
    uchar* ptr = image.ptr(point.y);
#if ___DEBUG___
    ptr[point.x] = debug_label;
#else
    ptr[point.x] = label;
#endif // ___DEBUG___
}

// В режиме отладки: запись изображения в каталог отладки.
void imageWrite(const cv::String& name, const cv::Mat& mat);

// В режиме отладки: запись изображения в каталог отладки.
// В обычном режиме: вывод изображения на экран.
void imageShow(const cv::String& winname, const cv::Mat& mat);

// В режиме отладки: запись изображения в текстовый файл в каталог отладки.
// В обычном режиме: вывод изображения на экран.
void contoursShow(const cv::String& winname, const cv::Mat& mat);

// В режиме отладки: запись вектора в текстовый файл.
// В обычном режиме: вывод графика на экран.
void curvatureShow(const cv::String& winname, const std::vector<float>& curvature);

#endif // __DEBUG_H__
