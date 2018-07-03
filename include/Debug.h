/*
    Дополнительный функционал для отладки.
*/

#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <opencv2/core.hpp>

#define ___DEBUG___ 0

// Каталог отладки.
#if ___DEBUG___
static const char debug_directory[] = "..\\result\\";
#endif // ___DEBUG___

// В режиме отладки: запись изображения в каталог отладки.
void imageWrite(const cv::String& name, const cv::Mat& mat);

// В режиме отладки: запись изображения в каталог отладки.
// В обычном режиме: вывод изображения на экран.
void imageShow(const cv::String& winname, const cv::Mat& mat);

#endif // __DEBUG_H__
