/*
    Функция для удаления мелких объектов с бинарного изображения.
*/

#ifndef __DELETENOISE_H__
#define __DELETENOISE_H__

#include <opencv2/core.hpp>

// Функция удаляет объекты, меньшие по площади, чем min_fg_area
// и "дырки" в объектах, меньшие по площади, чем min_bg_area.
void deleteNoise(cv::Mat &image, int min_fg_area, int min_bg_area);

#endif // __DELETENOISE_H__
