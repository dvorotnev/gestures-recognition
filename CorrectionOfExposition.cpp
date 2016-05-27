/*
    Реализация функции коррекции яркости.
*/

#include "CorrectionOfExposition.h"

#include <imgproc.hpp>
#include <math.h>

using namespace cv;

const uchar Background = 0;

void correctionOfExposition(const Mat& SegmentationMask, const Mat& BackgroundImage, Mat& CurrentImage)
{
    // Размер зоны вокруг объектов, зарезервированной под движение.
    // (Ориентировочное расстояние, на которое могли переместиться
    // объекты между двумя кадрами).
    const int reserved_motion_distance = 5;
    // Изображение с отмечеными фоновыми пикселями.
    Mat markedImage;

    // Создраём вокруг объектов область из reserved_motion_distance точек,
    // в которой может появиться движение.
    Matx<uchar, 3, 3> kernel = {
        0, 1, 0,
        1, 1, 1,
        0, 1, 0};
    dilate(SegmentationMask, markedImage, kernel, Point(-1, -1), reserved_motion_distance);

    // Высчитываем суммарную яркость точек фона.
    double background_light = 0; // Переменная для хранения суммарной яркости фона.
    double current_light = 0; // Переменная для хранения суммарной яркости текущего кадра.
    int counter = 0; // Переменная для хранения количества точек по которым посчитана сумма.
    for (int y = reserved_motion_distance; y < markedImage.rows - reserved_motion_distance; ++y)
    {
        const uchar* ptr_markedImage = markedImage.ptr(y);
        const uchar* ptr_BackgroundImage = BackgroundImage.ptr(y);
        const uchar* ptr_CurrentImage = CurrentImage.ptr(y);
        for (int x = reserved_motion_distance; x < markedImage.cols - reserved_motion_distance; ++x)
        {
            if (ptr_markedImage[x] != Background) 
                continue;
            background_light += 0.114 * ptr_BackgroundImage[3 * x];
            background_light += 0.587 * ptr_BackgroundImage[3 * x + 1];
            background_light += 0.299 * ptr_BackgroundImage[3 * x + 2];

            current_light += 0.114 * ptr_CurrentImage[3 * x];
            current_light += 0.587 * ptr_CurrentImage[3 * x + 1];
            current_light += 0.299 * ptr_CurrentImage[3 * x + 2];
            counter++;
        }
    }

    // Вычисляем среднее значение изменения яркости.
    if (counter == 0)
        return;
    double diff = (double)(background_light - current_light) / counter;

    //Корректируем яркость на изображении.
    for (int y = 0; y < CurrentImage.rows; ++y)
    {
        uchar* ptr_CurrentImage = CurrentImage.ptr(y);
        for (int x = 0; x < CurrentImage.cols; ++x)
        {
            int temp = (int)(0.114 * diff);
            ptr_CurrentImage[3 * x] = (uchar)min(max((ptr_CurrentImage[3 * x] + temp), 0), 255);
            temp = (int)(0.587 * diff);
            ptr_CurrentImage[3 * x + 1] = (uchar)min(max((ptr_CurrentImage[3 * x + 1] + temp), 0), 255);
            temp = (int)(0.299 * diff);
            ptr_CurrentImage[3 * x + 2] = (uchar)min(max((ptr_CurrentImage[3 * x + 2] + temp), 0), 255);
        }
    }
}
