/*
    Функция распознавания руки на основе анализа кривизны контура.
*/

#ifndef __HANDDETECTOR_H__
#define __HANDDETECTOR_H__

#include <vector>
#include <core.hpp>

#include "Contour.h"

struct Finger
{
    cv::Point2i start;
    cv::Point2i peak;
    double length;
};

class Hand
{
public:
    Hand() {};
    Hand(std::vector<cv::Point2i>& points);
    void print(cv::Mat& image);
private:
    Finger fingers[5];
    bool at_front;
};

/*
    Функция на основании анализа кривизны контура вычисляет, является ли контур рукой.
    Входные параметры:
        curvature    - вектор кривизны контура в каждой точке;
        min_treshold - наибольшее значение для минимума;
        max_trethold - наименьшее значение для максимума;
        min_counter  - минимальное количество локальных максимумов;
        max_counter  - максимальное количество локальных максимумов.

    Выходное значение:
        1 - контур является контуром руки;
        0 - контур не является контуром руки;
       -1 - ошибка.
*/
int handDetector(const Contour& contour, float min_treshold, float max_trethold, int min_counter, int max_counter, Hand& hand);

#endif // __HANDDETECTOR_H__
