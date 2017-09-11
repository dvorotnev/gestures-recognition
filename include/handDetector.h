/*
    Функция распознавания руки на основе анализа кривизны контура.
*/

#ifndef __HANDDETECTOR_H__
#define __HANDDETECTOR_H__

#include <vector>
#include <optional>
#include <opencv2\core.hpp>

#include <Contour.h>

struct Finger
{
    // Точка начала пальца (вблизи ладони).
    cv::Point2i start;
    // Вершина пальца.
    cv::Point2i peak;
    // Длина пальца.
    double length;
};

class Hand
{
public:
    // Создание объекта руки по точкам пальцев.
    Hand(const std::vector<cv::Point2i>& points);
    // Отрисовка точек пальцев на изображении.
    void print(cv::Mat& image);
private:
    // Массив пальцев руки.
    Finger fingers[5];
    // Флаг, равные true, если рука находится ладонью к камере.
    bool at_front;
};

/*
    Функция на основании анализа кривизны контура вычисляет, является ли контур рукой.
    Входные параметры:
        contour      - анализируемый контур;
        min_treshold - наибольшее значение для минимума;
        max_trethold - наименьшее значение для максимума;
        min_counter  - минимальное количество локальных максимумов;
        max_counter  - максимальное количество локальных максимумов.
*/
std::optional<Hand> handDetector(const Contour& contour, float min_treshold, float max_trethold, int min_counter, int max_counter);

#endif // __HANDDETECTOR_H__
