/*
    Класс детектора руки на основе анализа кривизны контура.
*/

#ifndef __HANDDETECTOR_H__
#define __HANDDETECTOR_H__

#include <vector>
#include <list>
#include <opencv2/core.hpp>

#include <Hand.h>

class HandDetector
{
public:
    HandDetector(float min_threshold, float max_threthold, int min_counter, int max_counter);

    // Отслеживание перемещения рук на изображении.
    void trace(cv::InputArray BinaryImage);
    // Обнаружение новых рук на изображении.
    void detect(cv::InputArray BinaryImage);
    // Отрисовка всех найденных рук.
    void printHands(cv::InputArray Image) const;
    // Возвращает список обнаруженных рук.
    const std::list<Hand>& getHands() const;

private:
    // Обновление маски рук.
    void updateMask(cv::Size size);

    // Список обнаруженных рук.
    std::list<Hand> hands_;
    // Маска рук.
    cv::Mat mask_;
    // Пирамида изображений с предыдущего кадра.
    std::vector<cv::Mat> prev_pyr_;
    // Наибольшее значение для минимума.
    float min_threshold_;
    // Наименьшее значение для максимума.
    float max_threthold_;
    // Минимальное количество локальных максимумов.
    int min_counter_;
    // Максимальное количество локальных максимумов.
    int max_counter_;
};

#endif // __HANDDETECTOR_H__
