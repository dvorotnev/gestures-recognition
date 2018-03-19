/*
    Класс алгоритма распознавания жестов.
*/

#ifndef __GESTURES_RECOGNITION_H__
#define __GESTURES_RECOGNITION_H__

#include <list>
#include <opencv2\core.hpp>

#include <Hand.h>

class GesturesRecognition
{
public:
    // Распознавание жестов в списке рук.
    void apply(const std::list<Hand>& hands);
    // Отрисовка на изображении найденных кликов.
    void printClicks(cv::Mat& image) const;

private:
    // Найденные клики.
    std::vector<cv::Point2i> clicks_;
};

#endif // __GESTURES_RECOGNITION_H__
