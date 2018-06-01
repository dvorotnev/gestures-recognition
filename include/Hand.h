/*
    Класс руки.
*/

#ifndef __HAND_H__
#define __HAND_H__

#include <vector>
#include <opencv2/core.hpp>

struct Finger
{
    // Точка начала пальца (вблизи ладони).
    cv::Point2i start;
    // Вершина пальца.
    cv::Point2i peak;
    // Длина пальца.
    double length;
    // Палец согнут.
    bool is_bent;
    // Состояние пальца изменено.
    bool status_changed;
};

class Hand
{
public:
    // Создание объекта руки по точкам пальцев.
    Hand(const std::vector<cv::Point2i>& points);
    // Возвращает точку на запястье.
    cv::Point2i getWrist() const;
    // Возвращает массив пальцев руки.
    const Finger* getHandFingers() const;
    // Отрисовка точек пальцев на изображении.
    void print(cv::Mat& image) const;
    // Возвращает прямоугольник, содержащий руку.
    cv::Rect2i getBoundingBox();
    // Обновление модели руки.
    int update(const std::vector<cv::Mat>& prevPyr, const std::vector<cv::Mat>& nextPyr);

private:
    // Массив пальцев руки.
    Finger fingers_[5];
    // Точка локального максимума кривизы контура между средним и безымянным пальцами.
    cv::Point2i midle_point_;
};

#endif // __HAND_H__
