/*
    Алгоритм сегментации движения ViBe.
*/

#ifndef __VIBE_H__
#define __VIBE_H__

#include <core.hpp>
#include <video.hpp>

class ViBe : public cv::BackgroundSubtractor
{
private:
    const uchar BackGround = 0;
    const uchar ForeGround = 255;

    int history_depth; // Количество хранимых значений для каждого пикселя.
    int sqr_rad; // Квадрат максимального расстояния для включения точки в модель.
    int min_overlap; // Минимальное количество совпадений значения пикселя с моделью.
    int prob; // Вероятность обновления модели.
    cv::Mat_<cv::Point3i*> samples; // Матрица для хранения значений пикселей.
    cv::RNG generator; // Генератор случайных чисел (используется равномерный закон распределения).

    // Функция, вычисляющая квадрат расстояния между двумя точками.
    double distancesqr(const cv::Point3i &, const cv::Point3i &) const;
    // Функция инизиализации модели.
    void initialization(const cv::Mat &);
    // Функция выдаёт случайную точку из восьмисвязной области.
    cv::Point2i GetRandNeibPixel(const cv::Point2i &); 

    // Копирование запрещено
    void operator=(const ViBe &) = delete;

public:
    // Функция вычисляет маску сегментации и обновляет модель.
    // TODO: вычислять prob через learningrate.
    void apply(const cv::InputArray &Image, cv::OutputArray &mask, double);
    // TODO: функция не реализована.
    void getBackgroundImage(cv::OutputArray &backgroundImage) const;
    ViBe();
    ViBe(int history_depth, int rad, int h, int prob);
    ~ViBe();
};

#endif // __VIBE_H__
