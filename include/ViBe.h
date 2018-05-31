/*
    Алгоритм сегментации движения ViBe.
*/

#ifndef __VIBE_H__
#define __VIBE_H__

#include <opencv2/core.hpp>
#include <opencv2/video.hpp>

class ViBe : public cv::BackgroundSubtractor
{
public:
    ViBe();
    ViBe(int history_depth, int radius, int min_overlap, int probability);
    ~ViBe();

    // Функция вычисляет маску сегментации и обновляет модель.
    // TODO: вычислять prob через learningrate.
    void apply(const cv::InputArray &image, cv::OutputArray &mask, double);
    // Функция вычисляет изображение фона.
    void getBackgroundImage(cv::OutputArray& backgroundImage) const;

protected:
    // Возвращаемое значение равно true, если необходима инициализация
    // модели фона, и false, иначе.
    bool needToInit();
    // Функция инизиализации модели.
    void initialize(const cv::Mat &);
    // Функция классификации точек изображения.
    void getSegmentationMask(const cv::Mat &image, cv::Mat &segmentation_mask);
    // Обновление модели фона в заданной точке.
    void updatePixel(const cv::Mat& image, int y, int x);
    // Обновление модели фона случайного соседа из восьмисвязной области заданной точки.
    void updateNeiborPixel(const cv::Mat& image, int y, int x);

private:
    int history_depth_; // Количество хранимых значений для каждого пикселя.
    int sqr_rad_; // Квадрат максимального расстояния для включения точки в модель.
    int min_overlap_; // Минимальное количество совпадений значения пикселя с моделью.
    int probability_; // Вероятность обновления модели.
    bool initialized_; // Флаг инициализации модели.
    cv::Mat_<cv::Point3_<uchar>*> samples_; // Матрица для хранения значений пикселей.
    cv::Mat bg_mat_; // Матрица для хранения фона.
    cv::RNG generator_; // Генератор случайных чисел (используется равномерный закон распределения).

    // Функция выдаёт случайную точку из восьмисвязной области.
    cv::Point2i getRandomNeiborPixel(const cv::Point2i &);
    // Функция обновления модели алгоритма.
    void update(const cv::Mat &image, const cv::Mat &update_mask);

    // Копирование запрещено
    void operator=(const ViBe &) = delete;
};

#endif // __VIBE_H__
