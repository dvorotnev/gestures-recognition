/*
    Алгоритм поиска контуров на бинарном изображении.
*/

#ifndef __CONTOUR_H__
#define __CONTOUR_H__

#include <opencv2\highgui.hpp>
#include <opencv2\video\video.hpp>
#include <vector>

class Contour
{
public:
    // Функция считывает контур, содержащий заданную точку, с изображения
    // при помощи морфологических преобразований.
    Contour(cv::Mat& image, const cv::Point2i& point);

    size_t length() const;
    // Возвращает вектор точек контура.
    std::vector<cv::Point2i> getContour() const;
    // Функция рисует контур на заданном изображении.
    void printContour(cv::Mat& image, uchar label) const;
    // Функция возвращает точку контура с заданным индексом.
    cv::Point2i getContourPoint(size_t point_index) const;
    // Функция возвращает вектор точек контура с индексами из вектора индексов.
    std::vector<cv::Point2i> getContourPoints(std::vector<size_t>& point_indexes) const;
private:
    // Начало контура.
    cv::Point2i start_;
    // Вектор для хранения цепного кода.
    std::vector<int> chain_code_;
};

// Поиск контуров на изображении.
std::vector<Contour> extractContours(cv::InputArray& BinImage);
// Функция рисует все контуры на изображении.
void printContours(cv::Mat& image, const std::vector<Contour>& contours);
// Функция упорядочивает контуры по убыванию длины.
void sortContours(std::vector<Contour>& contours);
// Функция вычисляет кривизну контура в каждой точке.
int getCurvature(std::vector<float>& curvature, Contour contour, const int chord_length);

#endif // __CONTOUR_H__
