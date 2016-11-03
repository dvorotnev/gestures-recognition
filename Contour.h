/*
    Алгоритм поиска контуров на бинарном изображении.
*/

#include <highgui.hpp>
#include <video\video.hpp>
#include <vector>

struct Contour
{
    // Начало контура.
    cv::Point2i start;
    // Вектор для хранения цепного кода.
    std::vector<int> chain_code;
};

class ContourMap
{
public:
    // Функция возвращает количество хранимых контуров.
    size_t getNumberOfContours() const;
    // Функция рисует контур на заданном изображении.
    void printContour(cv::Mat& image, int number) const;
    // Функция рисует все контуры на изображении.
    void printAllContours(cv::Mat& image) const;
    // Функция упорядочивает контуры по убыванию длины.
    void sortContours();
    // Функция вычисляет кривизну контура в каждой точке.
    int getCurvature(std::vector<float>& curvature, const int chord_length, int number) const;
    // Поиск контуров на изображении.
    virtual void extractContours(cv::InputArray& Image) = 0;

protected:
    // Вектор контуров на изображении.
    std::vector<Contour> contours_;
};

class ContourMapMorph : public ContourMap
{
public:
    //Функция считывает все контуры с изображения при помощи морфологических преобразований.
    void extractContours(cv::InputArray& Image);
};
