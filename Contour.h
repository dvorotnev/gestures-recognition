#include <highgui.hpp>
#include <video\video.hpp>
#include <vector>

#define ___DEBUG___ 0

struct Contour
{
    // Начало контура.
    cv::Point2i start;
    // Вектор для хранения изменения цепного кода.
    std::vector<int> chain_code;
};

class ContourMap
{
protected:
    // Вектор контуров на изображении.
    std::vector<Contour> contours;
public:
    // Функция возвращает количество хранимых контуров.
    size_t getNumberOfContours() const;
    // Функция рисует контур на заданном изображении.
    void printСontour(cv::Mat& image, int number) const;
    // Функция писует все контуры на изображении.
    void printAllContours(cv::Mat& image) const;
    // Функция упорядочивает контуры по убыванию длины.
    void sortContours();
    // Поиск контуров на изображении.
    virtual void findContours(cv::InputArray& Image) = 0;
};

class ContourMapMorph : public ContourMap
{
public:
    //Функция считывает все контуры с изображения при помощи морфологических преобразований.
    void findContours(cv::InputArray& Image);
};

//class ContourMap2 : public ContourMap
//{
//private:
//    //Функция считает сумму значений соседних точек.
//    bool IsContourPixel(const cv::Mat &image, int x, int y);
//    std::vector<cv::Point2i> ReturnContour(cv::Mat& image, const cv::Point2i start, int mark);
//    bool FindContour(const cv::Mat& image, cv::Point2i currentPixel, int mark) const;
//    cv::Point2i FindNextPoint(const cv::Mat& image, cv::Point2i analyzPixel, int mark, cv::Point2i prevPixel ) const;
//    void Preprocessing(cv::Mat& image);
//public:
//    void findContours(cv::InputArray& Image);
//};
