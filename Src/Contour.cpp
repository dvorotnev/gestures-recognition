/*
    Реализация алгоритма поиска контуров на бинарном изображении.
*/

#include <assert.h>
#include <optional>

#include <Contour.h>
#include <Debug.h>

using namespace std;
using namespace cv;

const uchar Background = 0;
const uchar ForeGround = 255;

// Кодирует направление между двумя точками.
static int codeDirection(const Point2i& first, const Point2i& second)
{
    const int dx = second.x - first.x;
    const int dy = second.y - first.y;
    if ((abs(dx) > 1) || (abs(dy) > 1))
        return -1;

    if (dx > 0)
    {
        if (dy > 0)
            return 3;
        else if (dy < 0)
            return 1;
        else
            return 2;
    }
    else if (dx < 0)
    {
        if (dy > 0)
            return 5;
        else if (dy < 0)
            return 7;
        else
            return 6;
    }
    else
    {
        if (dy > 0)
            return 4;
        else if (dy < 0)
            return 0;
        else
            return -1;
    }
}

// Функция декодирует направление между двумя точками.
static Point2i decodeDirection(const Point2i& point, const int code)
{
    if ((code > 7) || (code < 0))
        throw;

    Point2i result(0, 0);

    // Определяем координату x.
    switch (code)
    {
        case 0:
        case 4:
            result.x = point.x;
            break;
        case 1:
        case 2:
        case 3:
            result.x = point.x + 1;
            break;
        case 5:
        case 6:
        case 7:
            result.x = point.x - 1;
            break;
        default:
            throw;
    }

    // Определяем координату y.
    switch (code)
    {
        case 0:
        case 1:
        case 7:
            result.y = point.y - 1;
            break;
        case 2:
        case 6:
            result.y = point.y;
            break;
        case 3:
        case 4:
        case 5:
            result.y = point.y + 1;
            break;
        default:
            throw;
    }

    return result;
}

Contour::Contour(const Mat& image, const Point2i& point) : image_size_(image.size()), start_(point), chain_code_()
{
    Point2i current_contour = point;
    Point2i current_bg = {point.x - 1, point.y};

    while (true)
    {
        int direction = codeDirection(current_contour, current_bg);
        int next_chain = -1;

        // Просматриваем 8ку соседей по часовой стрелке, начиная с точки фона.
        for (int i = 1; i <= 7; ++i)
        {
            int new_direction = (direction + i) % 8;
            Point2i next_point = decodeDirection(current_contour, new_direction);

            if (next_point.x < 0 || next_point.x >= image.cols || next_point.y < 0 || next_point.y >= image.rows)
            {
                current_bg = next_point;
                continue;
            }

            uchar value = image.ptr(next_point.y)[next_point.x];
            if (value == Background)
            {
                current_bg = next_point;
                continue;
            }

            next_chain = new_direction;
            if (chain_code_.size() > 0 && current_contour == start_ && next_chain == chain_code_[0])
                return;

            current_contour = next_point;
            break;
        }

        // Контур из одной точки.
        if (next_chain == -1)
            break;

        chain_code_.push_back(next_chain);
    }

    return;
}

size_t Contour::length() const
{
    return chain_code_.size() + 1;
}

vector<Point2i> Contour::getContour() const
{
    vector<Point2i> points(length());
    points[0] = start_;
    for (size_t i = 0; i < length() - 1; ++i)
    {
        points[i + 1] = decodeDirection(points[i], chain_code_[i]);
    }

    return points;
}

void Contour::printContour(Mat& image, uchar label) const
{
    if (image.empty())
        throw;

    Point2i point = start_;

    uchar* image_ptr = image.ptr(point.y);
    image_ptr[point.x] = label;

    Point2i next_point(-1, -1);
    for (vector<int>::const_iterator i = chain_code_.begin(); i != chain_code_.end(); ++i)
    {
        next_point = decodeDirection(point, *i);
        assert((next_point.x >= 0) && (next_point.x < image.cols) &&
               (next_point.y >= 0) && (next_point.y < image.rows));

        // Отмечаем точку контура на изображении.
        uchar* image_ptr = image.ptr(next_point.y);
        image_ptr[next_point.x] = label;

        point = next_point;
    }

    return;
}

Point2i Contour::getContourPoint(size_t point_index) const
{
    if (point_index > length())
        throw;

    Point2i result = start_;
    for (size_t i = 0; i < point_index; ++i)
    {
        result = decodeDirection(result, chain_code_[i]);
    }

    return result;
}

vector<Point2i> Contour::getContourPoints(vector<size_t>& point_indexes) const
{
    size_t size = point_indexes.size();
    vector<Point2i> points(size);

    for (size_t i = 0; i < size; ++i)
    {
        points[i] = getContourPoint(point_indexes[i]);
    }

    return points;
}

Size Contour::getImageSize() const
{
    return image_size_;
}

vector<Contour> extractContours(InputArray BinImage, InputArray Mask)
{
    Mat image(BinImage.getMat());
    Mat mask(Mask.getMat());

    // С помощью бинарной морфологии получаем границы объектов (обводим сверху).
    Matx <uchar, 3, 3> kernel_erode = { 0, 1, 0,
                                        1, 1, 1,
                                        0, 1, 0 };

    Mat contours_image(image.rows, image.cols, CV_8U);
    erode(image, contours_image, kernel_erode, Point(-1,-1), 1, BORDER_CONSTANT, Background);
    contours_image = image - contours_image;
    imageWrite("Erode", contours_image);

    bitwise_and(contours_image, mask, contours_image);

    // Записываем все контуры, найденные на изображении.
    vector<Contour> contours;
    for (int y = 0; y < contours_image.rows; ++y)
    {
        uchar* ptr = contours_image.ptr(y);
        for (int x = 0; x < contours_image.cols; ++x)
        {
            if (ptr[x] != ForeGround)
                continue;

            Point2i point(x, y);
            Contour current(contours_image, point);
            // Удаляем контур с изображения.
            current.printContour(contours_image, Background);
            if (current.length() >= 4)
                contours.push_back(current);
        }
    }

    return contours;
}

void printContours(Mat& image, const vector<Contour>& contours)
{
    image.setTo(0);
    for (size_t i = 0; i < contours.size(); ++i)
    {
#if ___DEBUG___
        uchar label = (uchar)i + 1;
#else
        uchar label = ForeGround;
#endif
        contours[i].printContour(image, label);
    }

    return;
}

// TODO: использовать функцию qsort.
void sortContours(vector<Contour>& contours)
{
    size_t size = contours.size();
    if (size == 0)
        return;

    for (size_t i = size - 1; i > 1; --i)
    {
        for (size_t j = 0; j < i; ++j)
        {
            const size_t current_size = contours[j].length();
            const size_t next_size = contours[j + 1].length();
            if (current_size < next_size)
                swap(contours[j], contours[j + 1]);
        }
    }

    return;
}
