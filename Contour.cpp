/*
    Реализация алгоритма поиска контуров на бинарном изображении.
*/

#include <assert.h>

#include "Contour.h"
#include "Debug.h"

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
// В second записываются координаты второй точки.
// TODO: вместо ошибки выдавать исключение. Возвращать всегда полученную точку.
static int DecodeDirection(const Point2i& first, Point2i& second, const int code)
{
    if ((code > 7) || (code < 0))
        return -1;

    // Определяем координату x.
    switch (code)
    {
        case 0:
        case 4:
            second.x = first.x;
            break;
        case 1:
        case 2:
        case 3:
            second.x = first.x + 1;
            break;
        case 5:
        case 6:
        case 7:
            second.x = first.x - 1;
            break;
        default:
            assert("Unreachable part.");
            return -1;
    }

    // Определяем координату y.
    switch (code)
    {
        case 0:
        case 1:
        case 7:
            second.y = first.y - 1;
            break;
        case 2:
        case 6:
            second.y = first.y;
            break;
        case 3:
        case 4:
        case 5:
            second.y = first.y + 1;
            break;
        default:
            break;
    }

    return 0;
}

// Поиск направления, в котором расположены точки контура.
static int findDirection(const Mat& image, const Point2i& start, Point2i& next_point)
{
    int direction = -1;
    for (int y = start.y - 1; y <= start.y + 1; ++y)
    {
        if ((y < 0) || (y >= image.rows))
            continue;
        const uchar* image_ptr = image.ptr(y);
        for (int x = start.x - 1; x <= start.x + 1; ++x)
        {
            if ((x < 0) || (x >= image.cols))
                continue;
            if ((x == start.x) && (y == start.y))
                continue;
            if (image_ptr[x] == ForeGround)
            {
                next_point.x = x;
                next_point.y = y;
                direction = codeDirection(start, next_point);
                return direction;
            }
        }
    }

    return direction;
}

// Функция для нахождения следующей точки контура на изображении.
static int findNextPoint(const Mat& image,
                         const Point2i& current_point,
                         Point2i& next_point,
                         const int last_direction)
{
    // TODO: exception.
    if ((last_direction > 7) || (last_direction < 0))
        return -1;

    // Массив всех возможных направлений в порядке
    // наиболее вероятного поялвения.
    int directions_queue[7] = { last_direction,
        last_direction + 1, last_direction - 1,
        last_direction + 2, last_direction - 2,
        last_direction + 3, last_direction - 3 };

    // Приводим значения направлений в диапазон от 0 до 7.
    for (int i = 0; i < 7; ++i)
    {
        if (directions_queue[i] > 7)
            directions_queue[i] -= 8;
        else if (directions_queue[i] < 0)
            directions_queue[i] += 8;
    }

    // Ищем следующую точку среди возможных направлений.
    for (int i = 0; i < 7; ++i)
    {
        if (DecodeDirection(current_point, next_point, directions_queue[i]) != 0)
            return -1;
        if ((next_point.x < 0) || (next_point.x >= image.cols) ||
            (next_point.y < 0) || (next_point.y >= image.rows))
            continue;

        const uchar* ptr = image.ptr(next_point.y);
        if (ptr[next_point.x] == ForeGround)
            return directions_queue[i];
    }
    return -1;
}

// Функция считывает точки контура, у которого задано начало, и удаляет его с изображения.
Contour::Contour(Mat& image, Point2i& point) : start_(point), chain_code_()
{
    uchar* ptr = image.ptr(start_.y);
    ptr[start_.x] = Background;

    for (int i = 0; i < 2; ++i)
    {
        Point2i current_point = { 0 };
        Point2i next_point = { 0 };
        int direction = findDirection(image, start_, next_point);

        // Ищем точки контура.
        while (direction != -1)
        {
            current_point = next_point;
            chain_code_.push_back(direction);
            uchar* ptr = image.ptr(current_point.y);
            ptr[current_point.x] = Background;
            direction = findNextPoint(image, current_point, next_point, direction);
        }

        // Инвертируем записанные элементы и меняем начало.
        if (i == 0)
        {
            for (int j = 0; j < chain_code_.size(); ++j)
            {
                const int direction = chain_code_[j];
                const int inverse_direction = (direction + 4) % 8;
                chain_code_[j] = inverse_direction;
            }

            reverse(chain_code_.begin(), chain_code_.end());
            start_ = current_point;
        }
    }

    return;
}

size_t Contour::length() const
{
    return chain_code_.size() + 1;
}

int Contour::getContour(vector<Point2i>& points) const
{
    points.resize(length());
    points[0] = start_;
    for (int i = 0; i < length() - 1; ++i)
    {
        if (DecodeDirection(points[i], points[i + 1], chain_code_[i]) != 0)
            return -1;
    }

    return 0;
}

void Contour::printContour(Mat& image, uchar label) const
{
    // TODO: exception.
    if (image.empty())
        return;

    Point2i point = start_;

    uchar* image_ptr = image.ptr(point.y);
    image_ptr[point.x] = label;

    Point2i next_point(-1, -1);
    for (vector<int>::const_iterator i = chain_code_.begin(); i != chain_code_.end(); ++i)
    {
        // TODO: exception.
        if (DecodeDirection(point, next_point, *i) != 0)
            return;
        assert((next_point.x >= 0) && (next_point.x < image.cols) &&
               (next_point.y >= 0) && (next_point.y < image.rows));

        // Отмечаем точку контура на изображении.
        uchar* image_ptr = image.ptr(next_point.y);
        image_ptr[next_point.x] = label;

        point = next_point;
    }
}

int Contour::getContourPoint(Point2i& point, int point_index) const
{
    if (point_index > length())
        return -1;

    Point2i result = start_;
    for (int i = 0; i < point_index; ++i)
    {
        DecodeDirection(result, result, chain_code_[i]);
    }

    point = result;
    return 0;
}

int Contour::getContourPoints(vector<Point2i>& points, vector<int>& point_indexes) const
{
    size_t size = point_indexes.size();
    points.resize(size);

    int result = 0;
    for (int i = 0; i < size; ++i)
    {
        result = getContourPoint(points[i], point_indexes[i]);
        if (result != 0)
            break;
    }

    return result;
}

void extractContours(InputArray& BinImage, vector<Contour>& contours)
{
    Mat image(BinImage.getMat());

    // С помощью бинарной морфологии получаем границы объектов (обводим сверху).
    Matx <uchar, 3, 3> kernel_erode = { 0, 1, 0,
        1, 1, 1,
        0, 1, 0 };

    Mat contours_image(image.rows, image.cols, CV_8U);
    erode(image, contours_image, kernel_erode);
    contours_image = image - contours_image;
    imageWrite("Erode", contours_image);

    // Записываем все контуры, найденные на изображении.
    for (int y = 0; y < contours_image.rows; ++y)
    {
        uchar* ptr = contours_image.ptr(y);
        for (int x = 0; x < contours_image.cols; ++x)
        {
            if (ptr[x] != ForeGround)
                continue;

            Point2i point(x, y);
            Contour current(contours_image, point);
            if (current.length() >= 4)
                contours.push_back(current);
        }
    }

    return;
}

void printContours(Mat& image, const vector<Contour>& contours)
{
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
