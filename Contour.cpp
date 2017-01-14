/*
    Реализация алгоритма поиска контуров на бинарном изображении.
*/

#include <assert.h>

#include "Contour.h"

#define ___DEBUG___ 0

#if ___DEBUG___
#include <string>

extern unsigned int debug_counter;
extern char path[];
#endif

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

size_t ContourMap::getNumberOfContours() const
{
    return contours_.size();
}

void ContourMap::printContour(Mat& image, int number) const
{
    // TODO: exception.
    if (image.empty())
        return;

    size_t size = getNumberOfContours();
    if ((number + 1 > size) || (number < 0))
        return;

    Contour current = contours_[number];
    Point2i point = current.start;

    uchar* ptr = image.ptr(point.y);

#if ___DEBUG___
    ptr[point.x] = (uchar)number;
#else
    ptr[point.x] = ForeGround;
#endif

    vector<int>& code = current.chain_code;
    Point2i next_point(-1,-1);
    for (vector<int>::iterator i = code.begin(); i != code.end(); ++i)
    {
        // TODO: exception.
        if (DecodeDirection(point, next_point, *i) != 0)
            return;
        assert((next_point.x >= 0) && (next_point.x < image.cols) &&
               (next_point.y >= 0) && (next_point.y < image.rows));

        // Отмечаем точку контура на изображении.
        ptr = image.ptr(next_point.y);

#if ___DEBUG___
        ptr[next_point.x] = (uchar)(number+1);
#else
        ptr[next_point.x] = ForeGround;
#endif

        point = next_point;
    }
}

void ContourMap::printAllContours(Mat& image) const
{
    if (image.empty())
        return;

    for (int y = 0; y < image.rows; ++y)
    {
        uchar* ptr = image.ptr(y);
        for (int x = 0; x < image.cols; ++x)
        {
            if (ptr[x] != Background)
                ptr[x] = Background;
        }
    }

    for (int i = 0; i < getNumberOfContours(); ++i)
    {
        printContour(image, i);
    }
}

// TODO: использовать функцию qsort.
void ContourMap::sortContours()
{
    size_t size = contours_.size();
    if (size == 0)
        return;

    for (size_t i = size - 1; i > 1; --i)
    {
        for (size_t j = 0; j < i; ++j)
        {
            const size_t current_size = contours_[j].chain_code.size();
            const size_t next_size    = contours_[j + 1].chain_code.size();
            if (current_size < next_size)
                swap(contours_[j], contours_[j + 1]);
        }
    }

    return;
}

int ContourMap::getCurvature(std::vector<float>& curvature,
                             const int chord_length,
                             int number) const
{
    size_t size = contours_.size();
    if ((number + 1 > size) || (number < 0))
        return 0;

    const Contour& contour = contours_[number];
    const vector<int>& chain_code = contour.chain_code;
    const size_t length = chain_code.size() + 1;
    if (length < 4)
        return -1;

    // Декодируем все точки контура.
    vector<Point2i> points(length);
    points[0] = contour.start;
    for (int i = 0; i < length - 1; ++i)
    {
        if (DecodeDirection(points[i], points[i + 1], chain_code[i]) != 0)
            return -1;
    }

    curvature.resize(length, 0);
    for (int i = 0; i < length; ++i)
    {
        // Вычисляем координаты концов хорды.
        const int first_point = i - chord_length + 2;
        const int last_point = i + 1;

        // Смещаем хорду по контуру.
        for (int shift = 0; shift <= chord_length - 3; ++shift)
        {
            int start_index = first_point + shift;
            if (start_index < 0)
                start_index = 0;
            else if (start_index >= length)
                start_index = (int)length - 1;

            int end_index = last_point + shift;
            if (end_index >= length)
                end_index = (int)length - 1;

            // Вычисляем расстояние от точки до хорды.
            const Point2i chord = points[end_index] - points[start_index];
            const Point2i point_to_chord = points[i] - points[start_index];

            double distance = abs(point_to_chord.x * chord.y -
                                 point_to_chord.y * chord.x);
            distance /= sqrt(chord.x * chord.x + chord.y * chord.y);
            // Максимальное расстояние и будет кривизной контура в точке.
            if (distance > curvature[i])
                curvature[i] = (float)distance;
        }
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
    int directions_queue[7] = {last_direction,
        last_direction + 1, last_direction - 1,
        last_direction + 2, last_direction - 2,
        last_direction + 3, last_direction - 3};

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
static void extractContour(Mat& image, Contour& contour)
{
    Point2i start(contour.start);
    uchar* ptr = image.ptr(start.y);
    ptr[start.x] = Background;
    vector<int>& chain_code = contour.chain_code;

    for (int i = 0; i < 2; ++i)
    {
        Point2i current_point = { 0 };
        Point2i next_point = { 0 };
        int direction = findDirection(image, start, next_point);

        // Ищем точки контура.
        while (direction != -1)
        {
            current_point = next_point;
            chain_code.push_back(direction);
            uchar* ptr = image.ptr(current_point.y);
            ptr[current_point.x] = Background;
            direction = findNextPoint(image, current_point, next_point, direction);
        }

        // Инвертируем записанные элементы и меняем начало.
        if (i == 0)
        {
            for (int j = 0; j < chain_code.size(); ++j)
            {
                const int direction = chain_code[j];
                const int inverse_direction = (direction + 4) % 8;
                chain_code[j] = inverse_direction;
            }

            reverse(contour.chain_code.begin(), contour.chain_code.end());
            contour.start = current_point;
        }
    }
}

void ContourMapMorph::extractContours(InputArray& BinImage)
{
    Mat image(BinImage.getMat());

    //TODO: проверить.
    // С помощью бинарной морфологии получаем границы объектов (обводим сверху).
    Matx <uchar, 4, 4> kernel_close = { 1, 1, 1, 1,
                                        1, 1, 1, 1,
                                        1, 1, 1, 1,
                                        1, 1, 1, 1};
    morphologyEx(image, image, MORPH_CLOSE, kernel_close);
#if ___DEBUG___
    imwrite(String(path) + "res_close\\" + std::to_string(debug_counter) + ".png", image);
#endif

    Matx <uchar, 3, 3> kernel_erode = {0, 1, 0,
                                       1, 1, 1,
                                       0, 1, 0};
    Mat temp(image.rows, image.cols, CV_8U);
    erode(image, temp, kernel_erode);
    temp = image - temp;
#if ___DEBUG___
    //TODO: rename
    imwrite(String(path) + "res_erode\\" + std::to_string(debug_counter) + ".png", temp);
#endif

    // Записываем все контуры, найденные на изображении.
    for (int y = 0; y < temp.rows; ++y)
    {
        uchar* ptr = temp.ptr(y);
        for (int x = 0; x < temp.cols; ++x)
        {
            if (ptr[x] != ForeGround)
                continue;
            Contour current = {Point2i(x, y), vector<int>()};
            extractContour(temp, current);
            if (current.chain_code.size() >= 4)
                contours_.push_back(current);
        }
    }
}
