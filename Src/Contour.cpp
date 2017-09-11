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
static Point2i DecodeDirection(const Point2i& point, const int code)
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

// Поиск направления, в котором расположены точки контура.
static optional<pair<int, Point2i>> findDirection(const Mat& image, const Point2i& start)
{
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
                Point2i next_point(x, y);
                int direction = codeDirection(start, next_point);
                return pair(direction, next_point);
            }
        }
    }

    return {};
}

// Функция для нахождения следующей точки контура на изображении.
static optional<pair<int, Point2i>> findNextPoint(const Mat& image,
                                                  const Point2i& current_point,
                                                  const int last_direction)
{
    if ((last_direction > 7) || (last_direction < 0))
        throw;

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
    Point2i next_point(0, 0);
    for (int i = 0; i < 7; ++i)
    {
        next_point = DecodeDirection(current_point, directions_queue[i]);

        if ((next_point.x < 0) || (next_point.x >= image.cols) ||
            (next_point.y < 0) || (next_point.y >= image.rows))
            continue;

        const uchar* ptr = image.ptr(next_point.y);
        if (ptr[next_point.x] == ForeGround)
            return pair(directions_queue[i], next_point);
    }

    return {};
}

// Функция считывает точки контура, у которого задано начало, и удаляет его с изображения.
Contour::Contour(Mat& image, const Point2i& point) : start_(point), chain_code_()
{
    uchar* ptr = image.ptr(start_.y);
    ptr[start_.x] = Background;

    for (int i = 0; i < 2; ++i)
    {
        Point2i current_point = { 0 };
        auto next_point = findDirection(image, point);

        // Ищем точки контура.
        while (next_point)
        {
            int direction = get<int>(*next_point);
            current_point = get<Point2i>(*next_point);
            chain_code_.push_back(direction);
            uchar* ptr = image.ptr(current_point.y);
            ptr[current_point.x] = Background;
            next_point = findNextPoint(image, current_point, direction);
        }

        // Инвертируем записанные элементы и меняем начало.
        if (i == 0)
        {
            for (size_t j = 0; j < chain_code_.size(); ++j)
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

vector<Point2i> Contour::getContour() const
{
    vector<Point2i> points(length());
    points[0] = start_;
    for (size_t i = 0; i < length() - 1; ++i)
    {
        points[i + 1] = DecodeDirection(points[i], chain_code_[i]);
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
        next_point = DecodeDirection(point, *i);
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
        result = DecodeDirection(result, chain_code_[i]);
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

vector<Contour> extractContours(InputArray& BinImage)
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
