/*
    Реализация функции распознавания руки на основе анализа кривизны контура.
*/

#include <algorithm>

#include <handDetector.h>

using namespace std;
using namespace cv;

// Функция вычисления средней точки между двумя заданными.
static Point2i midPoint(const Point2i& first, const Point2i& second)
{
    int x = (first.x + second.x) / 2;
    int y = (first.y + second.y) / 2;
    return Point2i(x, y);
}

// Функция заполнения структуры пальца.
static void fillFinger(Finger& finger, const Point2i& start, const Point2i& peak, double length)
{
    finger.start = start;
    finger.peak = peak;
    finger.length = length;
    return;
}

// Проектирование точки на прямую, заданную двумя точками.
static Point2i projection(const Point2i& line_start, const Point2i& line_end, const Point2i& point)
{
    Point2i line_vec = line_end - line_start;
    Point2i point_vec = point - line_start;
    double coef = line_vec.x * point_vec.x + line_vec.y * point_vec.y;
    coef /= line_vec.x * line_vec.x + line_vec.y * line_vec.y;
    Point2i result = line_vec * coef + line_start;
    return result;
}

Hand::Hand(const vector<Point2i>& points)
{
    if (points.size() != 9)
        throw;

    double first_difference = abs(norm(points[1] - points[2]) - norm(points[2] - points[3]));
    double second_difference = abs(norm(points[5] - points[6]) - norm(points[6] - points[7]));

    if (first_difference > second_difference)
    {
        // Большой палец
        fillFinger(fingers_[0], points[1], points[0], norm(points[0] - points[1]));
        // Указательный палец
        fillFinger(fingers_[1], points[3], points[2], norm(points[2] - points[3]));
        // Мизинец
        fillFinger(fingers_[4], points[7], points[8], norm(points[8] - points[7]));
        // Безымянный палец
        Point2i mid_point = midPoint(points[5], points[7]);
        mid_point = projection(fingers_[1].start, fingers_[4].start, mid_point);
        fillFinger(fingers_[3], mid_point, points[6], norm(points[6] - mid_point));
    }
    else
    {
        // Большой палец
        fillFinger(fingers_[0], points[7], points[8], norm(points[8] - points[7]));
        // Указательный палец
        fillFinger(fingers_[1], points[5], points[6], norm(points[6] - points[5]));
        // Мизинец
        fillFinger(fingers_[4], points[1], points[0], norm(points[0] - points[1]));
        // Безымянный палец
        Point2i mid_point = midPoint(points[1], points[3]);
        mid_point = projection(fingers_[1].start, fingers_[4].start, mid_point);
        fillFinger(fingers_[3], mid_point, points[2], norm(points[2] - mid_point));
    }

    // Средний палец
    Point2i mid_point = midPoint(points[3], points[5]);
    mid_point = projection(fingers_[1].start, fingers_[4].start, mid_point);
    fillFinger(fingers_[2], mid_point, points[4], norm(points[4] - mid_point));

    wrist_ = fingers_[2].start + 1.1 * (fingers_[2].start - fingers_[2].peak);
}

static void printRing(Mat& image, const Point2i& center, int big_radius, int little_radius)
{
    for (int y = center.y - big_radius; y <= center.y + big_radius; ++y)
    {
        if (y < 0 || y >= image.rows)
            continue;

        uchar* ptr = image.ptr(y);
        for (int x = center.x - big_radius; x <= center.x + big_radius; ++x)
        {
            if (x < 0 || x >= image.cols)
                continue;

            int distance = (y - center.y) * (y - center.y);
            distance += (x - center.x) * (x - center.x);
            if (distance >= little_radius && distance <= big_radius)
                ptr[x] = 255;
        }
    }

    return;
}

void Hand::print(Mat& image)
{
    printRing(image, fingers_[0].peak, 30, 20);
    printRing(image, fingers_[0].start, 30, 20);

    printRing(image, fingers_[1].peak, 25, 15);
    printRing(image, fingers_[1].start, 25, 15);

    printRing(image, fingers_[2].peak, 20, 10);
    printRing(image, fingers_[2].start, 20, 10);

    printRing(image, fingers_[3].peak, 15, 5);
    printRing(image, fingers_[3].start, 15, 5);

    printRing(image, fingers_[4].peak, 10, 0);
    printRing(image, fingers_[4].start, 10, 0);


    printRing(image, wrist_, 50, 0);

    return;
}

static int countLocalMax(const vector<float>& extremums, float min_threshold, float max_threthold)
{
    const size_t length = extremums.size();
    int counter = 0; // Счётчик локальных максимумов.
    float prev = 0; // Предыдущее значение в точке экстремума.
    bool search_max = true; // Флаг, указывающий ищется максимум или минимум.

    for (size_t i = 0; i < length - 1; ++i)
    {
        const float current = extremums[i];
        if (search_max)
        {
            // Если найден локальный максимум, больший чем заданный порог.
            if ((current < prev) && (prev >= max_threthold))
            {
                search_max = false;
                ++counter;
            }
        }
        else
        {
            // Если найден локальный минимум, меньший чем заданный порог.
            if ((current > prev) && (prev < min_threshold))
                search_max = true;
        }

        prev  = current;
    }

    return counter;
}

static pair<vector<float>, vector<size_t>> findExtremums(const vector<float>& curvature)
{
    vector<float> extremums;
    vector<size_t> extremum_indexes;

    const size_t length = curvature.size();
    // Вычисляем первую производную в каждой точке функции кривизны.
    vector<float> derivative(length, 0.0);
    derivative[0] = curvature[1] - curvature[0];
    derivative[length - 1] = curvature[length - 1] - curvature[length - 2];
    for (size_t i = 1; i < length - 1; ++i)
    {
        derivative[i] = (curvature[i + 1] - curvature[i - 1]) / 2;
    }

    // Находим экстремумы с помощью пересечения нуля.
    for (size_t i = 0; i < length - 1; ++i)
    {
        if (derivative[i] * derivative[i + 1] > 0)
            continue;

        extremums.push_back(curvature[i]);
        extremum_indexes.push_back(i);
    }

    return pair(extremums, extremum_indexes);
}

std::optional<Hand> handDetector(const Contour& contour, float min_treshold, float max_trethold, int min_counter, int max_counter)
{

    vector<float> curvature;
    if (getCurvature(curvature, contour, 75) != 0)
        throw;

    const size_t length = curvature.size();
    if (length < 2)
        throw;

    auto extremums = findExtremums(curvature);
    vector<float> extremum_values = get<0>(extremums);
    vector<size_t> extremum_indexes = get<1>(extremums);

    int counter = countLocalMax(extremum_values, min_treshold, max_trethold);
    if ((counter < min_counter) || (counter > max_counter))
    {
        return {};
    }

    vector<size_t> fingers(9, 0);
    for (size_t i = 0; i < 9; ++i)
    {
        float max = 0.0;
        size_t index = 0;
        for (size_t j = i; j < extremum_values.size(); ++j)
        {
            if (extremum_values[j] < max)
                continue;

            // Отсеиваем максимумы, обнаруженные ранее
            bool local = false;
            for (size_t k = 0; k < i; ++k)
            {
                int difference = abs((int)fingers[k] - (int)extremum_indexes[j]);
                if (difference < 50)
                {
                    local = true;
                    break;
                }
            }

            if (local == true)
                continue;

            max = extremum_values[j];
            index = j;
        }

        fingers[i] = extremum_indexes[index];
        std::swap(extremum_values[index], extremum_values[i]);
        std::swap(extremum_indexes[index], extremum_indexes[i]);
    }

    std::sort(fingers.begin(), fingers.end());
    vector<Point2i> points = contour.getContourPoints(fingers);
    return Hand(points);
}
