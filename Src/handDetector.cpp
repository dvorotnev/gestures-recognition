/*
    Реализация функции распознавания руки на основе анализа кривизны контура.
*/

#include <algorithm>

#include <handDetector.h>

using namespace std;
using namespace cv;

// Функция заполнения структуры пальца.
static void fillFinger(Finger& finger, const Point2i& start, const Point2i& peak, double length)
{
    finger.start = start;
    finger.peak = peak;
    finger.length = length;
    return;
}

Hand::Hand(const vector<Point2i>& points)
{
    if (points.size() != 10)
        throw;

    // Большой палец
    fillFinger(fingers_[0], points[1], points[0], norm(points[0] - points[1]));
    // Указательный палец
    fillFinger(fingers_[1], points[3], points[2], norm(points[2] - points[3]));
    // Средний палец
    fillFinger(fingers_[2], points[5], points[4], norm(points[4] - points[5]));
    // Безымянный палец
    fillFinger(fingers_[3], points[7], points[6], norm(points[6] - points[7]));
    // Мизинец
    fillFinger(fingers_[4], points[9], points[8], norm(points[8] - points[9]));

    wrist_ = fingers_[2].start + 1.1 * (fingers_[2].start - fingers_[2].peak);
}

// Отрисовка кольца вокруг заданной точки.
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

cv::Rect2i Hand::getBoundingBox()
{
    int left = wrist_.x;
    int top = wrist_.y;
    int right = wrist_.x;
    int bottom = wrist_.y;

    for (const auto& elem : fingers_)
    {
        left = min(left, min(elem.start.x, elem.peak.x));
        right = max(right, max(elem.start.x, elem.peak.x));
        top = min(top, min(elem.start.y, elem.peak.y));
        bottom = max(bottom, max(elem.start.y, elem.peak.y));
    }

    int width = right - left;
    int height = bottom - top;
    left = max(0.0, left - width * 0.1);
    top = max(0.0, top - height * 0.1);
    Rect2i result(left, top, width * 1.2, height * 1.2);
    return result;
}

// Подсчёт локальных максимумов, чередующихся локальными минимумами.
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

// Поиск точек экстремума и их индексов в векторе кривизны.
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

// Поиск индексов максимумов.
static vector<size_t> findMaxIndexes(vector<float>& values, vector<size_t>& indexes)
{
    vector<size_t> max_indexes(9, 0);
    for (size_t i = 0; i < 9; ++i)
    {
        float max = 0.0;
        size_t index = 0;
        for (size_t j = i; j < values.size(); ++j)
        {
            if (values[j] < max)
                continue;

            // Отсеиваем максимумы, обнаруженные ранее
            bool local = false;
            for (size_t k = 0; k < i; ++k)
            {
                int difference = abs((int)max_indexes[k] - (int)indexes[j]);
                if (difference < 50)
                {
                    local = true;
                    break;
                }
            }

            if (local == true)
                continue;

            max = values[j];
            index = j;
        }

        max_indexes[i] = indexes[index];
        std::swap(values[index], values[i]);
        std::swap(indexes[index], indexes[i]);
    }

    std::sort(max_indexes.begin(), max_indexes.end());
    return max_indexes;
}

// Функция вычисления средней точки между двумя заданными.
static Point2i midPoint(const Point2i& first, const Point2i& second)
{
    int x = (first.x + second.x) / 2;
    int y = (first.y + second.y) / 2;
    return Point2i(x, y);
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

// Вычисление положения точек пальцев с использованием точек максимума кривизны контура.
static vector<Point2i> getFingers(const vector<Point2i>& max_points)
{
    double first_difference = abs(norm(max_points[1] - max_points[2]) - norm(max_points[2] - max_points[3]));
    double second_difference = abs(norm(max_points[5] - max_points[6]) - norm(max_points[6] - max_points[7]));

    vector<Point2i> fingers = max_points;
    if (first_difference < second_difference)
        reverse(fingers.begin(), fingers.end());

    fingers.resize(10);

    // Средний палец.
    Point2i mid_point = midPoint(max_points[3], max_points[5]);
    // Проекция на линию начала пальцев.
    fingers[5] = projection(max_points[3], max_points[7], mid_point);

    // Безымянный палец.
    mid_point = midPoint(max_points[5], max_points[7]);
    fingers[7] = projection(max_points[3], max_points[7], mid_point);

    // Мизинец.
    fingers[9] = max_points[7];

    return fingers;
}

// Проверка соотношений длин пальцев.
static bool checkFingersLength(const vector<Point2i>& fingers)
{
    if (fingers.size() != 10)
        return false;

    double length[5] = { 0.0 };
    for (int i = 0; i < 5; ++i)
    {
        length[i] = norm(fingers[2 * i] - fingers[2 * i + 1]);
    }

    // Указательный и безымянный пальцы должны быть примерно одной длины.
    double difference = abs(length[1] - length[3]) / length[1];
    if (difference > 0.1)
        return false;

    // Большой палец короче, чем указательный, средний и безымянный пальцы.
    if (length[0] > length[1] || length[0] > length[2] || length[0] > length[3])
        return false;

    // Мизинец короче, чем указательный, средний и безымянный пальцы.
    if (length[4] > length[1] || length[4] > length[2] || length[4] > length[3])
        return false;

    // Указательный и безымянный пальцы короче среднего пальца.
    if (length[1] > length[2] || length[3] > length[2])
        return false;

    return true;
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

    vector<size_t> max = findMaxIndexes(extremum_values, extremum_indexes);
    vector<Point2i> max_points = contour.getContourPoints(max);
    vector<Point2i> fingers = getFingers(max_points);

    if (!checkFingersLength(fingers))
    {
        return {};
    }

    return Hand(fingers);
}
