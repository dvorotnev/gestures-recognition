/*
    Реализация функции распознавания руки на основе анализа кривизны контура.
*/

#include <algorithm>

#include <handDetector.h>

using namespace std;
using namespace cv;

static Point2i midPoint(const Point2i& first, const Point2i second)
{
    int x = (first.x + second.x) / 2;
    int y = (first.y + second.y) / 2;
    return Point2i(x, y);
}

Hand::Hand(const vector<Point2i>& points)
{
    double first_difference = abs(norm(points[1] - points[2]) - norm(points[2] - points[3]));
    double second_difference = abs(norm(points[5] - points[6]) - norm(points[6] - points[7]));

    // Средний палец
    fingers[2].peak = points[4];
    fingers[2].start = midPoint(points[3], points[5]);
    fingers[2].length = norm(points[4] - fingers[2].start);

    if (first_difference > second_difference)
    {
        // Большой палец
        fingers[0].peak = points[0];
        fingers[0].start = points[1];
        fingers[0].length = norm(points[0] - points[1]);
        // Указательный палец
        fingers[1].peak = points[2];
        fingers[1].start = points[3];
        fingers[1].length = norm(points[2] - points[3]);
        // Безымянный палец
        fingers[3].peak = points[6];
        fingers[3].start = midPoint(points[5], points[7]);
        fingers[3].length = norm(points[6] - fingers[3].start);
        // Мизинец
        fingers[4].peak = points[8];
        fingers[4].start = points[7];
        fingers[4].length = norm(points[8] - points[7]);
    }
    else
    {
        // Большой палец
        fingers[0].peak = points[8];
        fingers[0].start = points[7];
        fingers[0].length = norm(points[8] - points[7]);
        // Указательный палец
        fingers[1].peak = points[6];
        fingers[1].start = points[5];
        fingers[1].length = norm(points[6] - points[5]);
        // Безымянный палец
        fingers[3].peak = points[2];
        fingers[3].start = midPoint(points[1], points[3]);
        fingers[3].length = norm(points[2] - fingers[3].start);
        // Мизинец
        fingers[4].peak = points[0];
        fingers[4].start = points[1];
        fingers[4].length = norm(points[0] - points[1]);
    }
}

static void printRing(Mat& image, const Point2i& center, int big_radius, int little_radius)
{
    for (int y = center.y - big_radius; y <= center.y + big_radius; ++y)
    {
        uchar* ptr = image.ptr(y);
        for (int x = center.x - big_radius; x <= center.x + big_radius; ++x)
        {
            int distance = (y - center.y) * (y - center.y);
            distance += (x - center.x) * (x - center.x);
            if (distance >= little_radius && distance <= big_radius)
                ptr[x] = 255;
        }
    }
}

void Hand::print(Mat& image)
{
    printRing(image, fingers[0].peak, 30, 20);
    printRing(image, fingers[0].start, 30, 20);

    printRing(image, fingers[1].peak, 25, 15);
    printRing(image, fingers[1].start, 25, 15);

    printRing(image, fingers[2].peak, 20, 10);
    printRing(image, fingers[2].start, 20, 10);

    printRing(image, fingers[3].peak, 15, 5);
    printRing(image, fingers[3].start, 15, 5);

    printRing(image, fingers[4].peak, 10, 0);
    printRing(image, fingers[4].start, 10, 0);

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

static int findExtremums(const vector<float>& curvature, vector<float>& extremums, vector<size_t>& extremum_indexes)
{
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

    return 0;
}

int handDetector(const Contour& contour, float min_treshold, float max_trethold, int min_counter, int max_counter, Hand& hand)
{
    vector<float> curvature;
    if (getCurvature(curvature, contour, 75) != 0)
        return -1;

    const size_t length = curvature.size();
    if (length < 2)
        return -1;

    vector<float> extremums;
    vector<size_t> extremum_indexes;
    findExtremums(curvature, extremums, extremum_indexes);

    int counter = countLocalMax(extremums, min_treshold, max_trethold);
    if ((counter < min_counter) || (counter > max_counter))
    {
        return 1;
    }

    vector<size_t> fingers(9, 0);
    for (size_t i = 0; i < 9; ++i)
    {
        float max = 0.0;
        size_t index = 0;
        for (size_t j = i; j < extremums.size(); ++j)
        {
            if (extremums[j] < max)
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

            max = extremums[j];
            index = j;
        }

        fingers[i] = extremum_indexes[index];
        std::swap(extremums[index], extremums[i]);
        std::swap(extremum_indexes[index], extremum_indexes[i]);
    }

    std::sort(fingers.begin(), fingers.end());
    vector<Point2i> points;
    contour.getContourPoints(points, fingers);
    hand = Hand(points);
    return 0;
}
