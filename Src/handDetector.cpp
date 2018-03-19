/*
    Реализация функции распознавания руки на основе анализа кривизны контура.
*/

#include <algorithm>
#include <optional>

#include <Contour.h>
#include <HandDetector.h>

using namespace std;
using namespace cv;

const uchar ForeGround = 255;
const uchar Background = 0;

HandDetector::HandDetector(float min_threshold, float max_threthold, int min_counter, int max_counter)
: hands_(), mask_(), prev_pyr_(), min_threshold_(min_threshold), max_threthold_(max_threthold),
min_counter_(min_counter), max_counter_(max_counter)
{
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

// Проверка соотношений длин пальцев.
static bool checkFingersLength(const Finger fingers[5])
{
    // Указательный и безымянный пальцы должны быть примерно одной длины.
    double difference = abs(fingers[1].length - fingers[3].length) / fingers[1].length;
    if (difference > 0.1)
        return false;

    // Большой палец короче, чем указательный, средний и безымянный пальцы.
    if (fingers[0].length > fingers[1].length ||
        fingers[0].length > fingers[2].length ||
        fingers[0].length > fingers[3].length)
    {
        return false;
    }

    // Мизинец короче, чем указательный, средний и безымянный пальцы.
    if (fingers[4].length > fingers[1].length ||
        fingers[4].length > fingers[2].length ||
        fingers[4].length > fingers[3].length)
    {
        return false;
    }

    // Указательный и безымянный пальцы короче среднего пальца.
    if (fingers[1].length > fingers[2].length ||
        fingers[3].length > fingers[2].length)
    {
        return false;
    }

    return true;
}

// Функция на основании анализа кривизны контура вычисляет, является ли контур рукой.
static std::optional<Hand> handDetector(const Contour& contour, float min_treshold, float max_trethold, int min_counter, int max_counter)
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
    Hand hand(max_points);

    if (!checkFingersLength(hand.getHandFingers()))
    {
        return {};
    }

    return hand;
}

void HandDetector::trace(InputArray BinaryImage)
{
    Mat image = BinaryImage.getMat();
    vector<Mat> next_pyr;
    buildOpticalFlowPyramid(image, next_pyr, Size(31, 31), 1);

    if (prev_pyr_.empty())
    {
        prev_pyr_ = move(next_pyr);
        return;
    }

    for (auto hand = hands_.begin(); hand != hands_.end(); )
    {
        int status = hand->update(prev_pyr_, next_pyr);
        // Обработка пропадания руки.
        if (status == -1)
        {
            auto to_remove = hand;
            ++hand;
            hands_.erase(to_remove);
            continue;
        }

        ++hand;
    }

    prev_pyr_ = move(next_pyr);
    return;
}

void HandDetector::detect(InputArray BinaryImage)
{
    Mat image = BinaryImage.getMat();
    updateMask(image.size());

    // Извлечение контуров.
    vector<Contour> contours = extractContours(image, mask_);
    sortContours(contours);

    for (size_t i = 0; i < contours.size(); ++i)
    {
        // Распознавание руки.
        optional<Hand> hand = handDetector(contours[i], min_threshold_, max_threthold_, min_counter_, max_counter_);
        if (hand)
            hands_.push_back(*hand);
    }

    return;
}

void HandDetector::printHands(InputArray Image) const
{
    Mat image = Image.getMat();
    for (const auto& hand : hands_)
    {
        hand.print(image);
    }
}

const list<Hand>* HandDetector::getHands() const
{
    return &hands_;
}

void HandDetector::updateMask(Size size)
{
    mask_.create(size, CV_8UC1);
    mask_.setTo(ForeGround);
    for (auto& hand : hands_)
    {
        Rect2i box = hand.getBoundingBox();
        rectangle(mask_, box, Background, FILLED);
    }
}
