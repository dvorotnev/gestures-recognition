/*
    Реализация класса руки.
*/

#include <opencv2/video/video.hpp>
#include <algorithm>

#include <Hand.h>

using namespace std;
using namespace cv;

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

    vector<Point2i> points = max_points;
    if (first_difference < second_difference)
        reverse(points.begin(), points.end());

    vector<Point2i> fingers(points);
    fingers.resize(10);

    // Проекция на линию начала пальцев.
    Point2i mid_point = projection(points[3], points[7], points[5]);
    // Средний палец.
    fingers[5] = midPoint(points[3], mid_point);
    // Безымянный палец.
    fingers[7] = midPoint(mid_point, points[7]);
    // Мизинец.
    fingers[9] = points[7];

    return fingers;
}

// Функция заполнения структуры пальца.
static void fillFinger(Finger& finger, const Point2i& start, const Point2i& peak, double length)
{
    finger.start = start;
    finger.peak = peak;
    finger.length = length;
    finger.is_bent = false;
    finger.status_changed = false;
    return;
}

Hand::Hand(const vector<Point2i>& points)
{
    if (points.size() != 9)
        throw;

    vector<Point2i> fingers_points = getFingers(points);

    // Большой палец
    fillFinger(fingers_[0], fingers_points[1], fingers_points[0], norm(fingers_points[0] - fingers_points[1]));
    // Указательный палец
    fillFinger(fingers_[1], fingers_points[3], fingers_points[2], norm(fingers_points[2] - fingers_points[3]));
    // Средний палец
    fillFinger(fingers_[2], fingers_points[5], fingers_points[4], norm(fingers_points[4] - fingers_points[5]));
    // Безымянный палец
    fillFinger(fingers_[3], fingers_points[7], fingers_points[6], norm(fingers_points[6] - fingers_points[7]));
    // Мизинец
    fillFinger(fingers_[4], fingers_points[9], fingers_points[8], norm(fingers_points[8] - fingers_points[9]));

    midle_point_ = (fingers_points[0] == points[0]) ? points[5] : points[3];
}

Point2i Hand::getWrist() const
{
    return fingers_[2].start + 1.1 * (fingers_[2].start - fingers_[2].peak);
}

const Finger* Hand::getHandFingers() const
{
    return fingers_;
}

void Hand::print(Mat& image) const
{
    Scalar red(0, 0, 255);
    circle(image, fingers_[0].peak, 6, red, 2);
    circle(image, fingers_[0].start, 6, red, 2);

    circle(image, fingers_[1].peak, 5, red, 2);
    circle(image, fingers_[1].start, 5, red, 2);

    circle(image, fingers_[2].peak, 4, red, 2);
    circle(image, fingers_[2].start, 4, red, 2);

    circle(image, fingers_[3].peak, 3, red, 2);
    circle(image, fingers_[3].start, 3, red, 2);

    circle(image, fingers_[4].peak, 2, red, 2);
    circle(image, fingers_[4].start, 2, red, 2);

    circle(image, midle_point_, 1, red, 2);
    circle(image, getWrist(), 5, red, 10);

    return;
}

cv::Rect2i Hand::getBoundingBox()
{
    Point2i wrist = getWrist();
    int left = wrist.x;
    int top = wrist.y;
    int right = wrist.x;
    int bottom = wrist.y;

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

// Функция, обновляющая состояние пальцев.
static void updateFingersStatus(Finger fingers[])
{
    for (int i = 0; i < 5; ++i)
    {
        Finger& finger = fingers[i];
        double new_length = norm(finger.start - finger.peak);
        if (new_length > finger.length)
        {
            finger.length = new_length;
        }

        if (!finger.is_bent && (new_length < finger.length * 0.5))
        {
            finger.is_bent = true;
            finger.status_changed = true;
        }
        else if (finger.is_bent && (new_length >= finger.length * 0.5))
        {
            finger.is_bent = false;
            finger.status_changed = true;
        }
        else if (finger.status_changed)
        {
            finger.status_changed = false;
        }
    }
}

int Hand::update(const vector<Mat>& prevPyr, const vector<Mat>& nextPyr)
{
    vector<Point2f> prev_pts = {
        fingers_[0].peak, fingers_[0].start,
        fingers_[1].peak, fingers_[1].start,
        fingers_[2].peak, fingers_[3].peak,
        fingers_[4].peak, fingers_[4].start,
        midle_point_
    };

    vector<Point2f> next_pts(prev_pts);
    vector<uchar> status(10, 0);
    int levels = min(prevPyr.size(), nextPyr.size());

    calcOpticalFlowPyrLK(prevPyr, nextPyr, prev_pts, next_pts, status, noArray(), Size(31, 31), levels);
    for (const auto& elem : status)
    {
        if (elem != 1)
            return -1;
    }

    fingers_[0].peak = next_pts[0];
    fingers_[0].start = next_pts[1];
    fingers_[1].peak = next_pts[2];
    fingers_[1].start = next_pts[3];
    fingers_[4].peak = next_pts[6];
    fingers_[4].start = next_pts[7];
    midle_point_ = next_pts[8];

    // Проекция на линию начала пальцев.
    Point2i mid_point = projection(fingers_[1].start, fingers_[4].start, midle_point_);
    // Средний палец.
    fingers_[2].start = midPoint(next_pts[3], mid_point);
    fingers_[2].peak = next_pts[4];

    // Безымянный палец.
    fingers_[3].start = midPoint(mid_point, next_pts[7]);
    fingers_[3].peak = next_pts[5];

    updateFingersStatus(fingers_);
    return 0;
}
