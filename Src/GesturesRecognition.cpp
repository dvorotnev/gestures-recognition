/*
    Реализация класса алгоритма распознавания жестов.
*/

#include <opencv2/imgproc.hpp>

#include <GesturesRecognition.h>

using namespace std;
using namespace cv;

void GesturesRecognition::printClicks(Mat& image) const
{
    for (const auto& click : clicks_)
    {
        drawMarker(image, click, Scalar(0, 255, 0));
    }
}

void GesturesRecognition::apply(const list<Hand>& hands)
{
    for (const auto& hand : hands)
    {
        const Finger* fingers = hand.getHandFingers();
        for (int i = 0; i < 5; ++i)
        {
            const Finger& finger = fingers[i];
            if (!finger.status_changed || !finger.is_bent)
                continue;

            clicks_.push_back(finger.peak);
        }
    }
}
