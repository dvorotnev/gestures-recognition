#include "Skin_threthold.h"

#include <highgui.hpp>

using namespace cv;

void main()
{
    VideoCapture video(0);
    Mat frame;
    namedWindow("Original");

    Mat skinRGB;
    namedWindow("RGB");
    Mat skinYCrCb;
    namedWindow("YCrCb");
    Mat skinHSV;
    namedWindow("HSV");
    Mat skinHSL;
    namedWindow("HSL");

    while (true)
    {
        video >> frame;
        imshow("Original", frame);

        thresholdRGB(frame, skinRGB);
        imshow("RGB", skinRGB);
        thresholdYCrCb(frame, skinYCrCb);
        imshow("YCrCb", skinYCrCb);
        thresholdHSV(frame, skinHSV);
        imshow("HSV", skinHSV);
        thresholdHSL(frame, skinHSL);
        imshow("HSL", skinHSL);

        char c = waitKey(30);
        if (c == 27)
            break;
    }

    destroyAllWindows();
    return;
}
