#include "Skin_threthold.h"

#include <highgui.hpp>

using namespace cv;

void main()
{
    VideoCapture video(0);
    Mat frame;
    Mat skinRGB;
    Mat skinYCrCb;
    Mat skinHSV;
    Mat skinHSL;

    namedWindow("Original");
    namedWindow("RGB");
    namedWindow("YCrCb");
    namedWindow("HSV");
    namedWindow("HSL");

    int min_R = 95, max_R = 255;
    int min_G = 40, max_G = 255;
    int min_B = 20, max_B = 255;

    namedWindow("RGB Options");
    createTrackbar("min_R", "RGB Options", &min_R, 255);
    createTrackbar("max_R", "RGB Options", &max_R, 255);
    createTrackbar("min_G", "RGB Options", &min_G, 255);
    createTrackbar("max_G", "RGB Options", &max_G, 255);
    createTrackbar("min_B", "RGB Options", &min_B, 255);
    createTrackbar("max_B", "RGB Options", &max_B, 255);

    int min_Y = 49, max_Y = 255;
    int min_Cr = 130, max_Cr = 186;
    int min_Cb = 79, max_Cb = 136;

    namedWindow("YCrCb Options");
    createTrackbar("min_Y", "YCrCb Options", &min_Y, 255);
    createTrackbar("max_Y", "YCrCb Options", &max_Y, 255);
    createTrackbar("min_Cr", "YCrCb Options", &min_Cr, 255);
    createTrackbar("max_Cr", "YCrCb Options", &max_Cr, 255);
    createTrackbar("min_Cb", "YCrCb Options", &min_Cb, 255);
    createTrackbar("max_Cb", "YCrCb Options", &max_Cb, 255);

    int min_HSV_H = 0, max_HSV_H = 21;
    int min_HSV_S = 44, max_HSV_S = 255;
    int min_HSV_V = 0, max_HSV_V = 255;

    namedWindow("HSV Options");
    createTrackbar("min_H", "HSV Options", &min_HSV_H, 255);
    createTrackbar("max_H", "HSV Options", &max_HSV_H, 255);
    createTrackbar("min_S", "HSV Options", &min_HSV_S, 255);
    createTrackbar("max_S", "HSV Options", &max_HSV_S, 255);
    createTrackbar("min_V", "HSV Options", &min_HSV_V, 255);
    createTrackbar("max_V", "HSV Options", &max_HSV_V, 255);

    int min_HSL_H = 0, max_HSL_H = 21;
    int min_HSL_S = 44, max_HSL_S = 255;
    int min_HSL_L = 49, max_HSL_L = 255;

    namedWindow("HSL Options");
    createTrackbar("min_H", "HSL Options", &min_HSL_H, 255);
    createTrackbar("max_H", "HSL Options", &max_HSL_H, 255);
    createTrackbar("min_S", "HSL Options", &min_HSL_S, 255);
    createTrackbar("max_S", "HSL Options", &max_HSL_S, 255);
    createTrackbar("min_L", "HSL Options", &min_HSL_L, 255);
    createTrackbar("max_L", "HSL Options", &max_HSL_L, 255);

    while (true)
    {
        video >> frame;
        imshow("Original", frame);

        thresholdRGB(frame, skinRGB, min_R, max_R, min_G, max_G, min_B, max_B);
        imshow("RGB", skinRGB);
        thresholdYCrCb(frame, skinYCrCb, min_Y, max_Y, min_Cr, max_Cr, min_Cb, max_Cb);
        imshow("YCrCb", skinYCrCb);
        thresholdHSV(frame, skinHSV, min_HSV_H, max_HSV_H, min_HSV_S, max_HSV_S, min_HSV_V, max_HSV_V);
        imshow("HSV", skinHSV);
        thresholdHSL(frame, skinHSL, min_HSL_H, max_HSL_H, min_HSL_S, max_HSL_S, min_HSL_L, max_HSL_L);
        imshow("HSL", skinHSL);

        char c = waitKey(30);
        if (c == 27)
            break;
    }

    destroyAllWindows();
    return;
}
