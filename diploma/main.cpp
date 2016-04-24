#include <highgui.hpp>
#include <video\video.hpp>
#include "..\Motion detection\ViBe\ViBe.h"
#include <stdio.h>
// Отключаем предупреждение компилятора о константных условиях
// в циклах, чтобы использовать бесконечные циклы.
__pragma(warning(disable:4127));

using namespace cv;

void main()
{
    VideoCapture video("..\\test_videos\\campus_raw.avi");

    ViBe motion(20, 20, 2, 15);

    namedWindow("Video");

    Mat frame;
    for (int i = 0; i < 10; i++)
    {
        video >> frame;
        if (frame.empty())
        {
            std::printf("Skip\n");
            //continue;
        }
        imshow("Video", frame);
        waitKey(33);
    }

    Mat fgmask;

    while (true)
    {
        video >> frame;
        imshow("Video", frame);

        motion.apply(frame, fgmask, 1/15);
        imshow("Motion", fgmask);

        int c = waitKey(30);
        if (c == 27) break;
    }

    frame.release();
    fgmask.release();
    destroyAllWindows();
}
