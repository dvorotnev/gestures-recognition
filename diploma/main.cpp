#include <highgui.hpp>
#include <video\video.hpp>

#include "..\Motion detection\ViBe\ViBe.h"
#include "..\deletenoise.h"
#include "..\Contour.h"
#include "..\CorrectionOfExposition.h"

#ifdef ___DEBUG___
#include <string>
#include <fstream>

unsigned int debug_counter = 1;
char path[] = "d:\\Dropbox\\Диплом\\";
#endif

// Отключаем предупреждение компилятора о константных условиях
// в циклах, чтобы использовать бесконечные циклы.
__pragma(warning(disable:4127));

using namespace std;
using namespace cv;

void main()
{
    //VideoCapture video("..\\test_videos\\campus_raw.avi");
    VideoCapture video("d:\\Работа\\Диплом\\test videos\\test18.avi");
    ViBe motion(20, 20, 2, 15);

    namedWindow("Video");

    Mat frame;
    for (int i = 0; i < 100; i++)
    {
        video >> frame;
        if (frame.empty()) continue;
        imshow("Video", frame);
        waitKey(33);
    }

    Mat bg_image(frame.size(), CV_8UC3);
    Mat fgmask(frame.size(), CV_8UC1);
    Mat marked_image(frame.size(), CV_32S);
    Mat contours_image(frame.size(), CV_8UC1);

    while (true)
    {
        video >> frame;
        if (frame.empty())
            break;
        imshow("Video", frame);
#if ___DEBUG___
        imwrite(String(path) + "res_input\\" + std::to_string(debug_counter) + ".png", frame);
#endif

        motion.getBackgroundImage(bg_image);
        if (!bg_image.empty())
        {
            correctionOfExposition(fgmask, bg_image, frame);
            imshow("Background", bg_image);
        }

        motion.apply(frame, fgmask, 1/15);
        imshow("Motion", fgmask);
#if ___DEBUG___
        imwrite(String(path) + "res_exposition\\" + std::to_string(debug_counter) + ".png", frame);
        imwrite(String(path) + "res_background\\" + std::to_string(debug_counter) + ".png", bg_image);
        imwrite(String(path) + "res_motion\\" + std::to_string(debug_counter) + ".png", fgmask);
#endif

        deleteNoise(fgmask, marked_image, 100);
        imshow("Noise", fgmask);
#if ___DEBUG___
        imwrite(String(path) + "res_noise\\" + std::to_string(debug_counter) + ".png", fgmask);
#endif

        ContourMapMorph contours;
        contours.findContours(fgmask);
        contours.sortContours();
        contours.printAllContours(contours_image);
        imshow("Contours", contours_image);

#if ___DEBUG___
        std::ofstream file(String(path) + "res_contours\\" + std::to_string(debug_counter) + ".txt");
        for (int y = 0; y < contours_image.rows; ++y)
        {
            uchar* ptr = contours_image.ptr(y);
            for (int x = 0; x < contours_image.cols; ++x)
            {
                file << (int)ptr[x] <<" ";
            }
            file << endl;
        }
#endif

        int c = waitKey(30);
        if (c == 27) break;
#if ___DEBUG___
        ++debug_counter;
#endif
    }

    frame.release();
    fgmask.release();
    marked_image.release();
    destroyAllWindows();
}
