#include <fstream>
#include <highgui.hpp>
#include <video\video.hpp>

#include "..\graphUtils\GraphUtils.h"

#include "..\Motion detection\ViBe_plus\ViBe_plus.h"
#include "..\deletenoise.h"
#include "..\Contour.h"
#include "..\CorrectionOfExposition.h"
#include "..\VideoSequenceCapture.h"
#include "..\Timer.h"

#define ___DEBUG___ 0

#if ___DEBUG___
#include <string>

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
    Timer total_timer, exposition_timer, motion_timer, contours_timer, curvature_timer;
    VideoCapture video(0);
    //VideoSequenceCapture video("d:\\test videos\\output2\\0.png");
    ViBe_plus motion(20, 20, 2, 15);

    namedWindow("Video");

    // Пропускаем первые кадры, чтобы стабилизировалась 
    // яркость на изображениях, полученных с камеры.
    Mat frame;
    for (int i = 0; i < 20; i++)
    {
        video >> frame;
        if (frame.empty()) continue;
        imshow("Video", frame);
        waitKey(30);
    }

    Mat bg_image(frame.size(), CV_8UC3);
    Mat fgmask(frame.size(), CV_8UC1);
    Mat contours_image(frame.size(), CV_8UC1);

    while (true)
    {
        total_timer.start();
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
            exposition_timer.start();
            correctionOfExposition(fgmask, bg_image, frame);
            exposition_timer.stop();
            imshow("Background", bg_image);
        }

        motion_timer.start();
        motion.apply(frame, fgmask, 1.0/15);
        motion_timer.stop();
        imshow("Motion", fgmask);
#if ___DEBUG___
        imwrite(String(path) + "res_exposition\\" + std::to_string(debug_counter) + ".png", frame);
        imwrite(String(path) + "res_background\\" + std::to_string(debug_counter) + ".png", bg_image);
        imwrite(String(path) + "res_motion\\" + std::to_string(debug_counter) + ".png", fgmask);
#endif

        contours_timer.start();
        ContourMapMorph contours;
        contours.extractContours(fgmask);
        contours.sortContours();
        contours.printAllContours(contours_image);
        contours_timer.stop();
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
        file.close();
#endif

        curvature_timer.start();
        vector<float> curvature;
        contours.getCurvature(curvature, 100, 0);
        curvature_timer.stop();
        if (curvature.size() > 0)
            showFloatGraph("Curvature", &curvature[0], (int)curvature.size(), 1);

#if ___DEBUG___
        file.open(String(path) + "res_curvature\\" + std::to_string(debug_counter) + ".txt");

        for (int i = 0; i < curvature.size(); ++i)
        {
            file << curvature[i] << " ";
        }
        file << endl;
        file.close();
#endif

        total_timer.stop();
        int c = waitKey(30);
        if (c == 27) break;
#if ___DEBUG___
        ++debug_counter;
#endif
    }

    frame.release();
    fgmask.release();
    destroyAllWindows();

    ofstream time_log("Time.txt");
    time_log << "Program time:" << endl;
    time_log << "Total time: " << total_timer.getTime() << " sec." << endl;
    time_log << "Correction of exposition: " << exposition_timer.getTime() << " sec." << endl;
    time_log << "Motion detection: " << motion_timer.getTime() << " sec." << endl;
    time_log << "Contours: " << contours_timer.getTime() << " sec." << endl;
    time_log << "Curvature: " << curvature_timer.getTime() << " sec." << endl;
    time_log.close();

    return;
}
