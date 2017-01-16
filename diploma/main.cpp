#include <fstream>
#include <highgui.hpp>
#include <video\video.hpp>

#include "..\Motion detection\ViBe_plus\ViBe_plus.h"
#include "..\deletenoise.h"
#include "..\Contour.h"
#include "..\CorrectionOfExposition.h"
#include "..\handDetector.h"
#include "..\VideoSequenceCapture.h"
#include "..\Timer.h"
#include "..\Debug.h"

// Отключаем предупреждение компилятора о константных условиях
// в циклах, чтобы использовать бесконечные циклы.
__pragma(warning(disable:4127));

using namespace std;
using namespace cv;

void main()
{
    Timer total_timer, exposition_timer, motion_timer, contours_timer, curvature_timer, detector_timer;
    VideoCapture video(0);
    //VideoSequenceCapture video("d:\\test videos\\output2\\0.png");

    ViBe_plus motion(20, 20, 2, 15);

    namedWindow("Input");
    namedWindow("Background");
    namedWindow("Motion");
    namedWindow("Contours");
    namedWindow("Curvature");
    namedWindow("Output");

    // Пропускаем первые кадры, чтобы стабилизировалась 
    // яркость на изображениях, полученных с камеры.
    Mat frame;
    for (int i = 0; i < 20; i++)
    {
        video >> frame;
        if (frame.empty()) continue;
        binaryImageShow("Input", frame);
        waitKey(30);
    }

    Mat bg_image(frame.size(), CV_8UC3);
    Mat fgmask(frame.size(), CV_8UC1);
    Mat contours_image(frame.size(), CV_8UC1);
    Mat result_image(frame.size(), CV_8UC1);

    while (true)
    {
        // Получение входного изображения.
        total_timer.start();
        video >> frame;
        if (frame.empty())
            break;
        binaryImageShow("Input", frame);

        // Коррекция яркости.
        motion.getBackgroundImage(bg_image);
        if (!bg_image.empty())
        {
            exposition_timer.start();
            correctionOfExposition(fgmask, bg_image, frame);
            exposition_timer.stop();
            binaryImageShow("Background", bg_image);
        }

        // Выделение движения.
        motion_timer.start();
        motion.apply(frame, fgmask, 1.0/15);
        motion_timer.stop();
        binaryImageShow("Motion", fgmask);

        // Извлечение контуров.
        contours_timer.start();
        ContourMapMorph contours;
        contours.extractContours(fgmask);
        contours.sortContours();
        contours.printAllContours(contours_image);
        contours_timer.stop();
        imageShow("Contours", contours_image);

        result_image.setTo(0);
        for (int i = 0; i < contours.getNumberOfContours(); ++i)
        {
            // Вычисление кривизны контура.
            curvature_timer.start();
            vector<float> curvature;
            contours.getCurvature(curvature, 75, i);
            curvature_timer.stop();

            if ((i == 0) && (curvature.size() > 0))
                curvatureShow("Curvature", curvature);

            // Распознавание руки.
            detector_timer.start();
            int hand = handDetector(curvature, 15, 25, 7, 11);
            detector_timer.stop();
            if (hand == 1)
                contours.printContour(result_image, i);
        }

        binaryImageShow("Output", result_image);

        total_timer.stop();
        int c = waitKey(30);
        if (c == 27) break;
    }

    frame.release();
    fgmask.release();
    destroyAllWindows();

    // Записываем время работы программы.
    ofstream time_log("Time.txt");
    time_log << "Program time:" << endl;
    time_log << "Total time: " << total_timer.getTime() << " sec." << endl;
    time_log << "Correction of exposition: " << exposition_timer.getTime() << " sec." << endl;
    time_log << "Motion detection: " << motion_timer.getTime() << " sec." << endl;
    time_log << "Contours: " << contours_timer.getTime() << " sec." << endl;
    time_log << "Curvature: " << curvature_timer.getTime() << " sec." << endl;
    time_log << "Hand detection: " << detector_timer.getTime() << " sec." << endl;
    time_log.close();

    return;
}
