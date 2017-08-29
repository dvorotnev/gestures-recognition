#include <fstream>
#include <opencv2\highgui.hpp>
#include <opencv2\video\video.hpp>

#include <ViBe_plus.h>
#include <deletenoise.h>
#include <Contour.h>
#include <CorrectionOfExposition.h>
#include <handDetector.h>
#include <VideoSequenceCapture.h>
#include <Timer.h>
#include <Debug.h>

using namespace std;
using namespace cv;

const uchar ForeGround = 255;

int main()
{
    Timer total_timer, exposition_timer, motion_timer, contours_timer, detector_timer;
    VideoCapture video(0);
    //VideoSequenceCapture video("d:\\test videos\\output2\\0.png");

    ViBe_plus motion(20, 20, 2, 15);

    namedWindow("Input");
    namedWindow("Background");
    namedWindow("Motion");
    namedWindow("Contours");
    namedWindow("Output");

    // Пропускаем первые кадры, чтобы стабилизировалась
    // яркость на изображениях, полученных с камеры.
    Mat frame;
    for (int i = 0; i < 20; i++)
    {
        video >> frame;
        if (frame.empty()) continue;
        imageShow("Input", frame);
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
        imageShow("Input", frame);

        // Коррекция яркости.
        motion.getBackgroundImage(bg_image);
        if (!bg_image.empty())
        {
            exposition_timer.start();
            correctionOfExposition(fgmask, bg_image, frame);
            exposition_timer.stop();
            imageShow("Background", bg_image);
        }

        // Выделение движения.
        motion_timer.start();
        motion.apply(frame, fgmask, 1.0/15);
        motion_timer.stop();
        imageShow("Motion", fgmask);

        // Извлечение контуров.
        contours_timer.start();
        Matx <uchar, 3, 3> kernel_open = { 1, 1, 1,
                                           1, 1, 1,
                                           1, 1, 1 };
        morphologyEx(fgmask, fgmask, MORPH_OPEN, kernel_open);
        imageWrite("Open", fgmask);
        vector<Contour> contours;
        extractContours(fgmask, contours);
        sortContours(contours);
        printContours(contours_image, contours);
        contours_timer.stop();
        contoursShow("Contours", contours_image);

        result_image.setTo(0);
        for (size_t i = 0; i < contours.size(); ++i)
        {
            // Распознавание руки.
            detector_timer.start();
            Hand hand;
            int is_hand = handDetector(contours[i], 15, 25, 7, 11, hand);
            detector_timer.stop();
            if (is_hand == 0)
            {
                contours[i].printContour(result_image, ForeGround);
                hand.print(result_image);
            }
        }

        imageShow("Output", result_image);

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
    time_log << "Hand detection: " << detector_timer.getTime() << " sec." << endl;
    time_log.close();

    return 0;
}
