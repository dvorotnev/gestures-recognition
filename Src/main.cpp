#include <fstream>
#include <optional>
#include <opencv2\highgui.hpp>
#include <opencv2\video\video.hpp>

#include <ViBe_plus.h>
#include <deletenoise.h>
#include <Contour.h>
#include <CorrectionOfExposition.h>
#include <HandDetector.h>
#include <VideoSequenceCapture.h>
#include <Timer.h>
#include <Debug.h>

using namespace std;
using namespace cv;

const uchar ForeGround = 255;
const uchar Background = 0;

int main()
{
    Timer total_timer, exposition_timer, motion_timer, detector_timer, tracker_timer;
    VideoCapture video(0);
    //VideoSequenceCapture video("d:\\test_videos\\output2\\0.png");

    ViBe_plus motion(20, 20, 2, 15);

    namedWindow("Input");
    namedWindow("Background");
    namedWindow("Motion");
    namedWindow("Open");
    namedWindow("Tracker");

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
    Mat tracker_image(frame.size(), CV_8UC3);

    HandDetector hand_detector(15, 25, 7, 11);

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
        motion.apply(frame, fgmask, 1.0 / 15);
        motion_timer.stop();
        imageShow("Motion", fgmask);

        // Размыкание маски движущихся объектов.
        const uchar kernel_values[25] = { 1, 1, 1, 1, 1,
                                          1, 1, 1, 1, 1,
                                          1, 1, 1, 1, 1,
                                          1, 1, 1, 1, 1,
                                          1, 1, 1, 1, 1 };
        Matx <uchar, 5, 5> kernel_open(kernel_values);
        morphologyEx(fgmask, fgmask, MORPH_OPEN, kernel_open);
        imageShow("Open", fgmask);

        tracker_timer.start();
        hand_detector.trace(fgmask);
        imageShow("Tracker", tracker_image);
        tracker_timer.stop();

        detector_timer.start();
        hand_detector.detect(fgmask);
        detector_timer.stop();

        frame.copyTo(tracker_image);
        hand_detector.printHands(tracker_image);

        total_timer.stop();
        int c = waitKey(30);
        if (c == 27)
            break;
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
    time_log << "Hand tracking: " << tracker_timer.getTime() << " sec." << endl;
    time_log << "Hand detection: " << detector_timer.getTime() << " sec." << endl;
    time_log.close();

    return 0;
}
