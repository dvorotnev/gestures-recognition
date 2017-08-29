/*
    Модуль последовательного воспроизведения кадров видеопоследовательности.
*/

#ifndef __VIDEOSEQUENCECAPTURE_H__
#define __VIDEOSEQUENCECAPTURE_H__

#include <opencv2\highgui.hpp>

class VideoSequenceCapture
{
public:
    // В качестве параметра передаётся путь к первому кадру видеопоследовательности.
    VideoSequenceCapture(const cv::String& filename);

    // Возвращает следующий кадр видеопоследовательности.
    void operator >> (cv::Mat& image);
private:
    cv::String directory; // Имя директории, в которой содержится видеопоследовательность.
    cv::String name; // Имя видеопоследовательности.
    cv::String extension; // Расширение файлов видеопоследовательности.
    int counter; // Счётчик текущего кадра.
};

#endif // __VIDEOSEQUENCECAPTURE_H__
