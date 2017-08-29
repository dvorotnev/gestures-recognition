/*
    Объявление функции коррекции яркости.
*/

#include <opencv2\highgui.hpp>

/*
    Функция подстраивает яркость текущего кадра под яркость фонового изображения.

    Входные параметры:
    SegmentationMask - бинарное изображение с отмеченными движущимися объектами
                       с предыдущего кадра.
    BackgroundImage  - цветное изображение текущего фона.
    CurrentImage     - текущий кадр в формате Y, яркость которого будет
                       изменена в соответствии с фоновым изображением.
*/
void correctionOfExposition(const cv::Mat& segmentationMask,
                            const cv::Mat& backgroundImage,
                            cv::Mat& currentImage);
