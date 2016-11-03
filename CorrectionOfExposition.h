/*
    ќбъ€вление функции коррекции €ркости.
*/

#include <highgui.hpp>

/*
    ‘ункци€ подстраивает €ркость текущего кадра под €ркость фонового изображени€.

    ¬ходные параметры:
    SegmentationMask - бинарное изображение с отмеченными движущимис€ объектами
                       с предыдущего кадра.
    BackgroundImage  - цветное изображение текущего фона.
    CurrentImage     - текущий кадр в формате Y, €ркость которого будет
                       изменена в соответствии с фоновым изображением.
*/
void correctionOfExposition(const cv::Mat& segmentationMask,
                            const cv::Mat& backgroundImage,
                            cv::Mat& currentImage);
