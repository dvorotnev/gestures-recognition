/*
Реализация алгоритма поиска контуров на бинарном изображении.
*/

#include <Contour.h>

using namespace std;
using namespace cv;

const uchar Background = 0;
const uchar ForeGround = 255;

vector<float> getCurvature(const vector<Point2i>& contour, const Size& image_size, const int chord_length)
{
    size_t length = contour.size();
    if (length < 4)
        throw;

    vector<float> curvature(length, 0.0);
    for (size_t i = 0; i < length; ++i)
    {
        // Вычисляем координаты концов хорды.
        const int first_point = i - chord_length + 2;
        const int last_point = i + 1;

        // Смещаем хорду по контуру.
        for (int shift = 0; shift <= chord_length - 3; ++shift)
        {
            int start_index = first_point + shift;
            if (start_index < 0)
                start_index = 0;
            else if (start_index >= (int)length)
                start_index = (int)length - 1;

            int end_index = last_point + shift;
            if (end_index >= (int)length)
                end_index = (int)length - 1;

            // Вычисляем расстояние от точки до хорды.
            const Point2i chord = contour[end_index] - contour[start_index];
            const Point2i point_to_chord = contour[i] - contour[start_index];

            // Отбрасываем хорды, которые касаются края изображения
            if (contour[end_index].x < 10 || contour[end_index].x >= (image_size.width - 10) ||
                contour[end_index].y < 10 || contour[end_index].y >= (image_size.height - 10) ||
                contour[start_index].x < 10 || contour[start_index].x >= (image_size.width - 10) ||
                contour[start_index].y < 10 || contour[start_index].y >= (image_size.height - 10))
            {
                continue;
            }

            double distance = abs(point_to_chord.x * chord.y -
                                  point_to_chord.y * chord.x);
            distance /= sqrt(chord.x * chord.x + chord.y * chord.y);
            // Максимальное расстояние и будет кривизной контура в точке.
            if (distance > curvature[i])
                curvature[i] = (float)distance;
        }
    }

    return curvature;
}
