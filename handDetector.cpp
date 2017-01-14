/*
    Реализация функции распознавания руки на основе анализа кривизны контура.
*/

#include "handDetector.h"

using namespace std;

int handDetector(const vector<float>& curvature, float min_treshold, float max_trethold, int min_counter, int max_counter)
{
    const size_t length = curvature.size();
    if (length < 2)
        return -1;

    vector<float> derivative(length, 0.0);

    // Вычисляем первую производную в каждой точке функции кривизны.
    derivative[0] = curvature[1] - curvature[0];
    derivative[length - 1] = curvature[length - 1] - curvature[length - 2];
    for (int i = 1; i < length - 1; ++i)
    {
        derivative[i] = (curvature[i + 1] - curvature[i - 1]) / 2;
    }

    // Подсчитываем количество локальных максимумов.
    int counter = 0; // Счётчик локальных максимумов.
    float extr = 0; // Предыдущее значение в точке экстремума.
    bool search_max = true; // Флаг, указывающий ищется максимум или минимум.
    for (int i = 0; i < length - 1; ++i)
    {
        const float multiply = derivative[i] * derivative[i + 1];
        if (multiply > 0)
            continue;

        const float current = curvature[i];
        if (search_max)
        {
            // Если найден локальный максимум, больший чем заданный порог.
            if ((current < extr) && (extr >= max_trethold))
            {
                search_max = false;
                ++counter;
            }
            extr = current;
        }
        else
        {
            // Если найден локальный минимум, меньший чем заданный порог.
            if ((current > extr) && (extr < min_treshold))
                search_max = true;
            extr = current;
        }

    }

    if ((counter >= min_counter) && (counter <= max_counter))
        return 1;

    return 0;
}
