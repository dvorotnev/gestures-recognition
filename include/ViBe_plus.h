/*
    Алгоритм сегментации движения ViBe+.
*/

#ifndef __VIBE_PLUS_H__
#define __VIBE_PLUS_H__

#include <opencv2\core.hpp>
#include <opencv2\video.hpp>

#include <ViBe.h>

class ViBe_plus : public ViBe
{
public:
    ViBe_plus();
    ViBe_plus(int history_depth, int radius, int min_overlap, int probability);

    // Функция вычисляет маску сегментации и обновляет модель.
    // TODO: вычислять prob через learningrate.
    void apply(const cv::InputArray &image, cv::OutputArray &mask, double);

private:
    void update(const cv::Mat& image, const cv::Mat& update_mask);
};

#endif // __VIBE_PLUS_H__
