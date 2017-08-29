/*
    Реализация алгоритма сегментации движения ViBe_plus.
*/

#include <ViBe_plus.h>

#include <deletenoise.h>

using namespace cv;

const uchar BackGround = 0;
const uchar ForeGround = 255;

ViBe_plus::ViBe_plus() : ViBe()
{
}

ViBe_plus::ViBe_plus(int history_depth, int radius, int min_overlap, int probability)
:ViBe(history_depth, radius, min_overlap, probability)
{
}

void ViBe_plus::apply(const InputArray &image, OutputArray &fgmask, double)
{
    const Mat image_ = image.getMat();
    fgmask.create(image_.rows, image_.cols, CV_8U);
    Mat fgmask_ = fgmask.getMat();

    getSegmentationMask(image_, fgmask_);

    Mat update_mask = fgmask_.clone();
    deleteNoise(fgmask_, 100, 200);
    deleteNoise(update_mask, 0, 51);

    update(image_, update_mask);
    return;
}

// Вычисление квадрата градиента в заданной точке изображения.
static double computeGradientSqr(const Mat& image, int y, int x)
{
    const uchar* top = image.ptr(y + 1);
    const uchar* current = image.ptr(y);
    const uchar* bottom = image.ptr(y - 1);

    // Переводим окрестность точки в градации серого.
    double gray[3][3] = { 0 };
    gray[0][0] = 0.114 * top[3 * (x - 1)] + 0.587 * top[3 * (x - 1) + 1] + 0.299 * top[3 * (x - 1) + 2];
    gray[0][1] = 0.114 * top[3 * x]       + 0.587 * top[3 * x + 1]       + 0.299 * top[3 * x + 2];
    gray[0][2] = 0.114 * top[3 * (x + 1)] + 0.587 * top[3 * (x + 1) + 1] + 0.299 * top[3 * (x + 1) + 2];

    gray[1][0] = 0.114 * current[3 * (x - 1)] + 0.587 * current[3 * (x - 1) + 1] + 0.299 * current[3 * (x - 1) + 2];
    gray[1][2] = 0.114 * current[3 * (x + 1)] + 0.587 * current[3 * (x + 1) + 1] + 0.299 * current[3 * (x + 1) + 2];

    gray[2][0] = 0.114 * bottom[3 * (x - 1)] + 0.587 * bottom[3 * (x - 1) + 1] + 0.299 * bottom[3 * (x - 1) + 2];
    gray[2][1] = 0.114 * bottom[3 * x]       + 0.587 * bottom[3 * x + 1]       + 0.299 * bottom[3 * x + 2];
    gray[2][2] = 0.114 * bottom[3 * (x + 1)] + 0.587 * bottom[3 * (x + 1) + 1] + 0.299 * bottom[3 * (x + 1) + 2];

    double grad_x = gray[0][2] - gray[0][0];
    grad_x += 2 * (gray[1][2] - gray[1][0]);
    grad_x += gray[2][2] - gray[2][0];

    double grad_y = gray[2][0] - gray[0][0];
    grad_y += 2 * (gray[2][1] - gray[0][1]);
    grad_y += gray[2][2] - gray[0][2];

    double sqr_grad = grad_x * grad_x + grad_y * grad_y;

    return sqr_grad;
}

void ViBe_plus::update(const Mat& image, const Mat& update_mask)
{
    if (needToInit())
    {
        initialize(image);
        return;
    }

    for (int y = 0; y < image.rows; ++y)
    {
        const uchar* mask = update_mask.ptr(y);
        for (int x = 0; x < image.cols; ++x)
        {
            if (mask[x] != BackGround)
                continue;

            updatePixel(image, y, x);

            // Ограничиваем пространственное распространение.
            if ((y > 0 && y < image.rows - 1 && x > 0 && x < image.cols - 1) &&
                (mask[x - 1] != BackGround || update_mask.ptr(y - 1)[x] != BackGround ||
                mask[x + 1] != BackGround || update_mask.ptr(y + 1)[x] != BackGround))
            {
                double sqr_grad = computeGradientSqr(image, y, x);
                if (sqr_grad > 2500)
                    continue;
            }

            updateNeiborPixel(image, y, x);
        }
    }

    return;
}
