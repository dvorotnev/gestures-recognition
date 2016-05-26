/*
    –еализаци€ алгоритма сегментации движени€ ViBe.
*/

#include "ViBe.h"

using namespace cv;

const uchar BackGround = 0;
const uchar ForeGround = 255;

ViBe::ViBe() 
    : history_depth(20), sqr_rad(20 * 20), min_overlap(2), prob(16), samples(), generator()
{
}

ViBe::ViBe(int _history_depth, int rad, int _min_overlap, int _prob)
    : history_depth(_history_depth), sqr_rad(rad*rad), min_overlap(_min_overlap), 
    prob(_prob), samples(), bg_mat(), generator()
{
}

ViBe::~ViBe()
{
    for (int y = 0; y < samples.rows; ++y)
    {
        for (int x = 0; x < samples.cols; ++x)
        {
            delete[] samples(y, x);
        }
    }
    samples.release();
    bg_mat.release();
}

double ViBe::distancesqr(const Point3_<uchar> &pixel, const Point3_<uchar> &sample) const
{
    double sum = 0;
    sum += (pixel.x - sample.x)*(pixel.x - sample.x);
    sum += (pixel.y - sample.y)*(pixel.y - sample.y);
    sum += (pixel.z - sample.z)*(pixel.z - sample.z);
    return sum;
}

void ViBe::initialization(const Mat &image)
{
    samples.release();
    bg_mat.release();
    samples.create(image.rows, image.cols);
    bg_mat.create(image.rows, image.cols, CV_8UC3);

    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            samples(y, x) = new Point3_<uchar>[history_depth];
            //«аполн€ем первое значение модели значением текущего пиксел€.
            samples(y, x)[0] = {image.ptr(y)[3 * x],
                                image.ptr(y)[3 * x + 1],
                                image.ptr(y)[3 * x + 2]};

            //ќстальные значени€ модели заполн€ем значени€ми соседних пикселей.
            for (int k = 1; k < history_depth; ++k)
            {
                Point2i neib_pixel = GetRandNeibPixel(Point2i(x, y)); // TODO: сделать передачу по ссылке.
                samples(y, x)[k] = {image.ptr(neib_pixel.y)[3 * neib_pixel.x],
                                    image.ptr(neib_pixel.y)[3 * neib_pixel.x + 1], 
                                    image.ptr(neib_pixel.y)[3 * neib_pixel.x + 2]};
            }

            // «начение фона равно значению текущего пиксел€.
            bg_mat.ptr(y)[3 * x] = image.ptr(y)[3 * x];
            bg_mat.ptr(y)[3 * x + 1] = image.ptr(y)[3 * x + 1];
            bg_mat.ptr(y)[3 * x + 2] = image.ptr(y)[3 * x + 2];
        }
    }
}

Point2i ViBe::GetRandNeibPixel(const Point2i &pixel)
{
    Point2i neib_pixel;
    do
    {
        //Ќаходим координату х соседнего пиксел€.
        if (pixel.x == 0)
            neib_pixel.x = generator.uniform(0, 2);
        else if (pixel.x == samples.cols - 1)
            neib_pixel.x = generator.uniform(samples.cols - 2, samples.cols);
        else
            neib_pixel.x = generator.uniform(pixel.x - 1, pixel.x + 2);

        //Ќаходим координату у соседнего пиксел€.
        if (pixel.y == 0)
            neib_pixel.y = generator.uniform(0, 2);
        else if (pixel.y == samples.rows - 1)
            neib_pixel.y = generator.uniform(samples.rows - 2, samples.rows);
        else
            neib_pixel.y = generator.uniform(pixel.y - 1, pixel.y + 2);
    } while ((neib_pixel.x == pixel.x) && (neib_pixel.y == pixel.y));

    return neib_pixel;
}

void ViBe::apply(const InputArray &_image, OutputArray &_fgmask, double)
{
    const Mat image = _image.getMat();
    _fgmask.create(image.rows, image.cols, CV_8U);
    Mat fgmask = _fgmask.getMat();

    // ѕереинициализаци€ если модель пуста€ или передан кадр с другими размерами.
    if ((samples.empty() == 1) || (samples.rows != image.rows) || (samples.cols != image.cols))
    {
        initialization(image);
        return;
    }

    for (int y = 0; y < image.rows; ++y)
    {
        const uchar* src = image.ptr(y);
        uchar* dst = fgmask.ptr(y);
        for (int x = 0; x < image.cols; ++x)
        {
            // Ќаходим количество совпадений текущего значени€ пиксел€ с моделью.
            Point3_<uchar> pixel(src[x * 3], src[x * 3 + 1], src[x * 3 + 2]);
            int counter = 0;
            int index = 0;
            double dist = 0;

            while ((counter < min_overlap) && (index < history_depth))
            {
                Point3_<uchar> model_pixel = samples(y, x)[index];
                dist = distancesqr(pixel, model_pixel);
                if (dist < sqr_rad)
                {
                    counter++;
                }
                index++;
            }

            // ≈сли пиксель принадлежит фону, то обновл€ем модель.
            if (counter >= min_overlap)
            {
                dst[x] = BackGround;
                // ќбновление модели текущего пиксел€.
                int randnumber = generator.uniform(0, prob);
                if (randnumber == 0)
                {
                    randnumber = generator.uniform(0, history_depth);
                    samples(y, x)[randnumber] = pixel;
                    bg_mat.ptr(y)[3 * x] = pixel.x;
                    bg_mat.ptr(y)[3 * x + 1] = pixel.y;
                    bg_mat.ptr(y)[3 * x + 2] = pixel.z;
                }

                // ќбновление модели случайного соседа из восьмисв€зной области.
                randnumber = generator.uniform(0, prob);
                if (randnumber == 0)
                {
                    Point2i neib_pixel = GetRandNeibPixel(Point2i(x, y));
                    randnumber = generator.uniform(0, history_depth);
                    samples(neib_pixel.y, neib_pixel.x)[randnumber] = pixel;
                }
            }
            // ѕомечаем движущуюс€ точку.
            else
            {
                dst[x] = ForeGround;
            }
        }
    }
}

void ViBe::getBackgroundImage(cv::OutputArray &Image) const
{
    Mat image = Image.getMat();
    bg_mat.copyTo(image);
}
