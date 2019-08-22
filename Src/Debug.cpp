/*
    Реализация дополнительного функционала для отладки.
*/

#include <stdio.h>
//#include <direct.h>
#include <opencv2/highgui.hpp>

#include <Debug.h>

using namespace cv;

static unsigned int debug_frame_counter = 0;

#if ___DEBUG___
static uchar BackGround = 0;
static uchar ForeGround = 255;
#endif // ___DEBUG___

void imageWrite(const String& name, const Mat& mat)
{
#if ___DEBUG___
    _mkdir(debug_directory);
    String file_name = String(debug_directory) + name;
    _mkdir(file_name.c_str());
    char frame_counter_str[10];
    sprintf_s(frame_counter_str, "%d", debug_frame_counter);
    file_name = file_name + "\\" + frame_counter_str + ".png";
    imwrite(file_name, mat);
#endif // ___DEBUG___
}

#if ___DEBUG___
static void binarizeImage(Mat& image)
{
    for (int y = 0; y < image.rows; ++y)
    {
        uchar* ptr = image.ptr(y);
        for (int x = 0; x < image.cols; ++x)
        {
            if (ptr[x] != BackGround)
                ptr[x] = ForeGround;
        }
    }
}
#endif // ___DEBUG___

void imageShow(const String& winname, const Mat& mat)
{
    if (winname == "Input")
        ++debug_frame_counter;

#if ___DEBUG___
    if (mat.channels() == 1)
    {
        Mat output = mat.clone();
        binarizeImage(output);
        imageWrite(winname, output);
    }
    else
    {
        imageWrite(winname, mat);
    }

#endif // ___DEBUG___
    imshow(winname, mat);
}
