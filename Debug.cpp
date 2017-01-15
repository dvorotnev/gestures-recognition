/*
    Реализация дополнительного функционала для отладки.
*/

#include <fstream>
#include <highgui.hpp>
#include <direct.h>

#include "graphUtils\GraphUtils.h"

#include "Debug.h"

static unsigned int debug_frame_counter = 0;

void imageWrite(const cv::String& name, const cv::Mat& mat)
{
#if ___DEBUG___
    cv::String file_name = cv::String(debug_directory) + name;
    _mkdir(file_name.c_str());
    char frame_counter_str[10];
    sprintf_s(frame_counter_str, "%d", debug_frame_counter);
    file_name = file_name + "\\" + frame_counter_str + ".png";
    imwrite(file_name, mat);
#endif // ___DEBUG___
}

void binaryImageShow(const cv::String& winname, const cv::Mat& mat)
{
    if (winname == "Input")
        ++debug_frame_counter;

#if ___DEBUG___
    imageWrite(winname, mat);
#else
    imshow(winname, mat);
#endif // ___DEBUG___
}

void imageShow(const cv::String& winname, const cv::Mat& mat)
{
#if ___DEBUG___
    cv::String file_name = cv::String(debug_directory) + winname;
    _mkdir(file_name.c_str());
    char frame_counter_str[10];
    sprintf_s(frame_counter_str, "%d", debug_frame_counter);
    file_name = file_name + "\\" + frame_counter_str + ".txt";
    std::ofstream file(file_name);

    for (int y = 0; y < mat.rows; ++y)
    {
        const uchar* ptr = mat.ptr(y);
        for (int x = 0; x < mat.cols; ++x)
        {
            file << (int)ptr[x] << " ";
        }
        file << std::endl;
    }
    file.close();
#else
    imshow(winname, mat);
#endif // ___DEBUG___
}

void curvatureShow(const cv::String& winname, const std::vector<float>& curvature)
{
#if ___DEBUG___
    cv::String file_name = cv::String(debug_directory) + winname;
    _mkdir(file_name.c_str());
    char frame_counter_str[10];
    sprintf_s(frame_counter_str, "%d", debug_frame_counter);
    file_name = file_name + "\\" + frame_counter_str + ".txt";
    std::ofstream file(file_name);

    for (int i = 0; i < curvature.size(); ++i)
    {
        file << curvature[i] << " ";
    }
    file << std::endl;

    file.close();
#else
    showFloatGraph("Curvature", &curvature[0], (int)curvature.size(), 1);
#endif // ___DEBUG___
}
