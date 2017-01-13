/*
    Реализация модуля последовательного воспроизведения кадров видеопоследовательности.
*/

#include "VideoSequenceCapture.h"

using namespace cv;

VideoSequenceCapture::VideoSequenceCapture(const String& filename)
{
    size_t directory_index = filename.find_last_of('\\');
    size_t extension_index = filename.find_last_of('.');

    directory = filename.substr(0, directory_index + 1);
    extension = filename.substr(extension_index, filename.length());

    name = filename.substr(directory_index + 1, extension_index - directory_index - 1);
    size_t counter_index = name.find_first_of("0123456789");
    String counter_str = name.substr(counter_index, name.length());
    counter = atoi(counter_str.c_str());
    name = name.substr(0, counter_index);
}

void VideoSequenceCapture::operator>>(Mat& image)
{
    char counter_str[10];
    sprintf(counter_str, "%d", counter);
    String filename = directory + name + counter_str + extension;
    image = imread(filename);
    ++counter;
    return;
}
