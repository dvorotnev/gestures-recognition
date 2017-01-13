/*
    Реализация таймера для измерения производительности.
*/

#include "Timer.h"

using namespace cv;

Timer::Timer() : last_tick_value(0), tick_counter(0)
{
}

void Timer::start()
{
    last_tick_value = getTickCount();
}

void Timer::stop()
{
    size_t time = getTickCount() - last_tick_value;
    tick_counter += time;
}

double Timer::getTime()
{
    return (double)tick_counter / getTickFrequency();
}
