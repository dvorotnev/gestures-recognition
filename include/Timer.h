/*
    Таймер для измерения производительности.
*/

#ifndef __TIMER_H__
#define __TIMER_H__

#include <opencv2/core.hpp>

class Timer
{
public:
    Timer();

    // Запуск таймера.
    void start();
    // Остановка таймера.
    void stop();
    // Возвращает суммарное время работы таймера в секундах.
    double getTime();
private:
    size_t last_tick_value; // Количество тиков во время последнего запуска таймера.
    size_t tick_counter; // Суммарное количество тиков за время работы таймера.
};

#endif // __TIMER_H__
