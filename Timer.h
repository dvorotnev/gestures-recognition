﻿/*
    Таймер для измерения производительности.
*/

#include <core.hpp>

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
