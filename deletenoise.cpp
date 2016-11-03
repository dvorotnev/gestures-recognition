﻿/*
    Реализация функции для удаления мелких объектов с бинарного изображения.
*/

#include <vector>

#include "deletenoise.h"

using namespace cv;
using namespace std;

const uchar BackGround = 0;
const uchar ForeGround = 255;

// Маркирует все объекты на бинарном изображении и удаляет
// объекты, которые по площади меньше, чем max_area.
static void markAndClearImage(Mat& srcImage, Mat& dstImage, int max_area);

// Слияние меток двух объектов.
static int mergeObjects(int top, int left, vector<int>& parents);

// Удаляет объекты, площадь которых меньше, чем max_area
// и находит для каждого объекта самого первого родителя в таблице.
static void setLabels(vector<int>& table, vector<int>& square, int max_area);

// Переобозначает объекты на входном изображении и отмечает оставшиеся объекты
// на выходном бинарном изображении.
static void reassignObjects(vector<int>& table, Mat& marked_image, Mat& binary_image);

// Инвертирует бинарное изображение.
static void inverseImage(Mat& binary_image, Mat& marked_image);

void deleteNoise(Mat &image, Mat &marked_image, int max_area)
{
    marked_image.create(image.rows, image.cols, CV_32S);

    // Удаляем шум "перец".
    inverseImage(image, marked_image);
    markAndClearImage(marked_image, image, max_area);

    // Удаляем шум "соль"
    inverseImage(image, marked_image);
    markAndClearImage(marked_image, image, max_area);
}

static void markAndClearImage(Mat& marked_image, Mat& dstImage, int max_area)
{
    // Вектор для хранения площадей объектов.
    vector<int> square;
    // Вектор для хранения родителей объектов.
    vector<int> parents;

    // Маркируем изображение и создаём таблицу со смежными классами.
    int counter = 0;
    for (int y = 0; y < marked_image.rows; ++y)
    {
        int* ptr = marked_image.ptr<int>(y);
        for (int x = 0; x < marked_image.cols; ++x)
        {
            // Получаем значение просматриваемого пикселя.
            int current = ptr[x];
            if (current == BackGround)
                continue;

            // Получаем значение левого пикселя.
            int left = 0;
            if (x < 1) left = BackGround;
            else left = ptr[x - 1];

            // Получаем значение верхнего пикселя.
            int top = 0;
            if (y < 1) top = BackGround;
            else top = marked_image.ptr<int>(y - 1)[x];

            if ((left == BackGround) && (top == BackGround))
            {
                // Найден новый объект.
                square.push_back(1);
                parents.push_back(-1);
                ++counter;
                ptr[x] = counter;
                continue;
            }
            if ((left != BackGround) && (top == BackGround))
            {
                ++square[left - 1];
                ptr[x] = left;
                continue;
            }
            if ((left == BackGround) && (top != BackGround))
            {
                ++square[top - 1];
                ptr[x] = top;
                continue;
            }
            if ((left != BackGround) && (top != BackGround))
            {
                if (left != top)
                    top = mergeObjects(left, top, parents);

                ++square[top - 1];
                ptr[x] = top;
                continue;
            }
        }
    }

    setLabels(parents, square, max_area);
    reassignObjects(parents, marked_image, dstImage);
}

static int mergeObjects(int top, int left, vector<int>& parents)
{
    // Делаем верхнюю метку наименьшей.
    if (left < top)
        std::swap(top, left);

    if (parents[left - 1] == top - 1)
        return top;

    if (parents[left - 1] == -1)
    {
        // Родителя не было.
        parents[left - 1] = top - 1;
    }
    else
    {
        // Меняем родительскую метку самому
        // верхнему объекту в иерархии родителей.
        int parent = top - 1;
        while (parents[parent] != -1)
            parent = parents[parent];

        int child = left - 1;
        while (parents[child] != -1)
            child = parents[child];

        if (child != parent)
        {
            // Меньшая метка - parent,
            // старшая метка - child.
            if (child < parent)
                std::swap(parent, child);

            parents[child] = parent;
        }
    }

    return top;
}

static void setLabels(vector<int>& table, vector<int>& square, int max_area)
{
    for (int i = 0; i < table.size(); ++i)
    {
        if (table[i] == -1) continue;

        // Находим самого первого родителя объекта.
        int parent = table[i];
        while (table[parent] != -1)
            parent = table[parent];

        table[i] = parent;
        square[parent] += square[i];
        square[i] = 0;
    }

    // Устанавливаем объекту нужную метку.
    for (int i = 0; i < table.size(); ++i)
    {
        if (table[i] == -1)
            table[i] = i + 1;
        else
            ++table[i];
    }

    // Если площадь объекта меньше порога, то удаляем его.
    for (int i = 0; i < table.size(); ++i)
    {
        if ((square[i] < max_area) && (square[i] != 0))
        {
            for (int j = 0; j < table.size(); ++j)
            {
                if (table[j] == i + 1)
                    table[j] = BackGround;
            }
        }
    }
}

static void reassignObjects(vector<int>& table, Mat& marked_image, Mat& binary_image)
{
    // Объединяем смежные объекты на маркированом изображении.
    for (int y = 0; y < marked_image.rows; ++y)
    {
        int* ptr = marked_image.ptr<int>(y);
        for (int x = 0; x < marked_image.cols; ++x)
        {
            if (ptr[x] == BackGround)
                continue;

            ptr[x] = table[ptr[x] - 1];
        }
    }

    // Переносим информацию на бинарное изображение.
    for (int y = 0; y < binary_image.rows; ++y)
    {
        const int* src = marked_image.ptr<int>(y);
        uchar* dst = binary_image.ptr(y);
        for (int x = 0; x < binary_image.cols; ++x)
        {
            if (src[x] == BackGround)
                dst[x] = BackGround;
            else
                dst[x] = ForeGround;
        }
    }
}

static void inverseImage(Mat& binary_image, Mat& marked_image)
{
    for (int y = 0; y < binary_image.rows; ++y)
    {
        const uchar* src = binary_image.ptr(y);
        int* dst = marked_image.ptr<int>(y);
        for (int x = 0; x < binary_image.cols; ++x)
        {
            if (src[x] == BackGround)
                dst[x] = -1;
            else
                dst[x] = BackGround;
        }
    }
}
