/*
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
// Удаляет объекты, площадь которых меньше, чем max_area
// и находит для каждого объекта самого первого родителя в таблице.
static void optimizeParentTable(vector<int>& table, vector<int>& square, int max_area);
// Переобозначает объекты на входном изображении и отмечает оставшиеся объекты
// на выходном бинарном изображении.
static void reassignObjects(vector<int>& table, Mat& srcImage, Mat& dstImage);
// Инвертирует бинарное изображение.
static void inverseImage(Mat& srcImage, Mat& dstImage);

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
    vector<int> parent;

    // Маркируем изображение и создаём таблицу со смежными классами.
    int counter = 0;
    for (int y = 0; y < marked_image.rows; ++y)
    {
        int* ptr = marked_image.ptr<int>(y);
        for (int x = 0; x < marked_image.cols; ++x)
        {
            // Получаем значение просматриваемого пикселя.
            int current = ptr[x];
            if (current == BackGround) continue;

            // Получаем значение левого пикселя.
            int left;
            if (x  < 1) left = BackGround;
            else left = ptr[x - 1];

            // Получаем значение верхнего пикселя.
            int top;
            if (y < 1) top = BackGround;
            else top = marked_image.ptr<int>(y - 1)[x];

            // Встретился новый объект.
            if ((left == BackGround) && (top == BackGround))
            {
                square.push_back(1);
                parent.push_back(-1);
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
                // Соединение двух разных объектов.
                if (left != top)
                {
                    // Делаем верхнюю метку наименьшей.
                    if (left < top) std::swap(top, left);
                    // Устраняем конфликты при слиянии.
                    if (parent[left - 1] != top - 1)
                    {
                        // Родителя не было.
                        if (parent[left - 1] == -1)
                        {
                            parent[left - 1] = top - 1;
                        }
                        // Меняем родительскую метку самому
                        // верхнему объекту в иерархии родителей.
                        else
                        {
                            int par = top - 1;
                            while (parent[par] != -1)
                                par = parent[par];

                            int child = left - 1;
                            while (parent[child] != -1)
                                child = parent[child];

                            if (child != par)
                            {
                                // Меньшая метка - родитель,
                                // старшая метка - ребёнок.
                                if (child < par) std::swap(par, child);
                                parent[child] = par;
                            }
                        }
                    }
                }
                ++square[top - 1];
                ptr[x] = top;
                continue;
            }
        }
    }

    optimizeParentTable(parent, square, max_area);
    reassignObjects(parent, marked_image, dstImage);
}

void optimizeParentTable(vector<int>& table, vector<int>& square, int max_area)
{
    for (int i = 0; i < table.size(); ++i)
    {
        if (table[i] == -1) continue;

        // Находим самого первого родителя объекта.
        int parent = table[i];
        while (table[parent] != -1)
            parent = table[parent];

        table[i] = parent;

        // Прибавляем площадь объекта к площади родителя.
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

void reassignObjects(vector<int>& table, Mat& srcImage, Mat& dstImage)
{
    // Объединяем смежные объекты на маркированом изображении.
    for (int y = 0; y < srcImage.rows; ++y)
    {
        int* ptr = srcImage.ptr<int>(y);
        for (int x = 0; x < srcImage.cols; ++x)
        {
            if (ptr[x] == BackGround) continue;
            // TODO: Возможно лишняя проверка.
            if (table[ptr[x] - 1] == 0)
            {
                ptr[x] = BackGround;
                continue;
            }
            ptr[x] = table[ptr[x] - 1];
        }
    }

    // Переносим информацию на бинарное изображение.
    for (int y = 0; y < dstImage.rows; ++y)
    {
        uchar* dst = dstImage.ptr(y);
        int* src = srcImage.ptr<int>(y);
        for (int x = 0; x < dstImage.cols; ++x)
        {
            if (src[x] == BackGround)
                dst[x] = BackGround;
            else
                dst[x] = ForeGround;
        }
    }
}

static void inverseImage(Mat& srcImage, Mat& dstImage)
{
    for (int y = 0; y < srcImage.rows; ++y)
    {
        const uchar* src = srcImage.ptr(y);
        int* dst = dstImage.ptr<int>(y);
        for (int x = 0; x < srcImage.cols; ++x)
        {
            // Немаркированный объект.
            if (src[x] == BackGround)
                dst[x] = -1;
            // Фон.
            else 
                dst[x] = BackGround;
        }
    }
}
