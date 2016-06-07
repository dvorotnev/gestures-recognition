#include "Contour.h"

#include <assert.h>

#if ___DEBUG___
#include <string>

extern unsigned int debug_counter;
extern char path[];
#endif

using namespace std;
using namespace cv;

const uchar Background = 0;
const uchar ForeGround = 255;

// Функция кодирует направление между двумя точками.
static int CodeDirection(const Point2i& first, const Point2i& second)
{
    const int dx = second.x - first.x;
    const int dy = second.y - first.y;
    if ((abs(dx) > 1) || (abs(dy) > 1))
        return -1;

    if (dx > 0)
    {
        if (dy > 0)
            return 3;
        else if (dy < 0)
            return 1;
        else
            return 2;
    }
    else if (dx < 0)
    {
        if (dy > 0)
            return 5;
        else if (dy < 0)
            return 7;
        else
            return 6;
    }
    else
    {
        if (dy > 0)
            return 4;
        else if (dy < 0)
            return 0;
        else
            return -1;
    }
}

// Функция декодирует направление между двумя точками. 
// В second записываются координаты второй точки.
static int DecodeDirection(const Point2i& first, Point2i& second, const int code)
{
    if ((code > 7) || (code < 0))
        return -1;

    // Определяем координату x.
    switch (code)
    {
        case 0:
        case 4:
            second.x = first.x;
            break;
        case 1:
        case 2:
        case 3:
            second.x = first.x + 1;
            break;
        case 5:
        case 6:
        case 7:
            second.x = first.x - 1;
            break;
        default:
            return -1;
    }

    // Определяем координату y.
    switch (code)
    {
    case 0:
    case 1:
    case 7:
        second.y = first.y - 1;
        break;
    case 2:
    case 6:
        second.y = first.y;
        break;
    case 3:
    case 4:
    case 5:
        second.y = first.y + 1;
        break;
    default:
        break;
    }

    return 0;
}

size_t ContourMap::getNumberOfContours() const
{
    return contours.size();
}

void ContourMap::printСontour(Mat& image, int number) const
{
    // TODO: exception.
    if (image.empty())
        return;

    size_t size = getNumberOfContours();
    if ((number + 1 > size) || (number < 0))
        return;
    Contour current(contours[number]);
    Point2i point = current.start;

    uchar* ptr = image.ptr(point.y);
#if ___DEBUG___
    ptr[point.x] = (uchar)number;
#else
    ptr[point.x] = ForeGround;
#endif

    vector<int>& code = current.chain_code;
    Point2i next_point(-1,-1);
    for (vector<int>::iterator iter = code.begin(); iter != code.end(); ++iter)
    {
        // TODO: exception.
        if (DecodeDirection(point, next_point, *iter) != 0)
            return;
        assert((next_point.x >= 0) && (next_point.x < image.cols) &&
            (next_point.y >= 0) && (next_point.y < image.rows));
        // Отмечаем точку контура на изображении.
        ptr = image.ptr(next_point.y);
#if ___DEBUG___
        ptr[next_point.x] = (uchar)(number+1);
#else
        ptr[next_point.x] = ForeGround;
#endif
        point = next_point;
    }
}

void ContourMap::printAllContours(Mat& image) const
{
    // Очищаем изображение с контурами.
    for (int y = 0; y < image.rows; ++y)
    {
        uchar* ptr = image.ptr(y);
        for (int x = 0; x < image.cols; ++x)
        {
            if (ptr[x] != Background)
                ptr[x] = Background;
        }
    }

    for (int i = 0; i < getNumberOfContours(); ++i)
    {
        printСontour(image, i);
    }
}

void ContourMap::sortContours()
{
    int size = (int)contours.size();
    for (int i = size - 1; i > 1; --i)
    {
        for (int j = 0; j < i; ++j)
        {
            if (contours[j].chain_code.size() < contours[j + 1].chain_code.size())
            {
                swap(contours[j], contours[j + 1]);
            }
        }
    }
}

int ContourMap::getCurvature(std::vector<float>& curvature, const int chord_length, int number) const
{
    // Считываем нужный контур.
    const Contour& contour = contours[number];
    const vector<int>& chain_code = contour.chain_code;
    const size_t length = chain_code.size() + 1;
    if (length < 4)
        return -1;

    // Декодируем все точки контура.
    vector<Point2i> points(length);
    points[0] = contour.start;
    for (int i = 0; i < length - 1; ++i)
    {
        if (DecodeDirection(points[i], points[i + 1], chain_code[i]) != 0)
            return -1;
    }

    curvature.resize(length, 0);
    for (int current = 0; current < length; ++current)
    {
        // Вычисляем координаты концов хорды.
        const int first_point = current - chord_length + 2;
        const int last_point = current + 1;

        // Смещаем хорду по контуру.
        for (int shift = 0; shift < chord_length - 3; ++shift)
        {
            int start_index = first_point + shift;
            if (start_index < 0)
                start_index = 0;
            if (start_index >= length)
                start_index = (int)length - 1;

            int end_index = last_point + shift;
            if (end_index >= length)
                end_index = (int)length - 1;

            // Вычисляем расстояние от точки до хорды.
            const Point2i chord = points[end_index] - points[start_index];
            const Point2i point_to_chord = points[current] - points[start_index];
            double distance = abs(point_to_chord.x * chord.y - point_to_chord.y * chord.x);
            distance /= sqrt(chord.x * chord.x + chord.y * chord.y);
            // Максимальное расстояние и будет кривизной контура в точке.
            if (distance > curvature[current])
                curvature[current] = (float)distance;
        }
    }

    return 0;
}

// Функция для нахождения следующей точки контура на изображении.
static int FindNextPoint(const Mat& image, Point2i& point, const int last_direction)
{
    // TODO: exception.
    if ((last_direction > 7) || (last_direction < 0))
        return -1;

    // Массив всех возможных направлений в порядке 
    // наиболее вероятного поялвения.
    int directions_queue[7] = {last_direction,
        last_direction + 1, last_direction - 1,
        last_direction + 2, last_direction - 2,
        last_direction + 3, last_direction - 3};

    // Приводим значения направлений в диапазон от 0 до 7.
    for (int i = 0; i < 7; ++i)
    {
        if (directions_queue[i] > 7)
            directions_queue[i] -= 8;
        else if (directions_queue[i] < 0)
            directions_queue[i] += 8;
    }

    // Ищем следующую точку среди возможных направлений.
    Point2i next_point(-1, -1);
    for (int i = 0; i < 7; ++i)
    {
        if (DecodeDirection(point, next_point, directions_queue[i]) == -1)
            return -1;
        if ((next_point.x < 0) || (next_point.x >= image.cols) ||
            (next_point.y < 0) || (next_point.y >= image.rows))
            continue;
        const uchar* ptr = image.ptr(next_point.y);
        if (ptr[next_point.x] == ForeGround)
        {
            point.x = next_point.x;
            point.y = next_point.y;
            return directions_queue[i];
        }
    }
    return -1;
}

// Функция определяет нужно ли удалять точку контура на изображении.
// TODO: переделать.
/*static bool DeletePoint(const Mat& image, const Point2i& point)
{
    int sum = 0;
    for (int y = point.y - 1; y <= point.y + 1; ++y)
    {
        if ((y < 0) || (y >= image.rows))
            continue;
        const uchar* ptr = image.ptr(y);
        for (int x = point.x - 1; x <= point.x + 1; ++x)
        {
            if ((x < 0) || (x > image.cols))
                continue;
            if (ptr[x] == Background)
                continue;
            ++sum;
            if (sum >= 3)
                return false;
        }
    }
    return true;
}*/

// Функция считывает точки контура у которого задано начало и удаляет его с изображения.
static void ReturnContour(Mat& image, Contour& contour)
{
    Point2i start(contour.start);
    uchar* ptr = image.ptr(start.y);
    ptr[start.x] = Background;

    Point2i point (-1, -1);
    for (int i = 0; i < 2; ++i)
    {
        // Определяем направление поиска.
        int direction = -1;
        for (int y = start.y - 1; y <= start.y + 1; ++y)
        {
            if ((y < 0) || (y >= image.rows))
                continue;
            uchar* ptr = image.ptr(y);
            for (int x = start.x - 1; x <= start.x + 1; ++x)
            {
                if ((x < 0) || (x >= image.cols))
                    continue;
                if ((x == start.x) && (y == start.y))
                    continue;
                if (ptr[x] == ForeGround)
                {
                    point.x = x;
                    point.y = y;
                    direction = CodeDirection(start, point);
                    break;
                }
            }
            if (direction != -1)
                break;
        }

        // Ищем точки контура.
        while (direction != -1)
        {
            // При первом проходе записываем обратные направления.
            if (i == 0)
            {
                const int inverse_direction = (direction + 4) % 8;
                contour.chain_code.push_back(inverse_direction);
            }
            else
            {
                contour.chain_code.push_back(direction);
            }

            // TODO: использовать изображение с объектами.
            // Удалять точку нужно только в том случае, если она
            // не принадлежит контурам разных объектов.
            //if (DeletePoint(image, point))
            {
                uchar* ptr = image.ptr(point.y);
                ptr[point.x] = Background;
            }
            direction = FindNextPoint(image, point, direction);
        }

        // Инвертируем записанные элементы и меняем начало.
        if ((i == 0) && (contour.chain_code.size() >= 4))
        {
            reverse(contour.chain_code.begin(), contour.chain_code.end());
            contour.start = point;
        }
    }
}

void ContourMapMorph::findContours(InputArray& BinImage)
{
    Mat image;
    image = BinImage.getMat();

    Mat temp;
    temp.create(image.rows, image.cols, CV_8U);

    // С помощью бинарной морфологии получаем границы объектов (обводим сверху).
    Matx <uchar, 4, 4> kernel_close = { 1, 1, 1, 1,
                                        1, 1, 1, 1,
                                        1, 1, 1, 1,
                                        1, 1, 1, 1};
    morphologyEx(image, image, MORPH_CLOSE, kernel_close);
#if ___DEBUG___
    imwrite(String(path) + "res_close\\" + std::to_string(debug_counter) + ".png", image);
#endif

    Matx <uchar, 3, 3> kernel_dilate = {0, 1, 0,
                                        1, 1, 1,
                                        0, 1, 0};
    dilate(image, temp, kernel_dilate);
    temp -= image;
#if ___DEBUG___
    imwrite(String(path) + "res_dilate\\" + std::to_string(debug_counter) + ".png", temp);
#endif

    // Записываем все контуры, найденные на изображении.
    for (int y = 0; y < temp.rows; ++y)
    {
        uchar* ptr = temp.ptr(y);
        for (int x = 0; x < temp.cols; ++x)
        {
            if (ptr[x] != ForeGround) 
                continue;
            Contour current = {Point2i(x, y), vector<int>()};
            ReturnContour(temp, current);
            if (current.chain_code.size() >= 4)
                contours.push_back(current);
        }
    }
}

//bool ContourMap2::IsContourPixel(const Mat &image, int x, int y){
//	
//	const int* ptr;
//	ptr = image.ptr<int>(y);
//	if (ptr[x] != Background) return false;
//
//	if (y == 0){
//		if (image.ptr<int>(y + 1)[x] != Background) return true;		
//		if (x == 0){
//			if (ptr[x + 1] != Background)return true;			
//		}
//		else if (x == image.cols - 1){
//			if ((ptr[x - 1] != Background) && (ptr[x-1] != -1))return true;			
//		}
//		else{
//			if (ptr[x + 1] != Background)return true;
//			if ((ptr[x - 1] != Background) && (ptr[x - 1] != -1))return true;			
//		}
//	}
//	else if (y == image.rows - 1){
//		if ((image.ptr<int>(y - 1)[x] != Background) && (image.ptr<int>(y - 1)[x] != -1)) return true;
//		if (x == 0){
//			if (ptr[x + 1] != Background)return true;			
//		}
//		if (x == image.cols - 1){
//			if ((ptr[x - 1] != Background) && (ptr[x - 1] != -1))return true;			
//		}
//		else{
//			if (ptr[x + 1] != Background)return true;
//			if ((ptr[x - 1] != Background) && (ptr[x - 1] != -1))return true;			
//		}
//	}
//	else{
//		if (image.ptr<int>(y + 1)[x] != Background) return true;
//		if ((image.ptr<int>(y - 1)[x] != Background) && (image.ptr<int>(y - 1)[x] != -1)) return true;
//		if (x == 0){
//			if (ptr[x + 1] != Background)return true;			
//		}
//		if (x == image.cols - 1){
//			if ((ptr[x - 1] != Background) && (ptr[x - 1] != -1))return true;			
//		}
//		else{
//			if (ptr[x + 1] != Background) return true;
//			if ((ptr[x - 1] != Background) && (ptr[x - 1] != -1))return true;			
//		}
//	}
//
//	return false;
//	/*if (y == 0){
//		if (x == 0){
//			ptr = image.ptr(y + 1);
//			sum += ptr[x] + ptr[x + 1];
//			ptr = image.ptr(y);
//			sum += ptr[x + 1];
//			
//		}
//		else if (x == image.cols - 1){
//
//			ptr = image.ptr(y + 1);
//			sum += ptr[x] + ptr[x - 1];
//			ptr = image.ptr(y);
//			sum += ptr[x - 1];
//			
//		}	
//		else{
//			ptr = image.ptr(y + 1);
//			sum += ptr[x - 1] + ptr[x] + ptr[x + 1];
//			ptr = image.ptr(y);
//			sum += ptr[x - 1] + ptr[x + 1];
//		}		
//	}
//	else if (y == image.rows - 1){
//		if (x == 0){
//			ptr = image.ptr(y - 1);
//			sum += ptr[x] + ptr[x + 1];
//			ptr = image.ptr(y);
//			sum += ptr[x + 1];			
//		}
//		else if (x == image.cols - 1){
//			ptr = image.ptr(y - 1);
//			sum += ptr[x] + ptr[x - 1];
//			ptr = image.ptr(y);
//			sum += ptr[x - 1];			
//		}
//		else{
//			ptr = image.ptr(y - 1);
//			sum += ptr[x - 1] + ptr[x] + ptr[x + 1];
//			ptr = image.ptr(y);
//			sum += ptr[x - 1] + ptr[x + 1];			
//		}
//	}
//	else{
//		ptr = image.ptr(y - 1);
//		sum += ptr[x - 1] + ptr[x] + ptr[x + 1];
//		ptr = image.ptr(y + 1);
//		sum += ptr[x - 1] + ptr[x] + ptr[x + 1];
//		ptr = image.ptr(y);
//		sum += ptr[x - 1] + ptr[x + 1];
//	}
//	if ((sum >= ForeGround * 3) && (sum < ForeGround * 9)) return true;
//	else return false;*/
//}
//
//Point2i ContourMap2::FindNextPoint(const Mat& image, Point2i currentPixel, int mark, Point2i prevPixel) const{
//	
//	const int* ptr;	
//	Point2i analyzPixel;
//	int x = currentPixel.x;
//	int y = currentPixel.y;
//	
//	if (y != 0){
//		analyzPixel.x = currentPixel.x;
//		analyzPixel.y = currentPixel.y - 1;
//		if ((analyzPixel != prevPixel) && (findcontour(image, analyzPixel, mark))) return analyzPixel;
//	}
//
//	analyzPixel.y = currentPixel.y;
//	if (x != 0){
//		analyzPixel.x = currentPixel.x - 1;		
//		if ((analyzPixel != prevPixel) && findcontour(image, analyzPixel, mark)) return analyzPixel;
//	}
//	if (x != image.cols - 1){
//		analyzPixel.x = currentPixel.x + 1;		
//		if ((analyzPixel != prevPixel) && findcontour(image, analyzPixel, mark)) return analyzPixel;
//	}
//
//	if (y != image.rows - 1){
//		analyzPixel.x = currentPixel.x;
//		analyzPixel.y = currentPixel.y + 1;
//		if ((analyzPixel != prevPixel) && findcontour(image, analyzPixel, mark)) return analyzPixel;
//	}
//	
//	if (y != 0){
//		analyzPixel.y = currentPixel.y - 1;
//		if (x != 0){
//			analyzPixel.x = currentPixel.x - 1;
//			if ((analyzPixel != prevPixel) && findcontour(image, analyzPixel, mark)) return analyzPixel;
//		}		
//
//		if (x != image.cols - 1){
//			analyzPixel.x = currentPixel.x + 1;
//			if ((analyzPixel != prevPixel) && findcontour(image, analyzPixel, mark)) return analyzPixel;
//		}
//	}
//
//	if (y != image.rows - 1){
//		analyzPixel.y = currentPixel.y + 1;
//		if (x != 0){
//			analyzPixel.x = currentPixel.x - 1;
//			if ((analyzPixel != prevPixel) && findcontour(image, analyzPixel, mark)) return analyzPixel;
//		}
//		
//		if (x != image.cols - 1){
//			analyzPixel.x = currentPixel.x + 1;
//			if ((analyzPixel != prevPixel) && findcontour(image, analyzPixel, mark)) return analyzPixel;
//		}
//	}
//
//	return Point2i(-1, -1);
//}
//
//bool ContourMap2::FindContour(const Mat& image, Point2i analyzPixel, int mark) const
//{
//	if (image.ptr<int>(analyzPixel.y)[analyzPixel.x] != -1) return false;
//
//	const int* ptr;
//	for (int y = analyzPixel.y - 1; y <= analyzPixel.y + 1; y++){
//		if ((y == image.rows) || (y == -1)) continue;
//		ptr = image.ptr<int>(y);
//		for (int x = analyzPixel.x - 1; x <= analyzPixel.x + 1; x++){
//			if ((x == image.cols) || (x == -1)) continue;
//			if (ptr[x] == mark){
//				return true;
//			}
//		}
//	}
//
//	return false;
//}
//
//vector<Point2i> ContourMap2::ReturnContour(Mat& image, const Point2i start, int mark){
//	vector<Point2i> current;
//	
//	current.push_back(start);
//	image.ptr<int>(start.y)[start.x] = Background;
//	Point2i prevPix(-1, -1);
//
//	while (FindNextPoint(image, start, mark, prevPix) != Point2i(-1, -1)){
//		//Проверить что контур всегда берётся от середины.
//		Point2i temppix;
//		for (int i = 1; ((i < current.size()) && (current.size() > 0)); i++){
//			temppix = current[current.size() - i - 1];
//			current.push_back(temppix);
//			current.erase(current.end() - i - 2);
//		}
//
//		Point2i nextpix(start);
//		
//		while ((nextpix = FindNextPoint(image, nextpix, mark, prevPix)) != Point2i(-1, -1)){
//			
//			current.push_back(nextpix);	
//			image.ptr<int>(nextpix.y)[nextpix.x] ++;
//			prevPix = current[current.size() - 2];
//									
//		}
//		if (current.size() > 1) prevPix = current[1];
//	}	
//
//	return current;
//}
//void ContourMap2::Preprocessing(Mat& image){
//
//	for (int y = 0; y < image.rows - 3; y++){
//		int* ptr = image.ptr<int>(y);
//		for (int x = 3; x < image.cols - 3; x++){
//			if (ptr[x] == Background) continue;
//
//			if (ptr[x] != ptr[x + 1]){
//				if (ptr[x] == ptr[x + 2])ptr[x + 1] = ptr[x];
//				if (ptr[x] == ptr[x + 3]){
//					ptr[x + 2] = ptr[x];
//					ptr[x + 1] = ptr[x];
//				}
//			}
//
//			if (ptr[x] != image.ptr<int>(y + 1)[x]){
//				if (ptr[x] == image.ptr<int>(y + 2)[x]) image.ptr<int>(y + 1)[x] = ptr[x];
//				if (ptr[x] == image.ptr<int>(y + 3)[x]){
//					image.ptr<int>(y + 2)[x] = ptr[x];
//					image.ptr<int>(y + 1)[x] = ptr[x];
//				}
//			}
//			if (ptr[x] != image.ptr<int>(y + 1)[x - 1]){
//				if (ptr[x] == image.ptr<int>(y + 2)[x - 2]) image.ptr<int>(y + 1)[x - 1] = ptr[x];
//				if (ptr[x] == image.ptr<int>(y + 3)[x - 3]){
//					image.ptr<int>(y + 1)[x - 1] = ptr[x];
//					image.ptr<int>(y + 2)[x - 2] = ptr[x];
//				}
//			}
//			if ((x + 3) < image.cols){
//				if (ptr[x] != image.ptr<int>(y + 1)[x + 1]) {
//					if (ptr[x] == image.ptr<int>(y + 2)[x + 2]) image.ptr<int>(y + 1)[x + 1] = ptr[x];
//					if (ptr[x] == image.ptr<int>(y + 3)[x + 3]){
//						image.ptr<int>(y + 1)[x + 1] = ptr[x];
//						image.ptr<int>(y + 2)[x + 2] = ptr[x];
//					}
//				}
//			}
//
//		}
//	}
//
//	for (int y = 0; y < image.rows - 3; y++){
//		int* ptr = image.ptr<int>(y);
//		for (int x = 0; x < image.cols - 3; x++){
//			if (ptr[x] == Background) continue;
//
//			if (ptr[x] != ptr[x + 1]){
//				if (ptr[x] == ptr[x + 2])ptr[x + 1] = ptr[x];
//				if (ptr[x] == ptr[x + 3]){
//					ptr[x + 2] = ptr[x];
//					ptr[x + 1] = ptr[x];
//				}
//			}
//
//			if (ptr[x] != image.ptr<int>(y + 1)[x]){
//				if (ptr[x] == image.ptr<int>(y + 2)[x]) image.ptr<int>(y + 1)[x] = ptr[x];
//				if (ptr[x] == image.ptr<int>(y + 3)[x]){
//					image.ptr<int>(y + 2)[x] = ptr[x];
//					image.ptr<int>(y + 1)[x] = ptr[x];
//				}
//			}
//		}
//	}
//
//	for (int y = 0; y < image.rows; y++){
//		int* ptr = image.ptr<int>(y);
//		for (int x = 0; x < image.cols; x++){
//			if (IsContourPixel(image, x, y)) ptr[x] = -1;
//		}
//	}
//
//	for (int y = 1; y < image.rows - 1; y++){
//		int* ptr = image.ptr<int>(y);
//		for (int x = 1; x < image.cols - 1; x++){
//			if (ptr[x] != -1) continue;
//
//			int temp = 0;
//			for (int y1 = y - 1; y1 <= y + 1; y1++){
//				for (int x1 = x - 1; x1 <= x + 1; x1++){
//					int pix = image.ptr<int>(y1)[x1];
//					if (pix > 0){
//						if (temp > 0){
//							if (temp != pix) image.ptr<int>(y1)[x1] = -1;
//						}
//						else temp = pix;
//					}
//				}
//			}
//
//			//int neighbours[4];
//			//if (y == 0) neighbours[0] = Background;
//			//else neighbours[0] = image.ptr<int>(y - 1)[x];
//			//if (y == image.rows - 1) neighbours[2] = Background;
//			//else neighbours[2] = image.ptr<int>(y + 1)[x];
//			//
//			//if (x == 0) neighbours[3] = Background;
//			//else neighbours[3] = ptr[x - 1];
//			//if (x == image.cols - 1) neighbours[1] = Background;
//			//else neighbours[1] = ptr[x + 1];
//
//			///*if ((neighbours[0] == neighbours[2]) && (neighbours[0] != Background) && (neighbours[0] != -1)){
//			//	if (x != image.cols - 1){
//			//		ptr[x + 1] = -1;
//			//	}				
//			//	ptr[x] = Background;
//			//}
//			//else if (neighbours[1] == neighbours[3] && (neighbours[1] != Background) && (neighbours[1] != -1)){
//			//	if (y != image.rows - 1){
//			//		image.ptr<int>(y + 1)[x] = -1;
//			//	}				
//			//	ptr[x] = Background;
//			//}*/
//			//if ((neighbours[0] > 0) && (neighbours[2] > 0) && (neighbours[0] != neighbours[2])){
//			//	image.ptr<int>(y + 1)[x] = -1;
//			//}
//			//if ((neighbours[1] > 0) && (neighbours[3] > 0) && (neighbours[1] != neighbours[3])){
//			//	ptr[x + 1] = -1;
//			//}
//			//if ((x > 0) && (x < image.cols - 1) && (y>0) && (y < image.rows - 1)){
//			//	if ((image.ptr<int>(y - 1)[x - 1]>0) && (image.ptr<int>(y+1)[x + 1] > 0) && (image.ptr<int>(y - 1)[x - 1] != image.ptr<int>(y + 1)[x + 1])){
//			//		image.ptr<int>(y + 1)[x + 1] = -1;
//			//	}
//			//	if ((image.ptr<int>(y - 1)[x + 1]>0) && (image.ptr<int>(y + 1)[x - 1] > 0) && (image.ptr<int>(y - 1)[x + 1] != image.ptr<int>(y + 1)[x - 1])){
//			//		image.ptr<int>(y + 1)[x - 1] = -1;
//			//	}
//			//}
//		}
//	}
//
//
//	/*for (int y = 1; y < image.rows - 1; y++){
//	for (int x = 1; x < image.cols - 1; x++){
//	if (image.ptr<int>(y)[x] != Background) continue;
//	int neighbours[8];
//	int* ptr = image.ptr<int>(y-1);
//	neighbours[0] = ptr[x];
//	neighbours[1] = ptr[x+1];
//	neighbours[7] = ptr[x-1];
//
//	ptr = image.ptr<int>(y+1);
//	neighbours[3] = ptr[x+1];
//	neighbours[4] = ptr[x];
//	neighbours[5] = ptr[x-1];
//
//	ptr = image.ptr<int>(y);
//	neighbours[2] = ptr[x+1];
//	neighbours[6] = ptr[x-1];
//
//	if ((neighbours[0] == neighbours[4]) && (neighbours[0] == -1) && ((neighbours[2] != neighbours[6]) || (neighbours[2] == Background))){
//	if ((neighbours[1]>0) || (neighbours[2] > 0) || (neighbours[3] > 0) || (neighbours[5] > 0) || (neighbours[6] > 0) || (neighbours[7] > 0))	ptr[x] = -2;
//	}
//	else if ((neighbours[2] == neighbours[6]) && (neighbours[2] == -1) && ((neighbours[0] != neighbours[4]) || (neighbours[0] == Background))){
//	if ((neighbours[0] > 0) || (neighbours[1] > 0) || (neighbours[3] > 0) || (neighbours[4] > 0) || (neighbours[5] > 0) || (neighbours[7] > 0))	ptr[x] = -2;
//	}
//	else if ((neighbours[1] == neighbours[5]) && (neighbours[1] == -1)){
//	if (((neighbours[2] != neighbours[6]) || (neighbours[2] == Background)) && ((neighbours[0] != neighbours[4]) || (neighbours[0] == Background))){
//	if (((neighbours[0] == neighbours[6]) && (neighbours[0] == -1)) || ((neighbours[2] == neighbours[4]) && (neighbours[2] == -1)));
//	else if ((neighbours[0] > 0) || (neighbours[2] > 0) || (neighbours[3] > 0) || (neighbours[4] > 0) || (neighbours[6] > 0) || (neighbours[7] > 0)) ptr[x] = -2;
//	}
//	}
//	else if ((neighbours[3] == neighbours[7]) && (neighbours[3] == -1)){
//	if (((neighbours[2] != neighbours[6]) || (neighbours[2] == Background)) && ((neighbours[0] != neighbours[4]) || (neighbours[0] == Background))){
//	if (((neighbours[0] == neighbours[2]) && (neighbours[0] == -1)) || ((neighbours[4] == neighbours[6]) && (neighbours[4] == -1)));
//	else if ((neighbours[0] > 0) || (neighbours[1] > 0) || (neighbours[2] > 0) || (neighbours[4] > 0) || (neighbours[5] > 0) || (neighbours[6] > 0)) ptr[x] = -2;
//	}
//	}
//	}
//	}
//
//	for (int y = 1; y < image.rows - 1; y++){
//	int* ptr = image.ptr<int>(y);
//	for (int x = 1; x < image.cols - 1; x++){
//	if (ptr[x] == -2) ptr[x] = -1;
//	}
//	}*/
//	/*for (int y = 2; y < image.rows - 2; y++){
//	int* ptr = image.ptr<int>(y);
//	for (int x = 2; x < image.cols - 2; x++){
//	if (ptr[x] != -1)continue;
//	int sum = 0;
//	int neighb = 0;
//	int temp = 0;
//
//	temp = image.ptr<int>(y - 1)[x];
//	if (temp == -1) sum++;
//	else if (temp > 0){
//	if ((image.ptr<int>(y + 2)[x] != temp) || (image.ptr<int>(y + 1)[x] != temp)) continue;
//	}
//	temp = image.ptr<int>(y + 1)[x];
//	if (temp == -1) sum++;
//	else if (temp > 0){
//	if ((image.ptr<int>(y - 2)[x] != temp) || (image.ptr<int>(y - 1)[x] != temp)) continue;
//	}
//	temp = ptr[x - 1];
//	if (temp == -1) sum++;
//	else if (temp > 0){
//	if ((ptr[x+1] != temp) || (ptr[x+2] != temp)) continue;
//	}
//	temp = ptr[x + 1];
//	if (temp == -1) sum++;
//	else if (temp > 0){
//	if ((ptr[x - 1] != temp) || (ptr[x - 2] != temp)) continue;
//	}
//	if (sum > 3) ptr[x] = Background;
//
//	}
//	}*/	
//}
//
//void ContourMap2::findContours(InputArray& MarkImage){
//
//	Mat image;
//	image = MarkImage.getMat();
//
//	/*ofstream f("2.txt");
//	for (int y = 0; y < image.rows; y++){
//		int* ptr = image.ptr<int>(y);
//		for (int x = 0; x < image.cols; x++){
//			f << ptr[x] << "\t";
//		}
//		f << endl;
//	}
//	f.close();*/
//
//	Preprocessing(image);
//
//	/*Mat temp1(image.rows, image.cols, CV_8U);
//	
//	for (int y = 0; y < temp1.rows; y++){
//		int* ptr = image.ptr<int>(y);
//		uchar* src = temp1.ptr(y);
//		for (int x = 0; x < temp1.cols; x++){
//			if (ptr[x] < 0) src[x] = 255;
//			else src[x] = 0;
//		}
//	}
//	imshow("temp", temp1);
//
//	ofstream f2("1.txt");
//	for (int y = 0; y < image.rows; y++){
//		int* ptr = image.ptr<int>(y);
//		for (int x = 0; x < image.cols; x++){
//			f2 << ptr[x] << "\t";
//		}
//		f2 << endl;
//	}
//	f2.close();*/
//		
//	for (int y = 0; y < image.rows; y++){
//		int* ptr = image.ptr<int>(y);
//		for (int x = 0; x < image.cols; x++){
//			if (ptr[x] >= 0){				
//				continue;
//			}
//
//			vector<Point2i> current;
//
//			int temp;
//			if (x != image.cols - 1) temp = ptr[x + 1];
//			else temp = 0;
//			if ((temp != Background) && (temp != -1)){
//				current = ReturnContour(image, Point2i(x, y), temp);
//				contours.push_back(current);
//			}
//			
//			if (y != image.rows - 1)  temp = image.ptr<int>(y + 1)[x];
//			else temp = 0;
//			if ((temp != Background) && (temp != -1)){
//				current = ReturnContour(image, Point2i(x, y), temp);
//				contours.push_back(current);
//			}					
//		}
//	}
//
//	for (int i = 0; i < contours.size();){
//		if (contours[i].size() < 2) contours.erase(contours.begin() + i);
//		else i++;
//	}
//	/*for (int i = 0; i < contours.size() - 1; i++){
//		if (contours[i].size() < contours[i + 1].size()) swap(contours[i], contours[i + 1]);
//	}*/
//}
