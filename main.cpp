#include "pch.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int main()
{
	string filename1 = "out_camera_dataL.xml";
	string filename2 = "out_camera_dataR.xml";

	int minDisparity(0), numDisparities(1), blockSize(3), p1(0), p2(0), setDisp12MaxDiff(0), setPreFilterCap(0), uniquenessRatio(5), speckleWindowSize(50), speckleRange(0);
	VideoCapture leftCam(1);
	if (leftCam.isOpened() == false) { cout << "error: Webcam connect unsuccessful\n";    return(0); }

	VideoCapture rightCam(0);
	if (rightCam.isOpened() == false) { cout << "error: Webcam connect unsuccessful\n";   return(0); }

	Mat left, right;

	char charCheckForEscKey = 0;

	while (charCheckForEscKey != 27 && leftCam.isOpened() && rightCam.isOpened())
	{

		leftCam.read(left);
		if (left.empty()) { cout << "No frame to read" << endl;  break; }
		Mat im_left = left.clone();

		rightCam.read(right);
		if (right.empty()) { cout << "No frame to read" << endl;  break; }
		Mat im_right = right.clone();

		//считываем разрешение изображения
		Size imagesize = im_left.size();
		//создаем матрицы под разрешение. CV_16S так как значение пикселей может быть отрицательным до - 255
		Mat disparity_left = Mat(imagesize.height, imagesize.width, CV_16S);

		Mat g1, g2, disp, disp8, outImgLeft, outImgRight;

		Mat cameraMatrix, distCoeffs;

		readCameraParameters(filename1, cameraMatrix, distCoeffs);
		undistort(im_left, outImgLeft, cameraMatrix, distCoeffs);

		readCameraParameters(filename2, cameraMatrix, distCoeffs);
		undistort(im_right, outImgRight, cameraMatrix, distCoeffs);

		cvtColor(outImgLeft, g1, COLOR_BGR2GRAY);
		cvtColor(outImgRight, g2, COLOR_BGR2GRAY);

		
		cv::Ptr<cv::StereoSGBM> sgbm = StereoSGBM::create(
			minDisparity, //minDisparity Наименьшее несоответствие, которое принимается во внимание, равен нулю, если не применен алгоритм ректификации
			16 * numDisparities, //numDisparities максимальное несоответствие - минимальное, всегда > 0 и кратно 16                 ((img_size.width / 8) + 15) & -16
			blockSize, //blockSize нечет, соответсвует размеру блока, диапозон 3 - 11
			p1, //p1 параметр, отвечающий за гладкость, штраф за изменение диспаритета на единицу (8 * cn*sgbmWinSize*sgbmWinSize, где cn это цветовые каналы изображения)
			p2, //p2 параметр, отвечающий за гладкость, штраф за изменение диспаритета более чем на единицу, он больше предыдущего (32 * cn*sgbmWinSize*sgbmWinSize, где cn это цветовые каналы изображения)
			setDisp12MaxDiff, //setDisp12MaxDiff максимально допустимая разница при проверке несоответствия
			setPreFilterCap,//setPreFilterCap значение усечения для предворительно обработанных пикселей изображения
			uniquenessRatio, //uniquenessRatio диапозон 5 - 15%
			speckleWindowSize, //speckleWindowSize максимальный размер гладких областей диспаритета, которые считаются шумом, диапозон 50 - 200
			speckleRange,//speckleRange фильтрация спеклов - такая оптическая хаотичная картина, обычно 1 или 2 норм. будет умножена на 16
			StereoSGBM::MODE_SGBM_3WAY//mode метод
		);

		cv::namedWindow("Track Bar Window", CV_WINDOW_NORMAL);
		cvCreateTrackbar("minDisparity", "Track Bar Window", &minDisparity, 100);
		cvCreateTrackbar("numDisparities", "Track Bar Window", &numDisparities, 5);
		cvCreateTrackbar("blockSize", "Track Bar Window", &blockSize, 11);
		cvCreateTrackbar("p1", "Track Bar Window", &p1, 200);
		cvCreateTrackbar("p2", "Track Bar Window", &p2, 200);
		cvCreateTrackbar("setDisp12MaxDiff", "Track Bar Window", &setDisp12MaxDiff, 255);
		cvCreateTrackbar("setPreFilterCap", "Track Bar Window", &setPreFilterCap, 100);
		cvCreateTrackbar("uniquenessRatio", "Track Bar Window", &uniquenessRatio, 15);
		cvCreateTrackbar("Speckle Window Size", "Track Bar Window", &speckleWindowSize, 200);
		cvCreateTrackbar("Speckle Range", "Track Bar Window", &speckleRange, 5);

		if (blockSize % 2 == 0)
		{
			blockSize = blockSize + 1;
		}

		sgbm->compute(g1, g2, disparity_left);

		normalize(disparity_left, disp8, 0, 255, CV_MINMAX, CV_8U);
		
		
		namedWindow("Left", WINDOW_AUTOSIZE);
		imshow("Left", outImgLeft);

		namedWindow("Right", WINDOW_AUTOSIZE);
		imshow("Right", outImgRight);
		
		
		namedWindow("Disparity map", WINDOW_AUTOSIZE);
		imshow("Disparity map", disp8);

		namedWindow("Depth map", WINDOW_AUTOSIZE);
		imshow("Depth map", disparity_left/2048);
		

		charCheckForEscKey = waitKey(1);
	}

	return(0);
}