// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"

using namespace std;


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

void testAdditiveFactor(int additiveFactor) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		double t = (double)getTickCount();

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				uchar grayVal = src.at<uchar>(i, j);
				int newGrayVal = grayVal + additiveFactor;

				if (newGrayVal < 0) newGrayVal = 0;
				else if (newGrayVal > 255) newGrayVal = 255;
				
				uchar gray = newGrayVal;
				dst.at<uchar>(i, j) = gray;
			}
		}

		t = ((double)getTickCount() - t) / getTickFrequency();

		printf("Time = %.3f[ms]\n", t * 1000);

		imshow("input image", src);
		imshow("output image", dst);
		waitKey(0);
	}
}

void testMultiplicativeFactor(int multiplicativeFactor){
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		double t = (double)getTickCount();

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				uchar grayVal = src.at<uchar>(i, j);
				int newGrayVal = grayVal * multiplicativeFactor;

				if (newGrayVal < 0) newGrayVal = 0;
				else if (newGrayVal > 255) newGrayVal = 255;

				uchar gray = static_cast<uchar>(newGrayVal);
				dst.at<uchar>(i, j) = gray;
			}
		}

		t = ((double)getTickCount() - t) / getTickFrequency();

		printf("Time = %.3f[ms]\n", t * 1000);

		imshow("input image", src);
		imshow("output image", dst);
		waitKey(0);

		imwrite("C:/Users/gheor/Desktop/Facultate/CTI An 3/Sem 2/PI/OpenCVApplication-VS2017_OCV340_basic/Images/multiplicativeFactor.bmp", dst);
	}
}

void newImg() {
	Mat image(256, 256, CV_8UC3, Scalar(0, 0, 0));

	Rect topLeft(0, 0, image.cols / 2, image.rows / 2);
	Rect topRight(image.cols / 2, 0, image.cols / 2, image.rows / 2);
	Rect bottomLeft(0, image.rows / 2, image.cols / 2, image.rows / 2);
	Rect bottomRight(image.cols / 2, image.rows / 2, image.cols / 2, image.rows / 2);
	
	image(topLeft) = Scalar(255, 255, 255);
	image(topRight) = Scalar(0, 0, 255);
	image(bottomLeft) = Scalar(0, 255, 0);
	image(bottomRight) = Scalar(0, 255, 255);

	imshow("Image", image);
	waitKey(0);

	imwrite("C:/Users/gheor/Desktop/Facultate/CTI An 3/Sem 2/PI/OpenCVApplication-VS2017_OCV340_basic/Images/newImage.bmp", image);
}

void newMatrix() {
	Mat matrix = Mat::zeros(3, 3, CV_32F);

	matrix.at<float>(0, 0) = -1.0f;
	matrix.at<float>(0, 1) = 2.0f;
	matrix.at<float>(0, 2) = 1.0f;
	matrix.at<float>(1, 0) = 3.0f;
	matrix.at<float>(1, 1) = 0.0f;
	matrix.at<float>(1, 2) = 2.0f;
	matrix.at<float>(2, 0) = 4.0f;
	matrix.at<float>(2, 1) = -1.0f;
	matrix.at<float>(2, 2) = 2.0f;

	double detA = matrix.at<float>(0, 0)* matrix.at<float>(1, 1)* matrix.at<float>(2, 2)
		+ matrix.at<float>(1, 0) * matrix.at<float>(2, 1) * matrix.at<float>(0, 2)
		+ matrix.at<float>(0, 1) * matrix.at<float>(1, 2) * matrix.at<float>(2, 0)
		- matrix.at<float>(0, 2) * matrix.at<float>(1, 1) * matrix.at<float>(2, 0)
		- matrix.at<float>(1, 2) * matrix.at<float>(2, 1) * matrix.at<float>(0, 0)
		- matrix.at<float>(0, 1) * matrix.at<float>(1, 0) * matrix.at<float>(2, 2);

	Mat transposedMatrix = Mat::zeros(3, 3, CV_32F);
	for (int i = 0; i < matrix.rows; ++i) {
		for (int j = 0; j < matrix.cols; ++j) {
			transposedMatrix.at<float>(j, i) = matrix.at<float>(i, j);
		}
	}

	Mat adjointMatrix = Mat::zeros(3, 3, CV_32F);
	adjointMatrix.at<float>(0, 0) = transposedMatrix.at<float>(1, 1) * transposedMatrix.at<float>(2, 2) - transposedMatrix.at<float>(1, 2) * transposedMatrix.at<float>(2, 1);
	adjointMatrix.at<float>(0, 1) = transposedMatrix.at<float>(0, 2) * transposedMatrix.at<float>(2, 1) - transposedMatrix.at<float>(0, 1) * transposedMatrix.at<float>(2, 2);
	adjointMatrix.at<float>(0, 2) = transposedMatrix.at<float>(0, 1) * transposedMatrix.at<float>(1, 2) - transposedMatrix.at<float>(0, 2) * transposedMatrix.at<float>(1, 1);
	adjointMatrix.at<float>(1, 0) = transposedMatrix.at<float>(1, 2) * transposedMatrix.at<float>(2, 0) - transposedMatrix.at<float>(1, 0) * transposedMatrix.at<float>(2, 2);
	adjointMatrix.at<float>(1, 1) = transposedMatrix.at<float>(0, 0) * transposedMatrix.at<float>(2, 2) - transposedMatrix.at<float>(0, 2) * transposedMatrix.at<float>(2, 0);
	adjointMatrix.at<float>(1, 2) = transposedMatrix.at<float>(0, 2) * transposedMatrix.at<float>(1, 0) - transposedMatrix.at<float>(0, 0) * transposedMatrix.at<float>(1, 2);
	adjointMatrix.at<float>(2, 0) = transposedMatrix.at<float>(1, 0) * transposedMatrix.at<float>(2, 1) - transposedMatrix.at<float>(1, 1) * transposedMatrix.at<float>(2, 0);
	adjointMatrix.at<float>(2, 1) = transposedMatrix.at<float>(0, 1) * transposedMatrix.at<float>(2, 0) - transposedMatrix.at<float>(0, 0) * transposedMatrix.at<float>(2, 1);
	adjointMatrix.at<float>(2, 2) = transposedMatrix.at<float>(0, 0) * transposedMatrix.at<float>(1, 1) - transposedMatrix.at<float>(0, 1) * transposedMatrix.at<float>(1, 0);

	Mat invMatrix = adjointMatrix / detA;

	std::cout << "Matricea inversa:\n";
	for (int i = 0; i < invMatrix.rows; ++i) {
		for (int j = 0; j < invMatrix.cols; ++j) {
			std::cout << invMatrix.at<float>(i, j) << "\t";
		}
		std::cout << std::endl;
	}
	waitKey(0);
}

void split_RGB()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		Mat dstRed = Mat(src.rows, src.cols, CV_8UC3);
		Mat dstGreen = Mat(src.rows, src.cols, CV_8UC3);
		Mat dstBlue = Mat(src.rows, src.cols, CV_8UC3);

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];

				dstRed.at<Vec3b>(i, j) = Vec3b(0, 0, r);
				dstGreen.at<Vec3b>(i, j) = Vec3b(0, g, 0);
				dstBlue.at<Vec3b>(i, j) = Vec3b(b, 0, 0);
			}
		}

		imshow("Initial image", src);
		imshow("Red image", dstRed);
		imshow("Green image", dstGreen);
		imshow("Blue image", dstBlue);
		
		waitKey(0);
	}
}

void RGB_to_Grayscale() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_COLOR);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("Initial image", src);
		imshow("Grayscale image", dst);
		waitKey(0);
	}
}

void grayscale_to_BlackWhite(int threshold) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				uchar initValue = src.at<uchar>(i, j);
				uchar newValue;

				if (initValue < threshold)
					newValue = 0;
				else newValue = 255;

				dst.at<uchar>(i, j) = newValue;
			}
		}
		
		imshow("Initial image", src);
		imshow("Black or White image", dst);
		waitKey(0);
	}
}

void RGB_to_HSV() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_COLOR);
		Mat dstH = Mat(src.rows, src.cols, CV_8UC1);
		Mat dstS = Mat(src.rows, src.cols, CV_8UC1);
		Mat dstV = Mat(src.rows, src.cols, CV_8UC1);

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				Vec3b vec = src.at<Vec3b>(i, j);
				float r = (float)vec[2] / 255;
				float g = (float)vec[1] / 255;
				float b = (float)vec[0] / 255;

				float M = max(max(r, g), b);
				float m = min(min(r, g), b);

				float C = M - m;
				float V = M;
				float S, H;

				if (V != 0)
					S = C / V;
				else S = 0;

				if (C != 0) {
					if (M == r) H = 60 * (g - b) / C;
					else if (M == g) H = 120 + 60 * (b - r) / C;
					else if (M == b) H = 240 + 60 * (r - g) / C;
				}
				else H = 0;
				if (H < 0) H += 360;

				uchar H_norm = H * 255 / 360;
				uchar S_norm = S * 255;
				uchar V_norm = V * 255;

				dstH.at<uchar>(i, j) = H_norm;
				dstS.at<uchar>(i, j) = S_norm;
				dstV.at<uchar>(i, j) = V_norm;
			}
		}

		imshow("Initial image", src);
		imshow("Hue image", dstH);
		imshow("Saturation image", dstS);
		imshow("Value image", dstV);
		waitKey(0);
	}
}

int isInside(Mat img, int i, int j) {
	bool okI, okJ, ok;

	if (i > 0 && i < img.rows) okI = 1;
	else okI = 0;
	if (j > 0 && j < img.cols) okJ = 1;
	else okJ = 0;

	if (okI == 0 && okJ == 0) ok = 0;
	else if (okI == 0 && okJ == 1) ok = 0;
	else if (okI == 1 && okJ == 0) ok = 0;
	else if (okI == 1 && okJ == 1) ok = 1;

	return ok;
}

int testIsInsider(Mat img, int i, int j) {
	if (isInside(img, i, j) == 0) return FALSE;
	else return TRUE;
}


/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}


int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Additive factor\n");
		printf(" 11 - Multiplicative factor\n");
		printf(" 12 - New imagine\n");
		printf(" 13 - New matrix\n");
		printf(" 14 - Split RGB\n");
		printf(" 15 - RGB to grayscale\n");
		printf(" 16 - Grayscale to Black and White\n");
		printf(" 17 - RGB to HSV\n");
		printf(" 18 - Position verif\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				int x;
				std::cin >> x;
				testAdditiveFactor(x);
				break;
			case 11:
				int a;
				std::cin >> a;
				testMultiplicativeFactor(a);
				break;
			case 12:
				newImg();
				break;
			case 13:
				newMatrix();
				break;
			case 14:
				split_RGB();
				break;
			case 15:
				RGB_to_Grayscale();
				break;
			case 16:
				int b;
				std::cin >> b;
				grayscale_to_BlackWhite(b);
				break;
			case 17:
				RGB_to_HSV();
				break;
			case 18:
				int i, j;
				std::cin >> i >> j;
				char fname[MAX_PATH];
				while (openFileDlg(fname)) {
					Mat img = imread(fname, IMREAD_ANYCOLOR);
					
					if (testIsInsider(img, i, j) == 1) printf("It's inside\n");
					else printf("It's outside\n");

					imshow("Image", img);
				}
		}
	}
	while (op!=0);
	return 0;
}