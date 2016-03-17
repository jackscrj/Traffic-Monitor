#include <opencv2/highgui.hpp>
#include <opencv2\tracking.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;
int do_frame_diff(void);
int playvid(void);

int main(int argc, char** argv) {
	return do_frame_diff();
	//return playvid();
}

/** @function main */
int do_frame_diff(void)
{
	string videosrc = "vid1.avi";
	//int min_area = 150;
	Mat prev_frame, cur_frame, next_frame, this_frame,
		source_frame, d1_frame, d2_frame, delta_frame, thresh_frame,
		dialted_frame, with_keypoints_frame;
	VideoCapture cap(videosrc);
	Mat fgMask;
	Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2();

	if (!cap.isOpened())
		return -1;

	char* source_window = "Source";
	char* framd_window = "FrameDelta";
	char* thresh_window = "threshold";
	char* test_window = "test";
	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	namedWindow(framd_window, CV_WINDOW_AUTOSIZE);
	namedWindow(thresh_window, CV_WINDOW_AUTOSIZE);
	namedWindow(test_window, CV_WINDOW_AUTOSIZE);

	SimpleBlobDetector::Params params;
	params.minThreshold = 10;
	params.maxThreshold = 255;

	params.filterByArea = true;
	params.minArea = 500;
	params.filterByCircularity = false;
	params.filterByConvexity = false;
	params.filterByInertia = false;


	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
	std::vector<KeyPoint> keypoints;

	/*
	//initialize
	cap >> prev_frame;
	//prev_frame.copyTo(source_frame);
	cvtColor(prev_frame, prev_frame, COLOR_BGR2GRAY, 0);
	GaussianBlur(prev_frame, prev_frame, cvSize(7, 7), 0, 0);


	prev_frame.copyTo(cur_frame);
	prev_frame.copyTo(next_frame);
	prev_frame.copyTo(next_frame);
	*/

	for (;;) {
		//prev_frame = cur_frame;
		//cur_frame = next_frame;
		cap >> this_frame;
		if (this_frame.empty()) {
			break;
		}

		this_frame.copyTo(source_frame);
		cvtColor(this_frame, this_frame, COLOR_BGR2GRAY, 0);
		GaussianBlur(this_frame, this_frame, cvSize(7, 7), 0, 0);

		//absdiff(next_frame, cur_frame, d1_frame);
		//absdiff(prev_frame, cur_frame, d2_frame);
		//addWeighted(d1_frame, 1, d2_frame, 1, 0.0, delta_frame);

		//threshold(delta_frame, thresh_frame, 10, 255, CV_THRESH_BINARY);


		pMOG2->apply(this_frame, fgMask);


		//erode(fgMask, thresh_frame, getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(-1, -1)));
		//dilate(thresh_frame, dialted_frame, getStructuringElement(MORPH_ELLIPSE, Size(3,3), Point(-1,-1)));





		bitwise_not(dialted_frame, dialted_frame);

		detector->detect(dialted_frame, keypoints);
		drawKeypoints(source_frame, keypoints, with_keypoints_frame, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);


		imshow(source_window, with_keypoints_frame);
		imshow(framd_window, this_frame);
		imshow(thresh_window, fgMask);
		imshow(test_window, dialted_frame);

		source_frame = this_frame;
		if (waitKey(10) >= 0) break;
	}


	return(0);
}

int playvid(void) {
	string videosrc = "vid1.avi";
	string livefeedsrc = "mmsh://sv04msmedia2.dot.ca.gov/D5-Los-Osos-Valley-Rd-at-101?MSWMExt=.asf";
	Mat frame;
	VideoCapture cap(livefeedsrc);

	if (!cap.isOpened())
		return -1;

	char* source_window = "Source";
	namedWindow(source_window, CV_WINDOW_AUTOSIZE);


	for (;;) {
		cap >> frame;
		if (frame.empty()) {
			break;
		}

		imshow(source_window, frame);

		if (waitKey(67) >= 0) break;
	}
}