//motionTracking.cpp

//Written by  Kyle Hounslow, December 2013

//Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software")
//, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
//and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

//The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//IN THE SOFTWARE.

#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2\highgui.hpp>
#include <opencv2\tracking.hpp>
#include <opencv2\features2d.hpp>
#include <opencv2\videoio.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\video.hpp>
#define LIVE 0
#if LIVE
#define VIDEOSRC  "mmsh://sv04msmedia2.dot.ca.gov/D5-Los-Osos-Valley-Rd-at-101?MSWMExt=.asf"
#else
#define VIDEOSRC "output.avi"
#endif

using namespace std;
using namespace cv;

//our sensitivity value to be used in the absdiff() function
const static int SENSITIVITY_VALUE = 20;
//size of blur used to smooth the intensity image output from absdiff() function
const static int BLUR_SIZE = 10;
//we'll have just one object to search for
//and keep track of its position.
int theObject[2] = { 0,0 };
//bounding rectangle of the object, we will use the center of this as its position.
Rect objectBoundingRectangle = Rect(0, 0, 0, 0);



//int to string helper function
string intToString(int number) {

	//this function has a number input and string output
	std::stringstream ss;
	ss << number;
	return ss.str();
}

void searchForMovement(Mat thresholdImage, Mat &cameraFeed) {
	//notice how we use the '&' operator for objectDetected and cameraFeed. This is because we wish
	//to take the values passed into the function and manipulate them, rather than just working with a copy.
	//eg. we draw to the cameraFeed to be displayed in the main() function.
	bool objectDetected = false;
	Mat temp;
	thresholdImage.copyTo(temp);
	//these two vectors needed for output of findContours
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//find contours of filtered image using openCV findContours function
	//findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );// retrieves all contours
	findContours(temp, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);// retrieves external contours

																					  //if contours vector is not empty, we have found some objects
	if (contours.size() > 0) objectDetected = true;
	else objectDetected = false;


	double minArea = 450;

	double totalarea = 0, trucksize = 10000;
	int largeboxes = 0;
	int totalboxes = 0;
	if (objectDetected) {
		for (vector<vector<Point>>::iterator it = contours.begin(); it != contours.end(); ++it) {


			//make a bounding rectangle around the largest contour then find its centroid
			//this will be the object's final estimated position.
			objectBoundingRectangle = boundingRect(*it);
			double area = objectBoundingRectangle.area();
			
			if ( area < minArea) continue;
			if (area > trucksize) {
				largeboxes += 1;
				rectangle(cameraFeed, objectBoundingRectangle, Scalar(255, 0, 0), 1, 8, 0);
				putText(cameraFeed, to_string(area), Point(objectBoundingRectangle.x, objectBoundingRectangle.y), 1, 1, Scalar(255, 0, 0), 1);
			}
			else {
				rectangle(cameraFeed, objectBoundingRectangle, Scalar(0, 255, 0), 1, 8, 0);
			}
			totalarea += area;
			totalboxes += 1;
			
			
			/*int xpos = objectBoundingRectangle.x + objectBoundingRectangle.width / 2;
			int ypos = objectBoundingRectangle.y + objectBoundingRectangle.height / 2;

			//update the objects positions by changing the 'theObject' array values
			theObject[0] = xpos, theObject[1] = ypos;

			//make some temp x and y variables so we dont have to type out so much
			int x = theObject[0];
			int y = theObject[1];

			//draw some crosshairs around the object
			circle(cameraFeed, Point(x, y), 20, Scalar(0, 255, 0), 2);
			line(cameraFeed, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
			line(cameraFeed, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
			line(cameraFeed, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
			line(cameraFeed, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);
			
			//write the position of the object to the screen
			putText(cameraFeed, "Tracking object at (" + intToString(x) + "," + intToString(y) + ")", Point(x, y), 1, 1, Scalar(255, 0, 0), 2);
			*/
		}
		rectangle(cameraFeed, Point(1, 460), Point(420, 480), Scalar(0, 0, 0), CV_FILLED);
		putText(cameraFeed, "numboxes: " + to_string(totalboxes) + " NumTrucks: " + to_string(largeboxes) + " totalarea: " + to_string((int)totalarea),
			Point(1, 475), 1, 1, Scalar(0, 0, 255), 1);
	}
}

int main() {

	//some boolean variables for added functionality
	bool objectDetected = false;
	//these two can be toggled by pressing 'd' or 't'
	bool debugMode = true;
	bool trackingEnabled = true;
	//pause and resume code
	bool pause = false;
	//set up the matrices that we will need
	//the two frames we will be comparing
	Mat frame1, frame2, this_frame;
	//their grayscale images (needed for absdiff() function)
	Mat grayImage1, grayImage2;
	//resulting difference image
	Mat differenceImage;
	//thresholded difference image (for use in findContours() function)
	Mat thresholdImage;
	//video capture object.
	VideoCapture capture(VIDEOSRC);
	Mat fgMask;
	Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2();
	//VideoWriter writer("output3.avi",
	//	VideoWriter::fourcc('X', 'V', 'I', 'D'), 15, cvSize(640, 480), 1);

	if (!capture.isOpened())
		return -1;

	for (;;) {

		capture >> this_frame;
		if (this_frame.empty()) {
			break;
		}

		cvtColor(this_frame, grayImage1, COLOR_BGR2GRAY);
		GaussianBlur(grayImage1, grayImage1, cvSize(7, 7), 0, 0);
		pMOG2->apply(grayImage1, grayImage1);

		/*if (debugMode == true) {
			//show the difference image and threshold image
			cv::imshow("Difference Image", differenceImage);
			cv::imshow("Threshold Image", grayImage1);
		}
		else {
			//if not in debug mode, destroy the windows so we don't see them anymore
			cv::destroyWindow("Difference Image");
			cv::destroyWindow("Threshold Image");
		}*/
		//blur the image to get rid of the noise. This will output an intensity image

		//cv::blur(, thresholdImage, cv::Size(BLUR_SIZE, BLUR_SIZE));

		//threshold again to obtain binary image from blur output
		//cv::threshold(thresholdImage, thresholdImage, SENSITIVITY_VALUE, 255, THRESH_BINARY);

		//if tracking enabled, search for contours in our thresholded image
		if (trackingEnabled) {
			searchForMovement(grayImage1, this_frame);
		}

		//show our captured frame
		imshow("Frame1", this_frame);
		
		//writer.write(this_frame);
		//check to see if a button has been pressed.
		//this 10ms delay is necessary for proper operation of this program
		//if removed, frames will not have enough time to referesh and a blank 
		//image will appear.

		if (waitKey(50) == 27) {
			return 0;
		}
	}
	//release the capture before re-opening and looping again.
	waitKey(0);

	capture.release();
	//writer.release();


	return 0;

}