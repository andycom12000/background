#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <cvaux.h>
#include <unistd.h>
#include <string>
#include <vector>
using namespace std;
using namespace cv;
int main(int argc, char* argv[])
{
    int thresh_low = 30;


    IplImage* pImgFrame = NULL;
    IplImage* pImgProcessed = NULL;
    IplImage* pImgBackground = NULL;
    IplImage* pyrImage = NULL;

    CvMat* pMatFrame = NULL;
    CvMat* pMatProcessed = NULL;
    CvMat* pMatBackground = NULL;

    CvCapture* pCapture = NULL;

    //Create trackbar
    cvCreateTrackbar("Low","processed",&thresh_low,255,NULL);

    if( !(pCapture = cvCaptureFromCAM(CV_CAP_ANY)))
    {
        fprintf(stderr, "Can not open camera./n");
        return -2;
    }
    cvSetCaptureProperty( pCapture, CV_CAP_PROP_FRAME_WIDTH, 640 );
    cvSetCaptureProperty( pCapture, CV_CAP_PROP_FRAME_HEIGHT, 480 );


    //first frame
    pImgFrame = cvQueryFrame( pCapture );
    pImgBackground = cvCreateImage(cvSize(pImgFrame->width, pImgFrame->height),  IPL_DEPTH_8U,1);
    pImgProcessed = cvCreateImage(cvSize(pImgFrame->width, pImgFrame->height),  IPL_DEPTH_8U,1);
    pyrImage = cvCreateImage(cvSize(pImgFrame->width/2, pImgFrame->height/2),  IPL_DEPTH_8U,1);

    pMatBackground = cvCreateMat(pImgFrame->height, pImgFrame->width, CV_32FC1);
    pMatProcessed = cvCreateMat(pImgFrame->height, pImgFrame->width, CV_32FC1);
    pMatFrame = cvCreateMat(pImgFrame->height, pImgFrame->width, CV_32FC1);

    cvSmooth(pImgFrame, pImgFrame, CV_GAUSSIAN, 3, 0, 0);
    cvCvtColor(pImgFrame, pImgBackground, CV_BGR2GRAY);
    cvCvtColor(pImgFrame, pImgProcessed, CV_BGR2GRAY);

    cvConvert(pImgProcessed, pMatFrame);
    cvConvert(pImgProcessed, pMatProcessed);
    cvConvert(pImgProcessed, pMatBackground);
    cvSmooth(pMatBackground, pMatBackground, CV_GAUSSIAN, 3, 0, 0);

    cvShowImage("BG",pMatBackground);

    while((pImgFrame = cvQueryFrame( pCapture )))
    {
        double t = (double)cvGetTickCount();
     //   cvShowImage("video", pImgFrame);
        cvSmooth(pImgFrame, pImgFrame, CV_GAUSSIAN, 3, 0, 0);

        cvCvtColor(pImgFrame, pImgProcessed, CV_BGR2GRAY);
        cvConvert(pImgProcessed, pMatFrame);

        cvSmooth(pMatFrame, pMatFrame, CV_GAUSSIAN, 3, 0, 0);

#pragma omp parallel
        cvAbsDiff(pMatFrame, pMatBackground, pMatProcessed);
        //cvConvert(pMatProcessed,pImgProcessed);
        //cvThresholdBidirection(pImgProcessed,thresh_low);
#pragma omp parallel
        cvThreshold(pMatProcessed, pImgProcessed, 30, 255.0, CV_THRESH_BINARY_INV);


        cvPyrDown(pImgProcessed,pyrImage,CV_GAUSSIAN_5x5);
        cvPyrUp(pyrImage,pImgProcessed,CV_GAUSSIAN_5x5);
        //Erode and dilate
        cvErode(pImgProcessed, pImgProcessed, 0, 1);
        cvDilate(pImgProcessed, pImgProcessed, 0, 1);

        //background update
        cvRunningAvg(pMatFrame, pMatBackground, 0.0003, 0);
        cvConvert(pMatBackground, pImgBackground);


        cv::Mat foreground(pImgProcessed);
        cv::Mat frame(pImgFrame);


        // 7e. fill graph
        cv::Mat e;
        cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(18, 18));
        cv::morphologyEx(foreground, e, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1));

        // 7f. find all contours
        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(e.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        std::vector<std::vector<cv::Point> > intruders;
        for(int i = 0; i < (int)contours.size(); i++)
        {
            double area = cv::contourArea(contours[i]);
            if(area > 1000 && area < 90000)
                intruders.push_back(contours[i]);
        }

        // 7g. draw contours on a mask
        cv::Mat mask = cv::Mat::zeros(foreground.size(), CV_8UC3);
        cv::Scalar color(255, 255, 255);
        cv::drawContours(mask, intruders, -1, color, CV_FILLED);



        // 7h. distance transform
        cv::cvtColor(mask, mask, CV_RGB2GRAY);
        cv::Mat dist,skeleton(480,640,CV_8UC1);
        cv::distanceTransform(mask, dist, CV_DIST_L2, 0);
        cv::normalize(dist, dist, 0, 1, cv::NORM_MINMAX);

        //find out critical points in DT
#pragma omp parallel
		for(int i=0;i<dist.rows;i++){
			for(int j=0;j<dist.cols;j++){
				if(i==0||j==0||i==dist.rows-1||j==dist.cols-1)
					skeleton.at<uchar>(i,j)=255;
					//set to black;
				else{
					int n=0;
					if((dist.at<unsigned int>(i,j)>=dist.at<unsigned int>(i-1,j-1))&&dist.at<unsigned int>(i,j)>0)
						n++;
					if((dist.at<unsigned int>(i,j)>=dist.at<unsigned int>(i,j-1))&&dist.at<unsigned int>(i,j)>0)
						n++;
					if((dist.at<unsigned int>(i,j)>=dist.at<unsigned int>(i+1,j-1))&&dist.at<unsigned int>(i,j)>0)
						n++;
					if((dist.at<unsigned int>(i,j)>=dist.at<unsigned int>(i-1,j))&&dist.at<unsigned int>(i,j)>0)
						n++;
					if((dist.at<unsigned int>(i,j)>=dist.at<unsigned int>(i+1,j))&&dist.at<unsigned int>(i,j)>0)
						n++;
					if((dist.at<unsigned int>(i,j)>=dist.at<unsigned int>(i-1,j+1))&&dist.at<unsigned int>(i,j)>0)
						n++;
					if((dist.at<unsigned int>(i,j)>=dist.at<unsigned int>(i,j+1))&&dist.at<unsigned int>(i,j)>0)
						n++;
					if((dist.at<unsigned int>(i,j)>=dist.at<unsigned int>(i+1,j+1))&&dist.at<unsigned int>(i,j)>0)
						n++;
					if(n==8)
						skeleton.at<uchar>(i,j)=255;
					else
						skeleton.at<uchar>(i,j)=0;
				}
			}
		}
        usleep(10000);

        // compute fps
        t = (double)cvGetTickCount() - t;
        cout << 1 / (t/((double)cvGetTickFrequency()*1000000)) << endl;


        // 7i. show image
        cv::imshow( "Distance Transform", dist );
        cv::imshow( "Contours Mask", mask);
        cv::imshow( "Foregroud", foreground);
        cv::imshow( "Original Image", frame);
        cv::imshow( "Skeleton" , skeleton);
        //cvZero(pImgProcessed);
        if( cvWaitKey(10) == 27 )
        {
            break;
        }
    }

    cvDestroyWindow("video");
    cvDestroyWindow("background");
    cvDestroyWindow("processed");

    cvReleaseImage(&pImgProcessed);
    cvReleaseImage(&pImgBackground);

    cvReleaseMat(&pMatFrame);
    cvReleaseMat(&pMatProcessed);
    cvReleaseMat(&pMatBackground);

    cvReleaseCapture(&pCapture);

    return 0;
}
