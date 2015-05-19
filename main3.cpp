#include<vector>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include"function.cpp"
using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    cv::Mat frame;
    cv::Mat fg;
    cv::Mat blurred;
    cv::Mat thresholded;
    cv::Mat bgmodel;
    
    /*********************
     * Color variable of black and white
     * color_black
     * color_white
     **********************/
    cv::Scalar color_white(255, 255, 255);
    cv::Scalar color_black(0, 0, 0);
    
    // Read stream from video
    // Set the resolution to 320x240
    /*
     cv::VideoCapture cap("/home/chiahao/video/13.avi");
     const int HEIGHT = 240;
     const int WIDTH = 320;
     cap.set(CV_CAP_PROP_FRAME_HEIGHT, HEIGHT);
     cap.set(CV_CAP_PROP_FRAME_WIDTH, WIDTH);
     */
    CvCapture *pCap;
    pCap = cvCaptureFromCAM(0);
    cvSetCaptureProperty(pCap, CV_CAP_PROP_FRAME_WIDTH, 640);
    cvSetCaptureProperty(pCap, CV_CAP_PROP_FRAME_HEIGHT, 480);
    const int HEIGHT = 240;
    const int WIDTH = 320;
    
    // Background Subtraction
    /*************************
     * Parameters:
     * int history: length of history
     * float varThreshold: threshold on the squared Mahalanobis distance to describe the variance
     * bool bShadowDection: whether shadow detection should be enabled
     *************************/
    cv::BackgroundSubtractorMOG2 bgs(60, 15, false);
    
    std::vector<std::vector<cv::Point> > contours;
    
    for(;;)
    {
        // Check if stream is correctly loaded
        /*if(!cap.read(frame))
         {
         cout << "Failure" << endl;
         return -1;
         }*/
        frame = cvQueryFrame(pCap);
        
        cv::blur(frame,blurred,cv::Size(8,8));
        
        Laplacian(fg, fg, 3,1,8);
        Laplacian(blurred, blurred, 3,1,8);
        
        // Get the background image
        /*************************************
         * Parameters:
         * InputArray image: next video frame
         * OutputArray fgmask: output foreground mask as an 8-bit binary image
         * double learningRate
         **************************************/
        bgs.operator()(blurred,fg, 0.008);
        bgs.getBackgroundImage(bgmodel);
        
        // Threshold the original frame and transform into binary image
        /*******************************
         * Parameters:
         * InputArray src: single-channel, 8-bit or 32-bit floating point
         * OutputArray dst: same size and type as src
         * double threshold: threshold value_comp
         * double maxval: maximum value to use with THRESH_BINARY and THRESH_BINARY_INV
         * type: THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV
         *******************************/
        imshow("E", fg);

        
        //dilate
        
        int dilation_size = 3;
        cv::Mat element3 = cv::getStructuringElement(cv::MORPH_RECT,
                                                     cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                                     cv::Point(dilation_size, dilation_size) );
        
        cv::dilate(fg , fg, element3);
        imshow("dilate mask", fg);
        
        
        int erosion_size = 6;
        cv::Mat element2 = cv::getStructuringElement(cv::MORPH_RECT,
                                                     cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                                     cv::Point(erosion_size, erosion_size) );
        cv::erode(fg, fg, element2);
        imshow("erode mask", fg);

        
        

        
      //  Laplacian(fg, fg, 3);
        
        
        cv::threshold(fg,thresholded,60.0f,255,CV_THRESH_BINARY);
        
        // Erode the whole frame
        /*****************
         * Parameters:
         * int shape: MORTH_RECT, MORTH_ELLIPSE, MORTH_CROSS
         * Size ksize: size of the structuring element
         *****************/
        Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(2,2));
        erode(thresholded, thresholded, element);
        
        // Erode and dilate the frame
        /********************
         * Parameters:
         * InputArray src
         * OutputArray dst
         * int op: MORPH_OPEN, MORPH_CLOSE, MORPH_GRADIENT, MORPH_TOPHAT, MORPH_BLACKHAT
         * InputArray kernel
         * Point anchor
         * int iteration: times to erode and dilate
         *********************/
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(20, 18));
        cv::Mat e = thresholded.clone();
        cv::morphologyEx(thresholded,thresholded,cv::MORPH_GRADIENT, kernel, cv::Point(-1, -1), 1);
        
        
        // Draw lines
        line(thresholded, Point(0, 0), Point(0, thresholded.rows-1), color_white, 3, 8, 0);
        line(thresholded, Point(0, thresholded.rows-1), Point(thresholded.cols-1, thresholded.rows-1), color_white, 3, 8, 0);
        line(thresholded, Point(thresholded.cols-1, 0), Point(thresholded.cols-1, thresholded.rows-1), color_white, 3, 8, 0);
        
        // Find the extreme outer contour
        /********************
         * Parameters:
         * InputOutputArray src
         * OutputArrayOfArrays contours
         * int mode: CV_RETR_EXTERNAL, CV_RETR_LIST, CV_RETR_CCOMP, CV_RETR_TREE
         * int method: CV_CHAIN_APPROX_NONE, CV_CHAIN_APPROX_SIMPLE
         *******************/
        cv::findContours(thresholded.clone(),contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
        
        std::vector<std::vector<cv::Point> > intruders;
        for(int i = 0; i < (int)contours.size(); i++)
        {
            double area = cv::contourArea(contours[i]);
            if(area > 6000 )
                intruders.push_back(contours[i]);
        }
        
        cv::Mat mask = cv::Mat::zeros(thresholded.size(), CV_8UC3);
        cv::drawContours(mask, intruders, -1, color_white, CV_FILLED);

        
        line(mask, Point(0, 0), Point(0, mask.rows-1), color_black, 3, 8, 0);
        line(mask, Point(0, mask.rows-1), Point(mask.cols-1, mask.rows-1), color_black, 3, 8, 0);
        line(mask, Point(mask.cols-1, 0), Point(mask.cols-1, mask.rows-1), color_black, 3, 8, 0);
        
        cv::cvtColor(mask, mask, CV_RGB2GRAY);

        

        ///////////////////
      
        
        
        line(mask, Point(0, 0), Point(0, mask.rows-1), color_black, 3, 8, 0);
        line(mask, Point(0, mask.rows-1), Point(mask.cols-1, mask.rows-1), color_black, 3, 8, 0);
        line(mask, Point(mask.cols-1, 0), Point(mask.cols-1, mask.rows-1), color_black, 3, 8, 0);
        

        
        
        
        
        // Distance transform
        cv::Mat dist;
        cv::distanceTransform(mask.clone(), dist, CV_DIST_L2, 0);
        
        
        cv::Mat dist8u = cv::Mat(dist.size(), CV_8UC4);
        cv::normalize(dist, dist, 0, 255, cv::NORM_MINMAX);
        
        dist.convertTo(dist8u, CV_8UC4);
        
        cv::Point *tmp_max_xy = new cv::Point[intruders.size()];
        cv::Point *tmp_min_xy = new cv::Point[intruders.size()];
        
        for(int i = 0; i < intruders.size(); i++)
        {
            //min
            int tempx1 = 65535;
            //max
            int tempx2 = 0;
            //min
            int tempy1 = 65535;
            //max
            int tempy2 = 0;
            
            tmp_max_xy[i] = cv::Point(0, 0);
            tmp_min_xy[i] = cv::Point(0, 0);
            
            for(int j = 0; j < intruders[i].size(); j++)
            {
                if(intruders[i][j].y < tempy1)
                {
                    tempy1 = tmp_min_xy[i].y = intruders[i][j].y;
                }
                if(intruders[i][j].y > tempy2)
                {
                    tempy2 = tmp_max_xy[i].y = intruders[i][j].y;
                }
                if(intruders[i][j].x < tempx1)
                {
                    tempx1 = tmp_min_xy[i].x = intruders[i][j].x;
                }
                if(intruders[i][j].x > tempx2)
                {
                    tempx2 = tmp_max_xy[i].x = intruders[i][j].x;
                }
            }
            
            cv::ellipse(frame, cv::Point(tempx1, tempy1), cv::Size(5, 5), 0, 0, 360, cv::Scalar(0, 0, 255), -1, 8);
            cv::ellipse(frame, cv::Point(tempx2, tempy2), cv::Size(5, 5), 0, 0, 360, cv::Scalar(0, 0, 255), -1, 8);
            cv::ellipse(frame, cv::Point(tempx1/2 + tempx2/2, tempy1/2 + tempy2/2), cv::Size(5, 5), 0, 0, 360, cv::Scalar(0, 0, 255), -1, 8);
        }
        
        int Xmax = 0, Ymax = 0;
        if(intruders.size() != 0)
        {
            Xmax = (tmp_max_xy[0].x + tmp_min_xy[0].x) / 2;
            Ymax = (tmp_max_xy[0].y + tmp_min_xy[0].y) / 2;
        }
        
        //find LMPs in DistanceTransform-Mat
        cv::Mat newdist = dist.clone();
        cv::Point *lightPoints;
        
        lightPoints = find_LMP(HEIGHT,WIDTH,newdist);
        
        cv::Point head;
        cv::Point hip;
        
        for(int i = 0; i < 10; i++)
        {
            double ydistance = lightPoints[i].y > Ymax ? lightPoints[i].y - Ymax : Ymax - lightPoints[i].y;
            double xdistance = lightPoints[i].x > Xmax ? lightPoints[i].x - Xmax : Xmax - lightPoints[i].x;
            if(ydistance < 300 && xdistance < 100)
            {
                head.x = lightPoints[i].x;
                head.y = lightPoints[i].y;
                hip.x = (head.x * 5 + Xmax * 3) / 8;
                hip.y = (head.y * 5 + Ymax * 3) / 8;
                break;
            }
        }
        delete[] lightPoints;
        lightPoints = NULL;
        
        int left_x = 0, right_x = 0;
        int limit_left_dis = 0, limit_right_dis = 0;
        
        for(int i = Xmax; i >= 0; i--)
        {
            if(dist.at<int>(Ymax + 80 >= HEIGHT ? HEIGHT-1 : Ymax + 80, i) <= 220)
            {
                limit_left_dis = hip.x - i;
                break;
            }
        }
        for(int i = Xmax; i < WIDTH; i++)
        {
            if(dist.at<int>(Ymax + 80 >= HEIGHT ? HEIGHT-1 : Ymax + 80, i) <= 220)
            {
                limit_right_dis = i - hip.x;
                break;
            }
        }
        
        
        for(int i = hip.x; i >= 0; i--)
        {
            if((dist.at<int>(hip.y, i) <= 220 && (hip.x - i) < limit_left_dis) || (hip.x - i) > limit_left_dis)
            {
                left_x = i;
                break;
            }
        }
        for(int i = hip.x; i < WIDTH; i++)
        {
            if((dist.at<int>(hip.y, i) <= 220 && (i - hip.x) < limit_right_dis) || (i - hip.x) > limit_right_dis)
            {
                right_x = i;
                break;
            }
        }
        
        
        cv::ellipse(frame, head, cv::Size(5, 5), 0, 0, 360, cv::Scalar(0, 255, 0), -1, 8);
        cv::ellipse(frame, hip, cv::Size(5, 5), 0, 0, 360, cv::Scalar(0, 255, 0), -1, 8);
        
        cv::ellipse(frame, cv::Point(left_x+40, hip.y), cv::Size(5, 5), 0, 0, 360, cv::Scalar(255, 255, 0), -1, 8);
        cv::ellipse(frame, cv::Point(right_x-40, hip.y), cv::Size(5, 5), 0, 0, 360, cv::Scalar(255, 255, 0), -1, 8);
        
        cv::imshow("Skeleton",frame);
        cv::imshow("Mask", mask);
        
        if(cv::waitKey(30) >= 0) break;
    }
    
    //cap.release();
    cvReleaseCapture(&pCap);
    
    return 0;
}

