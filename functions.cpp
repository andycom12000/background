#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

cv::Point* find_LMP(const int HEIGHT,const int WIDTH, const cv::Mat newdist){
    
    unsigned int count = 0;
    bool light[HEIGHT][WIDTH];
    
    for(int i = 0 ; i < HEIGHT ; i++)
        for(int j = 0 ; j < WIDTH ; j++)
            light[i][j] = false;
    
    for(int i = 1; i < HEIGHT - 1; i++)
        for(int j = 1; j < WIDTH - 1; j++)
        {
            int temp1 = newdist.at<int>(i-1, j-1);
            int temp2 = newdist.at<int>(i-1, j);
            int temp3 = newdist.at<int>(i-1, j+1);
            int temp4 = newdist.at<int>(i, j-1);
            int temp5 = newdist.at<int>(i, j);
            int temp6 = newdist.at<int>(i, j+1);
            int temp7 = newdist.at<int>(i+1, j-1);
            int temp8 = newdist.at<int>(i+1, j);
            int temp9 = newdist.at<int>(i+1, j+1);
            if(temp5 >= temp1 && temp5 >= temp2 && temp5 >= temp3 &&
               temp5 >= temp4 && temp5 >= temp6 && temp5 >= temp7 &&
               temp5 >= temp8 && temp5 >= temp9 && temp5 >= 40)
            {
                light[i][j] = true;
                count++;
            }
            else
            {
                light[i][j] = false;
            }
        }

    
    cv::Point *lightPoints = new cv::Point[count];
    int k = 0;
    for(int i = 0; i < HEIGHT; i++)
    {
        for(int j = 0; j < WIDTH; j++)
        {
            if(light[i][j])
            {
                lightPoints[k++] = cv::Point(j, i);
            }
        }
    }
    return lightPoints;
}