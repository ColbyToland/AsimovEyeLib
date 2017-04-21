// Code borrowed heavily from the OpenCV cailbration.cpp sample

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include <cmath>
#include <map>
#include <stdio.h>
#include <sstream>
#include <string>
#include <vector>

#include "../UtilityLib/common.hpp"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    // Variables
    string outputFilename;
    string camFilename;
    string inputFilename;

    // Command line argument parsing
    cv::CommandLineParser parser(argc, argv,
        "{help h usage ?|               | print this message        }"
        "{o             | bgimg.jpg     | output file name          }"
        "{c             |               | camera intrinsics yml     }"
        "{@vid_name     |               | video file name           }"
        );
        
    if (parser.has("help") || !parser.has("@vid_name"))
    {
        parser.printMessage();
        return 0;
    }
    
    outputFilename = parser.get<string>("o");
    camFilename = parser.get<string>("c");
    inputFilename = parser.get<string>("@vid_name");
    if (!parser.check())
    {
        parser.printMessage();
        parser.printErrors();
        return -1;
    }
    
    // Read intrinsic parameters
    Mat cam_mat;
    Mat dist_coeff;
    bool calibrated = false;
    if ( !camFilename.empty() )
    {
        calibrated = parseCamFile(camFilename, cam_mat, dist_coeff);
        if ( !calibrated )
            fprintf(stderr, "Bad camera intrinsic file\n");
    }
    
    // Open video
    VideoCapture capture;
    size_t width, height;
    capture.open(inputFilename);
    if ( !capture.isOpened() )
    {
        fprintf(stderr, "Failed to open video\n");
        return -1;
    }
    width = (size_t)capture.get(CV_CAP_PROP_FRAME_WIDTH);
    height = (size_t)capture.get(CV_CAP_PROP_FRAME_HEIGHT);

    // Read in video
    vector<Mat> vid;
    while (true)
    {
        Mat frame;
        capture >> frame;
        if ( frame.empty() ) 
            break;
            
        vid.push_back(frame);
    }
    size_t frameCount = vid.size();
    
    // Find the most common value for each pixel
    Mat bgImg(width, height, CV_8UC3);         
        // Histogram setup
        int bins = 32;
        int histSize[] = {bins, bins, bins};
        float colorranges[] = { 0, 256 };
        const float* ranges[] = { colorranges, colorranges, colorranges };
        int channels[] = {0, 1, 2};
    for ( size_t row = 0; row < height; ++row )
    {
        for ( size_t col = 0; col < width; ++col )
        {
            Mat colors(1, frameCount, CV_8UC3);
            for ( size_t frameNo = 0; frameNo < frameCount; ++frameNo )
                colors.at<Vec3b>(0,frameNo) = vid[frameNo].at<Vec3b>(row,col);
            
            Mat colorHist;
            calcHist(&colors, 1, channels, Mat(), colorHist, 3, histSize, ranges);
            
            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!
            // !!! CURRENT DEV POSITION !!!
            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            // !!! Figure out how to iterate over the histogram !!!
            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            // Find the most frequently occuring color
            int maxCount = 0;
            Scalar frequentColor = Scalar(0,0,0);
            MatIterator_<Vec3b> it, end;
            it = colorHist.begin<Vec3b>();
            end = colorHist.end<Vec3b>();
            for ( ; it != end; ++it )
            {
                if ( (*it) > maxCount )
                {
                    maxCount = *it;
                    frequentColor = it.pos()*8;
                }
            }
            
            // Store most frequently occuring color to the background
            bgImg.at<Vec3b>(row,col) = frequentColor;
        }
    }
    
    // Output best estimated background image
    imwrite(outputFilename, bgImg);
    
    capture.release();

    return 0;
}
