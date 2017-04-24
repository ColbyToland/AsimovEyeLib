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
    string bgimgFilename;

    // Command line argument parsing
    cv::CommandLineParser parser(argc, argv,
        "{help h usage ?|               | print this message        }"
        "{o             | fground.avi   | output file name          }"
        "{c             |               | camera intrinsics yml     }"
        "{@vid_name     |               | video file name           }"
        "{@bgimg        |               | bg image file             }"
        );
        
    if (parser.has("help") || !parser.has("@vid_name"))
    {
        parser.printMessage();
        return 0;
    }
    
    outputFilename = parser.get<string>("o");
    camFilename = parser.get<string>("c");
    inputFilename = parser.get<string>("@vid_name");
    bgimgFilename = parser.get<string>("@bgimg");
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
    
    // Read background image
    Mat bgimg = imread(bgimgFilename);
    
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
    
    if ( width != bgimg.cols || height != bgimg.rows )
    {
        fprintf(stderr, "Video and bg image are different sizes\n");
        return -1;
    }

    // Read in video
    VideoWriter writer;
    int codec = CV_FOURCC('M', 'J', 'P', 'G');
    double fps = 25.0;
    writer.open(outputFilename, codec, fps, bgimg.size(), true);
    if (!writer.isOpened()) {
        fprintf(stderr, "Could not open the output video file for write\n");
        return -1;
    }
    while(true)
    {
        Mat frame;
        capture >> frame;
        if ( frame.empty() ) 
            break;
            
        if (calibrated)
        {
            Mat temp;
            frame.copyTo(temp);
            undistort(temp, frame, cam_mat, dist_coeff, cam_mat);
        }
        
        Mat diffImg;
        subtract(frame,bgimg,diffImg);
        
        writer.write(diffImg);
    }
    
    cout << "success." << endl;
    
    capture.release();

    return 0;
}
