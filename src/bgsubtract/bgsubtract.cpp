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
    
    int startFrame, totalFrames;
    bool allFrames = false;

    // Command line argument parsing
    cv::CommandLineParser parser(argc, argv,
        "{help h usage ?|               | print this message        }"
        "{o             | fground.avi   | output file name          }"
        "{c             |               | camera intrinsics yml     }"
        "{s             | 0             | start frame number        }"
        "{n             | 1000          | total frames to convert   }"        
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
    startFrame = parser.get<int>("s");
    totalFrames = parser.get<int>("n");
    allFrames = !parser.has("n");
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
    
    // Open output video file
    VideoWriter writer;
    int codec = CV_FOURCC('M', 'J', 'P', 'G');
    double fps = 25.0;
    writer.open(outputFilename, codec, fps, bgimg.size(), true);
    if ( !writer.isOpened() ) 
    {
        fprintf(stderr, "Could not open the output video file for write\n");
        return -1;
    }

    // Process the video and store it
    capture.set(CV_CAP_PROP_POS_FRAMES, startFrame);
    for (int frameNo = 0; frameNo < totalFrames; ++frameNo)
    {
        // Reset the iterating variable to convert this to a while(true) loop
        if ( allFrames ) frameNo = 0;
    
        // Read and check
        Mat frame;
        capture >> frame;
        if ( frame.empty() ) 
            break;
            
        // Undistort if necessary
        if (calibrated)
        {
            Mat temp;
            frame.copyTo(temp);
            undistort(temp, frame, cam_mat, dist_coeff, cam_mat);
        }
        
        // Sadly basic bg subtraction
        Mat diffImg;
        subtract(frame,bgimg,diffImg);
        
        // Store frame
        writer.write(diffImg);
    }
    
    cout << "success." << endl;
    
    capture.release();

    return 0;
}
