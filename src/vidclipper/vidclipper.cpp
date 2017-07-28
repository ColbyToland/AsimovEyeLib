// Code borrowed heavily from the OpenCV samples

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>

#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>

using namespace cv;
using namespace std;
int main( int argc, char** argv )
{    
    // Command line argument parsing
    cv::CommandLineParser parser(argc, argv,
        "{help h usage ?|               | print this message        }"
        "{@vid_name     |               | video file                }"
        "{@frame_num    | 0             | start frame               }"
        "{@frame_count  | 100           | frames to extract         }"
        );
        
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    
    string inputFilename = parser.get<string>("@vid_name");
    int frameNumber = parser.get<int>("@frame_num");
    int frameCount = parser.get<int>("@frame_count");
    if (!parser.check())
    {
        parser.printMessage();
        parser.printErrors();
        return -1;
    }
    
    // Open video
    VideoCapture capture;
    capture.open(inputFilename);
    if ( !capture.isOpened() )
    {
        fprintf(stderr, "Failed to open video\n");
        return -1;
    }
    double fps = capture.get(CV_CAP_PROP_FPS);
    size_t width = (size_t)capture.get(CV_CAP_PROP_FRAME_WIDTH);
    size_t height = (size_t)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    Size imgSz(width,height);
    size_t totalFrames = (size_t)capture.get(CV_CAP_PROP_FRAME_COUNT);
    
    // Open output video files        
    VideoWriter vidwriter;
    int codec = CV_FOURCC('M', 'J', 'P', 'G');
    vidwriter.open("clip.avi", codec, fps, imgSz, true);
    if ( !vidwriter.isOpened() ) 
    {
        fprintf(stderr, "Could not open an output video file for write\n");
        return -1;
    }

    // Extract subset of frames
    Mat frame;
    capture.set(CV_CAP_PROP_POS_FRAMES, frameNumber);
    for (int frameInd = frameNumber; frameInd < frameNumber+frameCount; ++frameInd)
    {
        capture >> frame;
        if ( frame.empty() ) 
            break;
        vidwriter.write(frame);
    }
    capture.release();
    
    printf("Success.\n");

    return 0;
}
