// Code borrowed heavily from the OpenCV cailbration.cpp sample

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

static void help()
{
    printf( "This is a camera calibration sample.\n"
        "Usage: framegrabber\n"
        "     [-o=<out_image>]         # the output filename for the grabbed frame\n"
        "     [vid_name]               # input video\n"
        "     [frame_num]              # frame number to extract\n"
        "\n" );
}

int main( int argc, char** argv )
{
    // Variables
    string outputFilename;
    string inputFilename = "";
    
    int frameNumber;

    VideoCapture capture;

    // Command line argument parsing
    cv::CommandLineParser parser(argc, argv,
        "{help h usage ?|               | print this message        }"
        "{o             | newFrame.jpg  | output image name         }"
        "{@vid_name     |               | video file                }"
        "{@frame_num    | 0             | frame to extract          }"
        );
        
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    
    outputFilename = parser.get<string>("o");
    inputFilename = parser.get<string>("@vid_name");
    frameNumber = parser.get<int>("@frame_num");
    if (!parser.check())
    {
        help();
        parser.printErrors();
        return -1;
    }
    
    // Open video
    if( !inputFilename.empty() )
        capture.open(inputFilename);

    if( !capture.isOpened() )
        return fprintf( stderr, "Could not open video.\n" ), -2;

    // Extract frame
    Mat frame;
    capture.set(CV_CAP_PROP_POS_FRAMES, frameNumber);
    capture >> frame;
    
    // Save frame
    imwrite(outputFilename, frame);
    
    printf("Success.\n");

    return 0;
}
