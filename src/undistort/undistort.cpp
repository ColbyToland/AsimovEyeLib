// Code borrowed heavily from the OpenCV cailbration.cpp sample

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>

#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "CamIntrinsicReader.hpp"

using namespace cv;
using namespace std;

static void help()
{
    printf( "This is a camera calibration sample.\n"
        "Usage: undistort\n"
        "     [-o=<out_image>]         # the output filename for the grabbed frame\n"
        "     [cam_intrinsics]         # yml containing camera intrinsics parameters\n"
        "     [image]                  # input image\n"
        "\n" );
}

int main( int argc, char** argv )
{
    // Variables
    string outputFilename;
    string camFilename;
    string inputFilename;
    
    int frameNumber;

    VideoCapture capture;

    // Command line argument parsing
    cv::CommandLineParser parser(argc, argv,
        "{help h usage ?|               | print this message        }"
        "{o             | undistort.jpg | output image name         }"
        "{@instrinsics  |               | camera intrinsics yml     }"
        "{@image        |               | image                     }"
        );
        
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    
    outputFilename = parser.get<string>("o");
    camFilename = parser.get<string>("@instrinsics");
    inputFilename = parser.get<int>("@image");
    if (!parser.check())
    {
        help();
        parser.printErrors();
        return -1;
    }
    
    // Open image
    Mat image;
    if( !inputFilename.empty() )
    {
        if ( !imread(inputFilename, image) )
        {
            fprintf(stderr, "Cannot open image\n");
            return -1;
        }
    }
    else
    {
        fprintf(stderr, "No input file\n");
        return -1;
    }
    
    // Read intrinsic parameters
    

    // Undistort image
    Mat undist;
    
    // Save undistorted image
    imwrite(outputFilename, undist);
    
    printf("Success.\n");

    return 0;
}
