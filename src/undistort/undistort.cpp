// Code borrowed heavily from the OpenCV cailbration.cpp sample

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

#include <stdio.h>
#include <string>

#include "../UtilityLib/common.hpp"

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
    inputFilename = parser.get<string>("@image");
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
        image = imread(inputFilename);
        if ( image.empty() )
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
    Mat cam_mat;
    Mat dist_coeff;
    if ( !parseCamFile(camFilename, cam_mat, dist_coeff) )
    {
        fprintf(stderr, "Bad camera intrinsic file\n");
        return -1;
    }

    // Undistort image
    Mat undist;
    undistort(image, undist, cam_mat, dist_coeff, cam_mat);
    
    // Save undistorted image
    imwrite(outputFilename, undist);
    
    printf("Success.\n");

    return 0;
}
