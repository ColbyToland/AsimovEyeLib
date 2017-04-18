// Code borrowed heavily from the OpenCV cailbration.cpp sample

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

#include <stdio.h>
#include <string.h>

#include "../UtilityLib/common.hpp"

using namespace cv;
using namespace std;

static void help()
{
    printf( "This is a camera calibration sample.\n"
        "Usage: register\n"
        "     [-o=<out_params>]        # the output filename for the grabbed frame\n"
        "     [-c=<cam_intrinsics>]    # yml containing camera intrinsics parameters\n"
        "     [image_list]             # input image list\n"
        "\n" );
}

int main( int argc, char** argv )
{
    // Variables
    string outputFilename;
    string camFilename;
    string inputFilename;
    
    bool save_all = false;

    // Command line argument parsing
    cv::CommandLineParser parser(argc, argv,
        "{help h usage ?|               | print this message        }"
        "{all           |               |                           }"
        "{o             | undistort.jpg | output image name         }"
        "{c             |               | camera intrinsics yml     }"
        "{@image_list   |               | image list                }"
        );
        
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    
    save_all = parser.has("all");
    outputFilename = parser.get<string>("o");
    camFilename = parser.get<string>("c");
    inputFilename = parser.get<string>("@image_list");
    if (!parser.check())
    {
        help();
        parser.printErrors();
        return -1;
    }
    
    // Open images from image list
    vector<Mat> images;
    //if ( parseImageList(inputFilename, images) )
    {
        fprintf(stderr, "Bad image list file\n");
        return -1;
    }
    
    // Read intrinsic parameters
    Mat cam_mat;
    Mat dist_coeff;
    if ( !camFilename.empty() )
    {
        if ( !parseCamFile(camFilename, cam_mat, dist_coeff) )
        {
            fprintf(stderr, "Bad camera intrinsic file\n");
            return -1;
        }
    }

    // Find features in each image
    
    // Find corresponding features
    
    // Calculate transforms between all pairs of images
    
    if ( save_all )
    {
        // Write all transforms
    }
    
    // Find consensus among transforms
    
    // Write the consensus result
    
    printf("Success.\n");

    return 0;
}
