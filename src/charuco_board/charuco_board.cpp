// Code borrowed heavily from the OpenCV samples

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/aruco/charuco.hpp>

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
    
    // Create and save board image
    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_250); 
    aruco::CharucoBoard board = aruco::CharucoBoard::create(5, 7, 0.04, 0.02, dictionary);
    Mat boardImage; 
    board.draw( Size(600, 500), boardImage, 10, 1 );

    imwrite("charuco_board.png",boardImage);
    
    printf("Success.\n");

    return 0;
}
