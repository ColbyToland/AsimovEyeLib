// Code borrowed heavily from the OpenCV cailbration.cpp sample

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/videoio.hpp>

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
    
    int startFrame, totalFrames;
    string method;

    // Command line argument parsing
    cv::CommandLineParser parser(argc, argv,
        "{help h usage ?|               | print this message            }"
        "{o             | bgimg.jpg     | output file name              }"
        "{c             |               | camera intrinsics yml         }"
        "{m             | mog           | method (mog [default] or knn) }"
        "{s             | 0             | start frame                   }"
        "{n             | 10            | number of frames              }"
        "{@vid_name     |               | video file name               }"
        );
        
    if (parser.has("help") || !parser.has("@vid_name"))
    {
        parser.printMessage();
        return 0;
    }
    
    outputFilename = parser.get<string>("o");
    camFilename = parser.get<string>("c");
    method = parser.get<string>("m");
    startFrame = parser.get<int>("s");
    totalFrames = parser.get<int>("n");
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
    capture.open(inputFilename);
    if ( !capture.isOpened() )
    {
        fprintf(stderr, "Failed to open video\n");
        return -1;
    }

    // Create background subtractor
    Ptr<BackgroundSubtractor> bg_model = (method == "knn") ?
            createBackgroundSubtractorKNN().dynamicCast<BackgroundSubtractor>() :
            createBackgroundSubtractorMOG2().dynamicCast<BackgroundSubtractor>();

    // Read in video
    capture.set(CV_CAP_PROP_POS_FRAMES, startFrame);
    for (int frameNo = 0; frameNo < totalFrames; ++frameNo)
    {
        Mat frame;
        capture >> frame;
        if ( frame.empty() ) 
            break;
            
        // Preprocess image
        if (calibrated)
        {
            Mat temp;
            frame.copyTo(temp);
            undistort(temp, frame, cam_mat, dist_coeff, cam_mat);
        }
        
        //update the model
        Mat fgmask;
        bool update_bg_model = true;
        bg_model->apply(frame, fgmask, update_bg_model ? -1 : 0);
    }
    
    // Output best estimated background image
    Mat bgimg;
    bg_model->getBackgroundImage(bgimg);
    imwrite(outputFilename, bgimg);
    
    cout << "success." << endl;
    
    capture.release();

    return 0;
}
