#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>

#include "package_bgs/bgslibrary.h"

#include "../UtilityLib/common.hpp"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    // Variables
    string camFilename;
    string inputFilename;

    // Command line argument parsing
    cv::CommandLineParser parser(argc, argv,
        "{help h usage ?|               | print this message            }"
        "{c             |               | camera intrinsics yml         }"
        "{@vid_name     |               | video file name               }"
        );
        
    if (parser.has("help") || !parser.has("@vid_name"))
    {
        parser.printMessage();
        return 0;
    }
    
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
    
    // Create output file name roots
    size_t extpos = inputFilename.find_last_of('.');
    string fnameRoot = inputFilename.substr(0,extpos) + ".avi";
    size_t dirpos = inputFilename.find_last_of('/');
    if (dirpos != string::npos)
        fnameRoot = fnameRoot.substr(dirpos+1);
    
    // Open output video files        
    VideoWriter fgwriter;
    VideoWriter bgwriter;
    int codec = CV_FOURCC('M', 'J', 'P', 'G');
    fgwriter.open("fg_" + fnameRoot, codec, fps, imgSz, true);
    bgwriter.open("bg_" + fnameRoot, codec, fps, imgSz, true);
    if ( !fgwriter.isOpened() || !bgwriter.isOpened() ) 
    {
        fprintf(stderr, "Could not open an output video file for write\n");
        return -1;
    }

    // Create background subtractor
    IBGS *bgs = new FrameDifference;
    
    // Read in video
    while (bgs)
    {
        Mat frame;
        capture >> frame;
        if ( frame.empty() ) 
            break;
            
        // Preprocess image
        if (calibrated)
            undistort(frame, frame, cam_mat, dist_coeff, cam_mat);

        cv::Mat fgmask;
        cv::Mat bgmodel;
        try
        {
            bgs->process(frame, fgmask, bgmodel);
        } 
        catch (Exception e)
        {
            cout << "Update Exception: " << e.what() << endl;
        }
        
        cvtColor(fgmask, fgmask, COLOR_GRAY2BGR);
        
        fgwriter.write(fgmask);
        bgwriter.write(bgmodel);
    }
    
    delete bgs;
    capture.release();
    
    cout << "success." << endl;

    return 0;
}
