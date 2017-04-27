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
    Ptr<BackgroundSubtractor> bg_model = 
        createBackgroundSubtractorMOG2().dynamicCast<BackgroundSubtractor>();
        //createBackgroundSubtractorKNN().dynamicCast<BackgroundSubtractor>()
            
    // Read in video
    //capture.set(CV_CAP_PROP_POS_FRAMES, startFrame);
    double totalpixels = height * width;
    double update_thresh = 0.1;
    int update_period = 10; // seconds
    int min_update_frame_cnt = fps*update_period;
    int frameNo = 0;
    //for (int frameNo = 0; frameNo < totalFrames; ++frameNo)
    while (true)
    {
        Mat frame;
        capture >> frame;
        if ( frame.empty() ) 
            break;
        
        if ( frameNo % (int)fps == 0 )
        {
            cout << "Frame " << frameNo + 1 << " of " << totalFrames;
            cout << " (" << (int)(frameNo*100 / totalFrames) << "%)" << endl;
        }
            
        // Preprocess image
        if (calibrated)
        {
            Mat temp;
            frame.copyTo(temp);
            undistort(temp, frame, cam_mat, dist_coeff, cam_mat);
        }
        
        // Detect background
        Mat fgmask;
        Mat bgmask;
        const double NO_UPDATE = 0;
        const double AUTO_RATE_UPDATE = -1;
        double update = NO_UPDATE;
        if ( frameNo < min_update_frame_cnt )
        {
            if ( frameNo % (int)fps == 0 ) 
                cout << "\tInitial background model capture" << endl;;
            update = AUTO_RATE_UPDATE;
        }
        bg_model->apply(frame, fgmask, NO_UPDATE);
        
        // Clean-up mask
        erode(fgmask, fgmask, Mat());//, Point(-1,-1), 3);
        //dilate(fgmask, fgmask, Mat(), Point(-1,-1), 1);
        GaussianBlur(fgmask, fgmask, Size(11, 11), 3.5, 3.5);
        threshold(fgmask, fgmask, 10, 255, THRESH_BINARY);
        bitwise_not(fgmask, bgmask);
        
        // Write output video frames
        Mat bgImg;
        Mat fgImg;
        frame.copyTo(bgImg,bgmask);
        frame.copyTo(fgImg,fgmask);
        bgwriter.write(bgImg);
        fgwriter.write(fgImg);
        
        // If the foreground is empty then update the background model
        if ( frameNo >= min_update_frame_cnt && frameNo % min_update_frame_cnt == 0 )
        {
            vector<Point> black_pixels;   // output, locations of non-zero pixels
            cv::findNonZero(fgmask, black_pixels);
            double pct_fgpx = black_pixels.size() / totalpixels;
            cout << "\tForeground " << (int)(pct_fgpx*100) << "%";
            if ( pct_fgpx < update_thresh )
            {
                bg_model->apply(frame, fgmask, AUTO_RATE_UPDATE);
                
                cout << ": Updated background";
            }
            cout << endl;
        }
        
        ++frameNo;
    }
    
    cout << "\nsuccess." << endl;
    
    capture.release();

    return 0;
}
