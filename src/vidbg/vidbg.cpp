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
    
    int startFrame, totalFrames;

    // Command line argument parsing
    cv::CommandLineParser parser(argc, argv,
        "{help h usage ?|               | print this message        }"
        "{o             | bgimg.jpg     | output file name          }"
        "{c             |               | camera intrinsics yml     }"
        "{s             | 0             | start frame               }"
        "{n             | 10            | number of frames          }"
        "{@vid_name     |               | video file name           }"
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
    size_t width, height;
    capture.open(inputFilename);
    if ( !capture.isOpened() )
    {
        fprintf(stderr, "Failed to open video\n");
        return -1;
    }
    width = (size_t)capture.get(CV_CAP_PROP_FRAME_WIDTH);
    height = (size_t)capture.get(CV_CAP_PROP_FRAME_HEIGHT);

    // Read in video
    map<pair<int,int>, Mat> histograms;
    const int colors = 3;
    int bins = 32;
    Mat newHist = Mat::zeros(colors, bins, CV_32SC1);
    capture.set(CV_CAP_PROP_POS_FRAMES, startFrame);
    for (int frameNo = 0; frameNo < totalFrames; ++frameNo)
    {
        Mat frame;
        capture >> frame;
        if ( frame.empty() ) 
            break;
            
        if (calibrated)
        {
            Mat temp;
            frame.copyTo(temp);
            undistort(temp, frame, cam_mat, dist_coeff, cam_mat);
        }
            
        cout << "\tFrame: " << startFrame + frameNo << endl;
            
        // Iterate over each pixel and update the appropriate histogram
        MatIterator_<Vec3b> it, end;
        it = frame.begin<Vec3b>();
        end = frame.end<Vec3b>();
        for ( ; it != end; ++it )
        {
            Point rcpos = it.pos();
            pair<int,int> key(rcpos.y,rcpos.x);
            if ( histograms.count(key) == 0 )
            {
                histograms[key] = Mat();
                newHist.copyTo(histograms[key]);
            }
            Mat& curHist = histograms[key];
            for (int colorInd = 0; colorInd < colors; ++colorInd)
            {
                int bin = (*it)[colorInd] / 8;
                int curVal = curHist.at<int>(colorInd,bin);
                curHist.at<int>(colorInd,bin) = curVal + 1;
            }
        }
    }
    
    // Find the most common value for each pixel
    Mat bgImg(height, width, CV_8UC3);
    Mat rngImg(height, width, CV_8UC3);
    for ( size_t row = 0; row < height; ++row )
    {
        for ( size_t col = 0; col < width; ++col )
        {        
            // Find the most frequently occuring color
            Vec3b frequentColor(0,0,0);
            Vec3b secondColor(0,0,0);
            Mat& curHist = histograms[pair<int,int>(row,col)];
            for (int colorInd = 0; colorInd < colors; ++colorInd)
            {
                int maxCount = 0;
                unsigned short maxBin[] = {0, 0};
                for ( int binInd = 0; binInd < bins; ++binInd)
                {
                    int binVal = curHist.at<int>(colorInd,binInd);
                    if (binVal > maxCount)
                    {
                        maxCount = binVal;
                        maxBin[1] = maxBin[0];
                        maxBin[0] = binInd*8;
                    }
                }
                
                frequentColor[colorInd] = maxBin[0];
                secondColor[colorInd] = maxBin[1];
            }
            
            // Store most frequently occuring color to the background
            bgImg.at<Vec3b>(row,col) = frequentColor;
            rngImg.at<Vec3b>(row,col) = secondColor;
        }
    }
    
    // Output best estimated background image
    imwrite(outputFilename, bgImg);
    imwrite("range_" + outputFilename, rngImg);
    
    cout << "success." << endl;
    
    capture.release();

    return 0;
}
