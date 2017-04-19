// Code borrowed heavily from the OpenCV cailbration.cpp sample

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <map>
#include <stdio.h>
#include <string>
#include <vector>

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
    
    // Open images from image list
    vector<Mat> images;
    if ( !parseImageList(inputFilename, images, cam_mat, dist_coeff) )
    {
        fprintf(stderr, "Bad image list file\n");
        return -1;
    }

    // Find features in each image
    vector<vector<KeyPoint> > keypoints(images.size());
    vector<Mat> descriptors(images.size());
    Ptr<ORB> detector = ORB::create();
    for (int ind = 0; ind < images.size(); ++ind)
        detector->detectAndCompute( images[ind], noArray(), keypoints[ind], descriptors[ind] );
    
    // Find corresponding features and pairwise [R|t]
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");;
    map<pair<int,int>, pair<Mat,Mat> > positionMats;
    for (int topInd = 0; topInd < images.size() - 1; ++topInd)
    {
        for (int innerInd = 0; innerInd < images.size(); ++innerInd)
        {
            if (topInd == innerInd) continue;
            
            // Find matches
            vector<vector<DMatch> > matches;
            matcher->knnMatch( descriptors[topInd], descriptors[innerInd], matches, 2 );
            
            // Filter out matches that are less than 25% 
            //  closer than the next best match
            vector<Point2f> pts1;
            vector<Point2f> pts2;
            for(unsigned i = 0; i < matches.size(); i++)
            {
                if(matches[i][0].distance < 0.8 * matches[i][1].distance)
                {
                    pts1.push_back(keypoints[topInd][matches[i][0].queryIdx].pt);
                    pts2.push_back(keypoints[innerInd][matches[i][0].trainIdx].pt);
                }
            }
    
            // Calculate [R|t] transform between pair of images
            Mat E = findEssentialMat(pts1, pts2, cam_mat);
            Mat R, T;
            int inliers = recoverPose(E, pts1, pts2, cam_mat, R, T);
            pair<int,int> mapKey(topInd,innerInd);
            positionMats[mapKey] = pair<Mat,Mat>(R,T);  
            
            cout << "[" << topInd << "," << innerInd << "]:" << endl;
            cout << R << endl;
            cout << T << endl;
            cout << endl;          
    
            if ( save_all )
            {
                // Write pairwise transform
            }
            
        }
    }
    
    // Find consensus among transforms
    
    // Write the consensus result
    
    printf("Success.\n");

    return 0;
}
