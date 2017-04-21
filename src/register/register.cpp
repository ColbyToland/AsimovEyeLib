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

static void help()
{
    printf( "This is a camera calibration sample.\n"
        "Usage: register\n"
        "     [--all]                  # save intermediary transforms\n"
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
        "{o             | positions.yml | output file name          }"
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
    vector<string> image_names;
    if ( !parseImageList(inputFilename, images, image_names, cam_mat, dist_coeff) )
    {
        fprintf(stderr, "Bad image list file\n");
        return -1;
    }
    int imageCount = images.size();

    // Find features in each image
    vector<vector<KeyPoint> > keypoints(imageCount);
    vector<Mat> descriptors(imageCount);
    Ptr<ORB> detector = ORB::create();
    for (int ind = 0; ind < imageCount; ++ind)
        detector->detectAndCompute( images[ind], noArray(), keypoints[ind], descriptors[ind] );
        
    // Prep image of image nodes and transform connections
    const double PI = 3.14159;
    Mat graphImg = Mat::zeros(500,700,CV_8UC3);//(500, 700, CV_8UC3, Scalar(255,255,255));
    Point center(250,250);
    double radius = 200;
    double dTheta = 2*PI / (double)imageCount;
    vector<Point> nodes(imageCount);
    //circle(graphImg, center, radius, Scalar(0,0,0), 3);
    
    for (int imgInd = 0; imgInd < imageCount; ++imgInd)
    {
        double theta = imgInd*dTheta;
        double x = center.x + radius*cos(theta);
        double y = center.y + radius*sin(theta);
        nodes[imgInd] = Point(x,y);
        
        putText(graphImg, 
                image_names[imgInd], 
                nodes[imgInd], 
                FONT_HERSHEY_SIMPLEX, 
                1, 
                Scalar(255,0,0) );
                
        Mat tempImg;
        drawKeypoints(images[imgInd], keypoints[imgInd], tempImg);
        imwrite("kp_" + image_names[imgInd], tempImg);
    }
                
    int total_connections = imageCount*(imageCount - 1);
    int failed_connections = total_connections;
    
    // Find corresponding features and pairwise [R|t]
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    map<pair<int,int>, pair<Mat,Mat> > positionMats;
    map<pair<int,int>, pair<int,int> > matScore;
    FileStorage fs(outputFilename, FileStorage::WRITE);
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
            vector<DMatch> good_matches;
            int correspondence = matches.size();
            for(unsigned i = 0; i < correspondence; i++)
            {
                if(matches[i][0].distance < 0.8 * matches[i][1].distance)
                {
                    good_matches.push_back(matches[i][0]);
                    pts1.push_back(keypoints[topInd][matches[i][0].queryIdx].pt);
                    pts2.push_back(keypoints[innerInd][matches[i][0].trainIdx].pt);
                }
            }
            correspondence = pts1.size();
                
            Mat mergeImg;
            size_t lastdot = image_names[topInd].find_last_of(".");
            string topName = image_names[topInd].substr(0,lastdot);
            string newfname = "matches_" + topName + "_" + image_names[innerInd];
            drawMatches(images[topInd], keypoints[topInd], 
                        images[innerInd], keypoints[innerInd], 
                        good_matches, 
                        mergeImg);            
            imwrite(newfname, mergeImg);
            
            //line(graphImg, nodes[topInd], nodes[innerInd], Scalar(0,0,255));
            
            // Too little correspondence check
            if (correspondence < 20) continue;
    
            // Calculate [R|t] transform between pair of images
            Mat E = findEssentialMat(pts1, pts2, cam_mat);
            Mat R, T;
            int inliers = recoverPose(E, pts1, pts2, cam_mat, R, T);
            
            // Too few inliers check
            if (inliers*2 < correspondence) continue;
            
            --failed_connections;
            
            // Store result
            pair<int,int> mapKey(topInd,innerInd);
            positionMats[mapKey] = pair<Mat,Mat>(R,T);  
            matScore[mapKey] = pair<int,int>(correspondence,inliers);
            
            line(graphImg, nodes[topInd], nodes[innerInd], Scalar(0,255,0));
    
            if ( save_all )
            {
                // Write pairwise transform        
                fs << "Transform";
                fs << "{";
                fs << "One" << topInd;
                fs << "Two" << innerInd;
                fs << "Correspondence" << correspondence;
                fs << "Inliers" << inliers;
                fs << "R" << R;
                fs << "T" << T;
                fs << "}";
            }
            
        }
    }
    
    // Find consensus among transforms
    
    // Write the consensus result
    
    fs.release();
    
    printf("Success.\n");
    
    stringstream ss;
    ss << failed_connections << " of " << total_connections;
    putText(graphImg, 
            ss.str(), 
            center, 
            FONT_HERSHEY_SIMPLEX, 
            1, 
            Scalar(0,0,0) );
    
    imshow("Camera Transform Graph", graphImg);
    waitKey();

    return 0;
}
