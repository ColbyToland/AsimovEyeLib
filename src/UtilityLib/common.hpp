#include <opencv2/core.hpp>

#include <iostream>
#include <string>

using namespace std;
using namespace cv;

bool parseCamFile(string filename, Mat& intrinsics, Mat& dist_coef)
{
    FileStorage camFile;
    camFile.open(filename, FileStorage::READ);
    if ( !camFile.isOpened() )
    {
        cerr << "Cannot open " << filename << endl;
        return false;
    }
    
    Mat temp;
    
    // Read camera matrix
    camFile["camera_matrix"] >> temp;
    if ( temp.empty() )
    {
        cerr << "Failed to read camera_matrix in " << filename << endl;
        return false;
    }
    temp.copyTo(intrinsics);
    
    // Read camera distortion coefficients
    camFile["distortion_coefficients"] >> temp;
    if ( temp.empty() )
    {
        cerr << "Failed to read distortion_coefficients in " << filename << endl;
        return false;
    }
    temp.copyTo(dist_coef);
    
    return true;    
}

