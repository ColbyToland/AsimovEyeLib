#include <opencv2/core.hpp>

#include <iostream>
#include <string>

bool parseCamFile(const std::string filename, cv::Mat& intrinsics, cv::Mat& dist_coef)
{
    cv::FileStorage camFile;
    camFile.open(filename, cv::FileStorage::READ);
    if ( !camFile.isOpened() )
    {
        std::cerr << "Cannot open " << filename << std::endl;
        return false;
    }
    
    cv::Mat temp;
    
    // Read camera matrix
    camFile["camera_matrix"] >> temp;
    if ( temp.empty() )
    {
        std::cerr << "Failed to read camera_matrix in " << filename << std::endl;
        return false;
    }
    temp.copyTo(intrinsics);
    
    // Read camera distortion coefficients
    camFile["distortion_coefficients"] >> temp;
    if ( temp.empty() )
    {
        std::cerr << "Failed to read distortion_coefficients in " << filename << std::endl;
        return false;
    }
    temp.copyTo(dist_coef);
    
    return true;    
}



bool parseImageList(const std::string& filename, 
                    std::vector<cv::Mat>& img_list, 
                    std::vector<std::string>& img_names,
                    cv::Mat intrinsics = cv::Mat(), 
                    cv::Mat distortionCoeff = cv::Mat() )
{
    img_list.resize(0);
    
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if( !fs.isOpened() )
    {
        std::cerr << "Failed to open image list: " << filename << std::endl;
        return false;
    }
    
    cv::FileNode n = fs.getFirstTopLevelNode();
    if( n.type() != cv::FileNode::SEQ )
    {
        std::cerr << "Image list file " << filename << " did not contain an image list" << std::endl;
        return false;
    }
    
    cv::FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it )
    {
        std::string fname = (std::string)*it;
        cv::Mat img = cv::imread(fname);
        if ( img.empty() )
            std::cerr << "Image file not found: " << fname << std::endl;
        else
        {
            img_names.push_back(fname);
            if (intrinsics.empty()) 
                img_list.push_back(img);
            else
            {
                cv::Mat undist;
                cv::undistort(img, undist, intrinsics, distortionCoeff, intrinsics);
                img_list.push_back(undist);
            }
        }
    }
            
    if (img_list.size() == 0)
    {
        std::cerr << "No images found" << std::endl;
        return false;
    }
            
    return true;
}
