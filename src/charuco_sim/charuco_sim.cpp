// Code borrowed heavily from the OpenCV samples

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/aruco/charuco.hpp>

#include <GL/glut.h>

#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>

using namespace cv;
using namespace std;

/*
OpenGL functions for rendering and accepting user input.

This should include a save function that outputs the current
rendering with and without fisheye distortion as well as a file
containing distortion, intrinsic, and extrinsic parameters.
*/

void renderScene(void) 
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glBegin(GL_TRIANGLES);
		glVertex3f(-0.5,-0.5,0.0);
		glVertex3f(0.5,0.0,0.0);
		glVertex3f(0.0,0.5,0.0);
	glEnd();

    glutSwapBuffers();
}

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
    Ptr<aruco::CharucoBoard> board = aruco::CharucoBoard::create(5, 7, 0.04, 0.02, &dictionary);
    Mat boardImage; 
    board->draw( Size(600, 500), boardImage, 10, 1 );

    printf("Success.\n");

	// init GLUT and create Window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(320,320);
	glutCreateWindow("Lighthouse3D - GLUT Tutorial");

	// register callbacks
	glutDisplayFunc(renderScene);

	// enter GLUT event processing cycle
	glutMainLoop();

    return 0;
}
