
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;
using namespace boost::filesystem;

String body_cascade_name = "haarcascade_frontalface_alt.xml";

CascadeClassifier body_cascade;

int main(int argc, char** argv  )
{
    if( !body_cascade.load( body_cascade_name )) { 
	printf("--(!)Error loading body cascade\n"); return -1; 
    };
    path targetDir(argv[1]);
    directory_iterator end_itr;
    for(directory_iterator itr(targetDir); itr != end_itr; ++itr)
    {
       if(is_regular_file(itr->path()))
       {
		Mat image = imread(itr->path().string(), CV_LOAD_IMAGE_COLOR);
		 if(! image.data )                              // Check for invalid input
    		{
        		cout <<  "Could not open or find the image " << itr->path() << std::endl ;
        		return -1;
    		}
		std::vector<Rect> bodies;
  		Mat image_gray;

  		cvtColor( image, image_gray, CV_BGR2GRAY );
  		equalizeHist( image_gray, image_gray );

    		body_cascade.detectMultiScale( image_gray, bodies, 1.1, 2, 0, Size(80, 80) );

    		if(bodies.size() == 0){
        		cout << itr->path() << ",0" << std::endl; //printf("FALSE\n");
    		} else {
    			cout << itr->path() << ",1" << std::endl; //printf("TRUE\n");
                }
       }
    }
	
    return 0;
}
