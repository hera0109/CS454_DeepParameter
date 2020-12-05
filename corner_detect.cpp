
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <string>
#include <stdio.h>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;
using namespace boost::filesystem;

int main(int argc, char** argv  )
{
    path targetDir(argv[1]);
    directory_iterator end_itr;
	int cnt=0;
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
		std::vector<std::tuple<int, int>> corners;
  		Mat image_gray;

  		cvtColor( image, image_gray, CV_BGR2GRAY );
  		//equalizeHist( image_gray, image_gray );
		
		/// Detector parameters
		int blockSize = 2;
		int apertureSize = 3;
		double k = 0.04;
		Mat dst, dst_norm, dst_norm_scaled;
  		dst = Mat::zeros( image.size(), CV_32FC1 );
    	int thresh = 200;
		int max_thresh = 255;

		/// Detecting corners
  		cornerHarris( image_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );

  		/// Normalizing
  		normalize( dst, dst_norm, 0, max_thresh, NORM_MINMAX, CV_32FC1, Mat() );
  		convertScaleAbs( dst_norm, dst_norm_scaled );
		
		for( int j = 0; j < dst_norm.rows ; j++ ) {
			for( int i = 0; i < dst_norm.cols; i++ ) {
				if( (int) dst_norm.at<float>(j,i) > thresh ) {
					corners.push_back(std::make_tuple(i, j));
				}
        	}
     	}
		cout << itr->path().string();
		cout << "," << corners.size();
		cout << std::endl;
		/*
		cout << itr->path().string();
		for (vector<std::tuple<int, int>>::iterator iter = corners.begin(); iter != corners.end() ; ++iter){
			cout << "," << std::get<0>(*iter) << "," << std::get<1>(*iter);
		}
		cout << std::endl;
		*/		
       }
    }
	
    return 0;
}
