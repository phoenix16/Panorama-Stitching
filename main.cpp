//
// Developer : Prakriti Chintalapoodi - c.prakriti@gmail.com 
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/stitching/stitcher.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

void stitchLeftRight(Mat& leftImage, Mat& rightImage, Mat&rightImageWarped, Mat& panorama);

int main(int argc, char* argv[])
{
    if (argc != 5)
	{
		cout << "Usage: ./Panorama <image1> <image2> <image3> <image4>" << endl;
		return -1;
	}

	Mat image1 = imread(argv[1]);
	Mat image2 = imread(argv[2]);
	Mat image3 = imread(argv[3]);
        Mat image4 = imread(argv[4]);

	if( !image1.data || !image2.data || !image3.data || !image4.data)
	{ cout << "Error reading images " << endl; return -1; }
				
    Mat pano1, pano2, rightImageWarped1, rightImageWarped2, panoramaOut;

    stitchLeftRight(image1, image2, rightImageWarped1, pano1);
    imshow("rightImageWarped 1", rightImageWarped1);
    imshow("Stitched Panorama 1", pano1);

    stitchLeftRight(pano1, image3, rightImageWarped2, pano2);
    imshow("rightImageWarped 2", rightImageWarped2);
    imshow("Stitched Panorama 2", pano2);

	//stitchLeftRight(image2, pano1, pano2);
	//imshow("Stitched Panorama 2", pano2);

	//stitchLeftRight(image1, pano2, panoramaOut);
	//imshow("Stitched Panorama 3", panoramaOut);

    imshow("image 1", image1);
    imshow("image 2", image2);
    imshow("image 3", image3);

    waitKey(0);
    return 0;
}

void stitchLeftRight(Mat& leftImage, Mat& rightImage, Mat& rightImageWarped, Mat& panorama)
{
	// Detect the keypoints using SURF Detector
	int minHessian = 400;	
	SurfFeatureDetector detector(minHessian);	
	std::vector<KeyPoint> keypoints_leftImage, keypoints_rightImage;	
	detector.detect( leftImage, keypoints_leftImage );
	detector.detect( rightImage, keypoints_rightImage );
	
	// Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;	
	Mat descriptors_leftImage, descriptors_rightImage;	
	extractor.compute( leftImage, keypoints_leftImage, descriptors_leftImage );
	extractor.compute( rightImage, keypoints_rightImage, descriptors_rightImage );
	
	// Match descriptor vectors using FLANN matcher
	// FLANN matching serves as initialization to the RANSAC feature matching (future step)
	// FLANN finds the nearest neighbors of keypoints in left image present in the right image
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match( descriptors_leftImage, descriptors_rightImage, matches );
	
	double max_dist = 0, min_dist = 100;
	
	// Find max and min distances between keypoints
	for (int i = 0; i < descriptors_leftImage.rows; i++ )
	{ 
	  double dist = matches[i].distance;
	  if( dist < min_dist ) min_dist = dist;
	  if( dist > max_dist ) max_dist = dist;
	}
	
	// Use only "good" matches (i.e. whose distance is less than 3*min_dist ) to 
	// construct Homography (Projective Transformation)
	std::vector< DMatch > good_matches;	
	for (int i = 0; i < descriptors_leftImage.rows; i++)
	{ 
	    if (matches[i].distance < 3*min_dist)
	    {
		good_matches.push_back(matches[i]); 
	    }
	}

	// Isolate the matched keypoints in each image
	std::vector<Point2f> leftImage_matchedKPs;
	std::vector<Point2f> rightImage_matchedKPs;
	
    for (size_t i = 0; i < good_matches.size(); i++ )
	{
	    leftImage_matchedKPs.push_back( keypoints_leftImage[ good_matches[i].queryIdx ].pt );
	    rightImage_matchedKPs.push_back( keypoints_rightImage[ good_matches[i].trainIdx ].pt );
	}

	// Find the Homography relating rightImage and leftImage
	Mat H = findHomography( Mat(rightImage_matchedKPs), Mat(leftImage_matchedKPs), CV_RANSAC );
	// Warp rightImage to leftImage's space using the Homography just constructed
//    Mat rightImageWarped;  // warped image has twice the width to account for overlap
    warpPerspective(rightImage, rightImageWarped, H, Size(rightImage.cols*2, rightImage.rows), INTER_CUBIC);

    panorama = rightImageWarped.clone();
    // Overwrite leftImage on left end of final panorma image
    Mat roi(panorama, Rect(0, 0, leftImage.cols, leftImage.rows));
    leftImage.copyTo(roi);
}
