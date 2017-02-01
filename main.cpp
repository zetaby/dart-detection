//
//  main.cpp
//  LineDetection
//
//  Created by Boyang Zhao and Zhaozhen xu.
//  All rights reserved.
//

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <math.h>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include <string>
#include <sstream>

using namespace cv;
using namespace std;

const double pi = 3.1415926f;
const double RADIAN = 180.0/pi;
//Mat contrast(Mat image);
struct line
{
    int theta;
    int r;
};

Mat contrast(Mat image){
    Mat new_image = Mat::zeros( image.size(), image.type() );
    
    int alpha=2;
    int beta=20;
    for( int y = 0; y < image.rows; y++ )
    {
        for( int x = 0; x < image.cols; x++ )
        {
            new_image.at<uchar>(y,x) = saturate_cast<uchar>( alpha*( image.at<uchar>(y,x)) + beta );
        }
    }
    
    return new_image;
}

vector<struct line> houghLine(Mat &img, int threshold)
{
    vector<struct line> lines;
    int diagonal = floor(sqrt(img.rows*img.rows + img.cols*img.cols));
    vector< vector<int> >p(360 ,vector<int>(diagonal));
    Mat houghImage(360, diagonal, CV_8UC1, Scalar(0));
    
    for( int j = 0; j < img.rows ; j++ ) {
        for( int i = 0; i < img.cols; i++ ) {
            if( img.at<unsigned char>(j,i) > 0)
            {
                for(int theta = 0;theta < 360;theta++)
                {
                    int r = floor(i*cos(theta/RADIAN) + j*sin(theta/RADIAN));
                    if(r < 0)
                        continue;
                    p[theta][r]++;
                }
            }
        }
    }

    int maxNum = 0;
    int minNum = 0;
    
    //get local maximum
    for( int theta = 0;theta < 360;theta++)
    {
        for( int r = 0;r < diagonal;r++)
        {
            int thetaLeft = max(0,theta-1);
            int thetaRight = min(359,theta+1);
            int rLeft = max(0,r-1);
            int rRight = min(diagonal-1,r+1);
            int tmp = p[theta][r];
            if( tmp > threshold && tmp > p[thetaLeft][rLeft] && tmp > p[thetaLeft][r] && tmp > p[thetaLeft][rRight]
               && tmp > p[theta][rLeft] && tmp > p[theta][rRight] && tmp > p[thetaRight][rLeft]
               && tmp > p[thetaRight][r] && tmp > p[thetaRight][rRight])
            {
                struct line newline;
                newline.theta = theta;
                newline.r = r;
                lines.push_back(newline);
            }
            
            if(tmp>maxNum){
                maxNum = tmp;
            }
            if(tmp<minNum){
                minNum = tmp;
            }
        }
    }

    maxNum = (float)maxNum;
    
    //draw the hough space using the p[y][x]
    for (int y = 0; y<houghImage.rows; y++) {
        for (int x = 0; x<houghImage.cols; x++) {
            houghImage.at<uchar>(y,x)=(int)((float)p[y][x]/maxNum*255);
        }
    }
    namedWindow("houghSpace",1);
    imshow("houghSpace",houghImage);
    return lines;
}

Mat drawLines(Mat &img, const vector<struct line> &lines)
{
    for(int i = 0;i < lines.size();i++)
    {
        vector<Point> points;
        int theta = lines[i].theta;
        int r = lines[i].r;
        
        double ct = cos(theta/RADIAN);
        double st = sin(theta/RADIAN);
        
        //r = x*ct + y*st
        //left
        int y = int(r/st);
        if(y >= 0 && y < img.rows){
            Point p(0, y);
            points.push_back(p);
        }
        //right
        y = int((r-ct*(img.cols-1))/st);
        if(y >= 0 && y < img.rows){
            Point p(img.cols-1, y);
            points.push_back(p);
        }
        //top
        int x = int(r/ct);
        if(x >= 0 && x < img.cols){
            Point p(x, 0);
            points.push_back(p);
        }
        //down
        x = int((r-st*(img.rows-1))/ct);
        if(x >= 0 && x < img.cols){
            Point p(x, img.rows-1);
            points.push_back(p);
        }
        
        cv::line( img, points[0], points[1], Scalar(40), 0.0001, CV_AA);
    }
    return img;
}

Mat threshold(Mat img){
    for( int j = 0; j < img.rows ; j++ ) {
        for( int i = 0; i < img.cols; i++ ) {
            if(img.at<uchar>(j, i)<250){
                img.at<uchar>(j,i)=0;
            }
            else{
                img.at<uchar>(j,i)=255;
            }
        }
    }
    return img;
}

int cenPointDetect(Mat &src,const vector<struct line> &lines){
    
    Mat linePointSpace = Mat(src.size(), CV_8UC1, Scalar(0));
    int cenPointLineSum=0;//the number of lines of the point which is most gone through by lines
    
    int h[linePointSpace.cols][linePointSpace.rows];
    for(int m = 0; m < linePointSpace.cols; m++){
        for(int n = 0; n < linePointSpace.rows; n++){
            h[m][n] =0;
        }
    }
    
    
    cout << "cols and rows:  "<<linePointSpace.cols << "  "<< linePointSpace.rows<<endl;
    
    for(int i = 0;i < lines.size();i++){
        int theta = lines[i].theta;
        int r = lines[i].r;
        
        double ct = cos(theta/RADIAN);
        double st = sin(theta/RADIAN);
        
        int y;
        
        for(int x = 0; x < linePointSpace.cols; x++){
            if(st!=0){
                y = ((r-ct*x)/st);
                if(y>=0&&y<linePointSpace.rows)
                {
//                    if(x>500)
//                        cout << x<<"  "<<y <<endl;
                  h[x][y]++;
//                  if(x-1>=0)
//                      h[x-1][y]++;
//                  if(x+1<img.cols)
//                      h[x+1][y]++;
//                  if(x-1>=0&&y-1>=0)
//                      h[x-1][y-1]++;
//                  if(x+1<img.cols&&y-1>=0)
//                      h[x+1][y-1]++;
//                  if(x-1>=0&&y+1<img.rows)
//                      h[x-1][y+1]++;
//                  if(x+1<img.cols&&y-1>=0)
//                      h[x+1][y-1]++;
//                  if(y-1>=0)
//                      h[x][y-1]++;
//                  if(y+1<img.rows)
//                      h[x][y+1]++;
                }
            }
        }
    }
    
    for(int m = 0; m < linePointSpace.cols; m++){
        for(int n = 0; n < linePointSpace.rows; n++){
            //the point with how many lines going through will be considered as a central point of interected lines
            if(h[m][n]>cenPointLineSum){
                //cv::line(linePointSpace, Point(m,n), Point(m,n), Scalar(255));
                cenPointLineSum = h[m][n];
            }
        }
    }
//    namedWindow("linePointSpace",1);
//    imshow("linePointSpace",linePointSpace);
    
    return cenPointLineSum;
}


int runLineDetecter(cv::Mat src){
    //Mat src;
    cv::Mat edge,lineSpace;
    int cenPointLineSum;
    //src = imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
    lineSpace = cv::Mat(src.size(), CV_8UC1, Scalar(0));
    edge = cv::Mat(src.size(), CV_8UC1, Scalar(0));
    
    int alpha=5;
    int beta=20;
    for( int y = 0; y < src.rows; y++ )
    {
        for( int x = 0; x < src.cols; x++ )
        {
            src.at<uchar>(y,x) = saturate_cast<uchar>( alpha*( src.at<uchar>(y,x)) + beta );
        }
    }
    

    //blur( src, src, Size(3,3) );
    //medianBlur(src, src, 3);
    GaussianBlur(src, src, Size(3,3),2);
    Canny( src, edge, 50, 200);
 
    vector<struct line> lines = houghLine(edge, 50);   ///////////////////////////////change the hough threshold
    drawLines(lineSpace, lines);
    cenPointLineSum = cenPointDetect(src,lines);
    
    namedWindow("src", 1);
    imshow("src", src);
    
    namedWindow("gradient", 1);
    imshow("gradient", edge);
    
    namedWindow("lineSpace", 1);
    imshow("lineSpace", lineSpace);
    
    //waitKey(0);
    
    return cenPointLineSum;
}

/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////
/** Function Headers */
void detectAndDisplay( Mat &frame );

/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;

/** @function detectAndDisplay */
void detectAndDisplay( Mat &frame )
{
    vector<Rect> faces;
    cv::Mat frame_gray;
    stringstream aa;
    string filename;
    int cenPointLineSum=0;
    
    // 1. Prepare Image by turning it into Grayscale and normalising lighting
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    // 2. Perform Viola-Jones Object Detection
    cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
    
    // 3. Print number of Faces found
    cout << "face size: " <<faces.size() << std::endl;
    
    int lineSum[faces.size()][2];
    
    // 4. Draw box around faces found
    for( int i = 0; i < faces.size(); i++ )
    {
//        Mat image= imreadimag);
//        Rect rect(10, 20, 100, 50);
//        Mat image_roi = image(rect);
        Mat frameclone = frame.clone();
        Mat resultBefore = frameclone(Range(faces[i].y,faces[i].y + faces[i].height+1),Range(faces[i].x,faces[i].x + faces[i].width+1));
        
        //save the image into file and automatically name the image file
//          aa << i;
//          filename = "xxx" + aa.str()+".jpg";
//          imwrite(filename, resultBefore);
        
        cenPointLineSum = runLineDetecter(resultBefore);
        lineSum[i][0] = i;
        lineSum[i][1] = cenPointLineSum;
        //show the image in the window
//          namedWindow("filename", CV_WINDOW_AUTOSIZE );
//          imshow( "filename", resultBefore);
//          waitKey(0);
    }
    int maxLineSum =0;
    for(int i =0; i < faces.size(); i++){
        if(lineSum[i][1]>=maxLineSum){
            maxLineSum = lineSum[i][1];
        }
    }
    
    for(int i =0; i < faces.size(); i++){
        if(lineSum[i][1] >= maxLineSum-4){///this is the most important threshold
            int j = lineSum[i][0];
            rectangle(frame, Point(faces[j].x, faces[j].y), Point(faces[j].x +faces[j].width, faces[j].y + faces[j].height), Scalar( 0, 255, 0 ), 2);
        }
    }    
}



int main( int, char** argv )
{
    //int a = runLineDetecter();
    //////////////////////////////////////
    //looping to creat 16 detected image
    //////////////////////////////////////
//   for(int t =0;t<16;t++){
//        string tt = to_string(t);
//        string infile = "dart";
//        infile = infile + tt + ".jpg";
//        
//        Mat frame = imread(infile,1);
//        // 2. Load the Strong Classifier in a structure called `Cascade'
//        if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
//        // 3. Detect Faces and Display Result
//        detectAndDisplay( frame );
//        // 4. Save Result Image
//        string outfile ="detected" + tt + ".jpg";
//        imwrite( outfile, frame );
//    }
    
    Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    detectAndDisplay( frame );
    imwrite( "detected0.jpg", frame );

    return 0;
}
