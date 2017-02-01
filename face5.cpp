//
//  face5.cpp
//  coursework
//
//  Created by Carina XU on 28/11/2016.
//  Copyright © 2016 Carina XU. All rights reserved.
//

/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
//String cascade_name = "cascade.xml";
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{
    // 1. Read Input Image
    Mat frame = imread("dart5.jpg", CV_LOAD_IMAGE_COLOR);
    
    // 2. Load the Strong Classifier in a structure called `Cascade'
    if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    
    // 3. Detect Faces and Display Result
    detectAndDisplay( frame );
    
    // face 1 - 11
    rectangle(frame, Point(60,130), Point(120,210), Scalar( 0, 0, 255 ), 2);
    rectangle(frame, Point(50,240), Point(110,320), Scalar( 0, 0, 255 ), 2);
    rectangle(frame, Point(190,210), Point(250,290), Scalar( 0, 0, 255 ), 2);
    rectangle(frame, Point(250,150), Point(310,230), Scalar( 0, 0, 255 ), 2);
    rectangle(frame, Point(290,230), Point(350,310), Scalar( 0, 0, 255 ), 2);
    rectangle(frame, Point(380,180), Point(440,260), Scalar( 0, 0, 255 ), 2);
    rectangle(frame, Point(430,220), Point(490,300), Scalar( 0, 0, 255 ), 2);
    rectangle(frame, Point(510,170), Point(570,250), Scalar( 0, 0, 255 ), 2);
    rectangle(frame, Point(560,240), Point(620,320), Scalar( 0, 0, 255 ), 2);
    rectangle(frame, Point(640,180), Point(700,260), Scalar( 0, 0, 255 ), 2);
    rectangle(frame, Point(680,240), Point(740,320), Scalar( 0, 0, 255 ), 2);
    
    // 4. Save Result Image
    imwrite( "detected5.jpg", frame );
    
    return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
    std::vector<Rect> faces;
    Mat frame_gray;
    //float groundtruth = 60*60;
    float POS = 11;
    float TP = 0, FN = 0, FP = 0;
    float TPR, F1 = 0.0;
    int groundtruth[11][2]{60,130,50,240,190,210,250,150,290,230,380,180,430,220,510,170,560,240,640,180,680,240};
    // 1. Prepare Image by turning it into Grayscale and normalising lighting
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    // 2. Perform Viola-Jones Object Detection
    cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
    
    // 3. Print number of Faces found
    std::cout << faces.size() << std::endl;
    
    
    // 4. Draw box around faces found
    for( int i = 0; i < faces.size(); i++ )
    {
        float intersection = 0;
        rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
        for(int j=0; j<11; j++){
            // calculate the overlap of the ground truth and detected face rectangle
            if (groundtruth[j][0] > faces[i].x+faces[i].width);
            if (groundtruth[j][1] > faces[i].y + faces[i].height);
            if (groundtruth[j][0] + 60 < faces[i].x);
            if (groundtruth[j][1] + 80 < faces[i].y);
            else {
                float colInt =  min(groundtruth[j][0] + 60,faces[i].x+faces[i].width) - max(groundtruth[j][0], faces[i].x);
                float rowInt =  min(groundtruth[j][1] + 80,faces[i].y + faces[i].height) - max(groundtruth[j][1],faces[i].y);
                if(colInt*rowInt>=intersection){
                    intersection = colInt * rowInt;
                }
            }
            
        }
        if(intersection >= 400) TP++;
        else FP++;
        
    }
    cout << "FP"<<FP << endl;
    cout << "TP"<<TP << endl;
    FN = POS - TP;
    TPR = TP / POS;
    F1 = (2*((TP/faces.size())*TPR))/((TP/faces.size())+TPR);
    cout << "TPR = " << TPR << endl;
    cout << "F1 score = " << F1 << endl;
    
}
