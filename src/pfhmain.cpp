// Author: Hu Yuhuang.
// Date: 20140119
#include "sys_lib.h"
#include "pfhm_lib.h"

int main(void)
{
  PFHM * pfhm=new PFHM("../resources/haarcascade_frontalface_default.xml");

  cv::VideoCapture capture;
  capture.open("../resources/data/face8.mp4");

  string WINDOW_NAME="Pulse from Head Motion";
  int window_x=300;
  int window_y=300;

  cv::namedWindow(WINDOW_NAME.c_str(), cv::WINDOW_AUTOSIZE);
  moveWindow(WINDOW_NAME.c_str(), window_x, window_y);

  int counter=0;
  Mat savedFrame;
  Size savedSize;
  vector<vector<Point2f> > corners;
  while (true)
    {
      Mat frame, buffer, faceFrame;
      if (!capture.isOpened())
	{
	  cout << "error" << endl;
	  break;
	}
      capture >> buffer;

      if (buffer.data)
	{
	  pfhm->processImage(buffer, frame, 2); // no process image's size
	  Rect face(0,0,10,10);
	  pfhm->findFace(frame, frame, face, faceFrame);

	  if (counter==0)
	    {
	      savedFrame=faceFrame;
	      savedSize=savedFrame.size();
	    }
	  else
	    {
	      vector<Point2f> cornerB;
	      cv::resize(faceFrame, faceFrame, savedSize, 0, 0, INTER_LINEAR);
	      pfhm->performLKFilter(savedFrame, faceFrame, cornerB, 15, 200);
	      corners.push_back(cornerB);
	    }

	  imshow(WINDOW_NAME.c_str(), faceFrame);
	  counter++;
	  cout << counter << endl;
	}
      else break;
     
      if (cv::waitKey(5)==27)
	{
	  capture.release();
	  cv::destroyWindow(WINDOW_NAME.c_str());
	}
    }
  
  // release the memory attached to window
  capture.release();
  cv::destroyWindow(WINDOW_NAME.c_str());

  Mat data;
  for (int k=0;k<corners.size();k++)
    {    
      Mat curr;
      for (int i=0;i<corners[k].size();i++)
	{
	  curr.push_back(corners[k].at(i).y);
	}
      cout << "Processed :" << k << endl;
      curr=curr.t();
      data.push_back(curr);
    }
  
  Mat origin=data.t();
  Mat processed;

  pfhm->interpolation(origin, processed, 30, 250, true);

  Mat S;
  pfhm->calcuatePCAProjection(processed.t(), S);

  double heart_beat;
  pfhm->calculateHeartRate(S, heart_beat, 250);

  cout << heart_beat << endl;

  return 0;
}
