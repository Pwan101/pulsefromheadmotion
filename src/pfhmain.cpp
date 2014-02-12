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
  
  // Output processed data
  ofstream output;
  output.open("data.txt");
 
  output << data << endl;

  cout << data.size() << endl;

  PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW);

  Mat mean=pca.mean.clone();
  Mat eigenvalues=pca.eigenvalues.clone();
  Mat eigenvectors=pca.eigenvectors.clone();

  Mat s=data*eigenvectors.t();

  cout << s << endl;
  
  // the S is the final pulse signal obatined from signal
  // I wrote this small library that should be able to extend easily.
  // The rest processing is not my familiar area.
  // There are some missing parts
  // First of all, from my computer, it hard to say what is the frequency of processing data
  // However, according to the paper, we have to upgrade it to roughly 250Hz
  // Secondly, a temporal filter is not aplied in this version.
  // Third, the final calculation of heart rate is missing.

  return 0;
}
