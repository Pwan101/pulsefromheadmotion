
#include "sys_lib.h"

class PFHM
{
 private:
  string fn_haar;
  CascadeClassifier haar_cascade;

 public:
  PFHM(string path)
    {
      //fn_path=path;
      fn_haar=path;
      haar_cascade.load(fn_haar);
    }
  void processImage(Mat in, Mat & out)
  {
    cv::resize(in, out, Size(in.cols/4*3, in.rows/4*3), 0, 0, INTER_LINEAR);
  }
  
  void findFace(Mat in, Mat & out, Rect & face, Mat & faceFrame)
  {
    vector<Rect_<int> > faces;
    haar_faces(in, faces);

    if (faces.size()>0)
      {
	size_t n;
	findMaxFace(faces, n);

	Rect face=faces[n];

	findProcessRegion(in, face, faceFrame);

	drawFace(in, face, "face");
	out = in;
      }
    else
      {
	faceFrame=in(Rect(face.x, face.y, face.width, face.height));
	out = in;
      }
  }

  void findProcessRegion(Mat in, Rect face, Mat & out)
  {
    // Main Face Region
    face.x=face.x+face.width/4;
    face.width=face.width/2;
    face.height=face.height/10*9;

    Mat faceFrame=in(face);

    // Remove eye region
    
    Rect up(0,0,face.width,face.height);
    Rect down=up;

    up.height=up.height*10/45;
    out=faceFrame(up);

    down.y=down.y+down.height/20*11;
    down.height=down.height/20*9;
    Mat downFrame=faceFrame(down);

    out.push_back(downFrame);
    cv::cvtColor(out, out, CV_BGR2GRAY);
  }

  void drawFace(Mat & frame, cv::Rect & face_n, string box_text)
  {
    rectangle(frame, face_n, CV_RGB(0,255,0), 1);
    int pos_x=std::max(face_n.tl().x-10, 0);
    int pos_y=std::max(face_n.tl().y-10, 0);
    putText(frame, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0),2.0);
  }

  void findMaxFace(vector<Rect_<int> > faces, size_t & n)
  {
    n=0;
    int max=-1;
    for (size_t i=0; i<faces.size(); i++)
      {
	if (max<faces[i].height*faces[i].width)
	  {
	    max=faces[i].height*faces[i].width;
	    n=i;
	  }
      }
  }

  void haar_faces(Mat in, vector<cv::Rect_<int> > & faces)
  {
    Mat gray;
    cv::cvtColor(in, gray, CV_BGR2GRAY);

    haar_cascade.detectMultiScale(gray, faces);
    gray.release();
  }

  void performLKFilter(Mat & im1, Mat & im2, vector<Point2f> & cornerB, int win_size, const int MAX_CORNERS)
  {
    im1.convertTo(im1, CV_8UC1);
    im2.convertTo(im2, CV_8UC1);

    Size img_sz=im1.size();

    int corner_count=MAX_CORNERS;
    Mat cornerA(2, MAX_CORNERS, CV_32FC1);
   
    goodFeaturesToTrack(im1, cornerA, corner_count, 0.05, 5.0);
    cornerSubPix(im1, cornerA, Size(win_size, win_size), Size(-1,-1), TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 20, 0.03));
    std::vector<uchar> feature_found; 
    feature_found.reserve(MAX_CORNERS);
    std::vector<float> feature_errors; 
    feature_errors.reserve(MAX_CORNERS);

    calcOpticalFlowPyrLK(im1, im2, cornerA, cornerB, feature_found, feature_errors, Size(win_size, win_size), 5, TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 20, 0.01));
  }
};
