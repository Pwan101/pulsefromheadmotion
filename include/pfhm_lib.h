
#include "sys_lib.h"

class PFHM
{
 private:
  string fn_haar;
  CascadeClassifier haar_cascade;

 public:
  // Constructor: to load haar features
  // path: absolute path to haar features
  PFHM(string path)
    {
      fn_haar=path;
      haar_cascade.load(fn_haar);
    }

  // Function: resize image to desire size
  // in: input image
  // out: resized image
  // mode: specify process mode
  //       PROCESSIMAGE=1;
  //       NOPROCESSIMAGE=2;
  void processImage(Mat in, Mat & out, int mode)
  {
    const int PROCESSIMAGE=1;
    const int NOPROCESSIMAGE=2;

    if (mode==PROCESSIMAGE)
      cv::resize(in, out, Size(in.cols/4*3, in.rows/4*3), 0, 0, INTER_LINEAR);
    else if (mode==NOPROCESSIMAGE)
      out=in;
    else 
      {
	cout << "no valid image processing mode is available" << endl;
	out=in;
      }
  }
  
  // Function: to find the maximum face in given image
  // in: input image
  // out: processed image
  // face: location and size of the face
  // faceFrame: face with interested region
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

  // Function: process tracked face to desire face
  // in: input image
  // face: location and size of the face
  // out: processed face
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

  // Function: draw tracked face
  // in: input frame
  // face_n: location and size of the face
  // box_text: title of the face
  void drawFace(Mat & in, cv::Rect & face_n, string box_text)
  {
    rectangle(in, face_n, CV_RGB(0,255,0), 1);
    int pos_x=std::max(face_n.tl().x-10, 0);
    int pos_y=std::max(face_n.tl().y-10, 0);
    putText(in, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0),2.0);
  }

  // Function: find the maximum face from all acquired faces
  // faces: all acquired faces' information
  // n: maximum face
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

  // Function: track face
  // in: input image
  // faces: all tracked faces' information
  void haar_faces(Mat in, vector<cv::Rect_<int> > & faces)
  {
    Mat gray;
    cv::cvtColor(in, gray, CV_BGR2GRAY);

    haar_cascade.detectMultiScale(gray, faces, 1.1, 2);
    gray.release();
  }

  // Function: perform LK Optical Flow algorithm
  // im1: previous image
  // im2: next image
  // cornerB: tracked feature points
  // win_size: window size
  // MAX_CORNERS: maximum tracked feature points
  void performLKFilter(Mat & im1, Mat & im2, vector<Point2f> & cornerB, int win_size, const int MAX_CORNERS)
  {
    im1.convertTo(im1, CV_8UC1);
    im2.convertTo(im2, CV_8UC1);

    Size img_sz=im1.size();

    int corner_count=MAX_CORNERS;
    Mat cornerA(2, MAX_CORNERS, CV_32FC1);
   
    goodFeaturesToTrack(im1, cornerA, corner_count, 0.05, 5.0);
    cornerSubPix(im1, cornerA, Size(win_size, win_size), Size(-1,-1), TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01));
    std::vector<uchar> feature_found; 
    feature_found.reserve(MAX_CORNERS);
    std::vector<float> feature_errors; 
    feature_errors.reserve(MAX_CORNERS);

    calcOpticalFlowPyrLK(im1, im2, cornerA, cornerB, feature_found, feature_errors, Size(win_size, win_size), 5, TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.001));
  }

  ////// My method //////

  // Function: interpolation 1D data using cubic interpolation
  // in: input matrix in 1*N
  // out: output interpolated matrx in 1*M (M=round(N*outFr/inFr))
  // inFr: original smapling frequency
  // outFr: desire sampling frequency
  void interpolationOneRow(Mat in, Mat & out, double inFr, double outFr)
  {
    double Fr=outFr/inFr;
    cv::resize(in, out, Size(in.cols*Fr, in.rows), INTER_CUBIC);
  }

  // Function: apply interpolation for all data
  // in: original data matrix (M*N)
  // out: interpolated output matrix (M*(N*outFr/inFr))
  // printmsg: to check if interpolation is working.
  void interpolation(Mat in, Mat & out, double inFr, double outFr, bool printmsg)
  {
    int inrows=in.rows;

    for (int i=0;i<inrows;i++)
      {
	Mat in_row=in.row(i);
	Mat out_row;
	interpolationOneRow(in_row, out_row, inFr, outFr);
	out.push_back(out_row);
      }

    // for checking only
    if (printmsg)
      {
	cout << "[INPUT] Rows: " << in.rows << " Cols: " << in.cols << endl;
	cout << "[OUTPUT] Rows: " << out.rows << " Cols: " << out.cols << endl;
      }
  }

  // FROM: http://mechatronics.ece.usu.edu/yqchen/filter.c/FILTER.C
  // Function: C implemenation of Matlab filter
  // ord: order (ord=order*2)
  // a: butter parameter a
  // b: butter parameter b
  // np: size of signal
  // x: input signal
  // y: filtered signal
  void filter(int ord, float *a, float *b, int np, float *x, float *y)
  {
    y[0]=b[0]*x[0];
    for (int i=1;i<ord+1;i++)
      {
        y[i]=0.0;
        for (j=0;j<i+1;j++)
	  y[i]=y[i]+b[j]*x[i-j];
        for (j=0;j<i;j++)
	  y[i]=y[i]-a[j+1]*y[i-j-1];
      }
    /* end of initial part */
    for (int i=ord+1;i<np+1;i++)
      {
	y[i]=0.0;
        for (int j=0;j<ord+1;j++)
	  y[i]=y[i]+b[j]*x[i-j];
        for (int j=0;j<ord;j++)
	  y[i]=y[i]-a[j+1]*y[i-j-1];
      }
  }

  // FROM: http://mechatronics.ece.usu.edu/yqchen/filter.c/FILTER.C
  // Function: C implemenation of Matlab filtfilt
  // ord: order (ord=order*2)
  // a: butter parameter a
  // b: butter parameter b
  // np: size of signal
  // x: input signal
  // y: filtered signal
  void filtfilt(int ord, float *a, float *b, int np, float *x, float *y)
  {
    filter(ORDER,a,b,NP,x,y);

    for (i=0;i<NP;i++)
      { 
	x[i]=y[NP-i-1];
      }
    /* do FILTER again */
    filter(ORDER,a,b,NP,x,y);
    /* reverse the series back */
    for (i=0;i<NP;i++)
      { 
	x[i]=y[NP-i-1];
      }
    for (i=0;i<NP;i++)
      { 
	y[i]=x[i];
      }
  }

  // Funtion: to return bandpassw butter worth filter parameters
  // in: input data matrix in 1*N
  // a: butter worth filter parameter
  // b: butter worth filter parameter
  // Fr: Sampling frequency
  // 4th order
  void butterWorthBandPass(Mat in, Mat & a, Mat & b, double Fr)
  {
    
  }

  void applyFilter(Mat in, Mat & out)
  {
    // apply butter worth filter
    
  }

  // in: input filtered matrix (M*N)
  // out: PCA projection (data in rows)
  void calcuatePCAProjection(Mat in, Mat & out)
  {
    Mat temp=in.t();

    PCA pca(temp, Mat(), CV_PCA_DATA_AS_ROW);
    Mat eigenvectors=pca.eigenvectors.clone();

    Mat o=temp*eigenvectors;
    out=o.t();
  }

  // Function: calcuate heart beat in minute
  // in: Input signal
  // heart_beat: output heart beat (Hz)
  // Fr: sampling rate
  void calculateHeartRate(Mat in, double & heart_beat, double Fr)
  {
    // to simplify, here we use second signal
    // should be a selection rule for this
    Mat signal=in.row(1);

    cv::Scalar signalMean=cv::mean(signal);
    double meanValue=signalMean[0];
    signal=signal-meanValue;

    // fft
    Mat X;
    cv::dft(signal, X);
    
    Mat P=abs(X);
    Mat PX;
    cv::pow(P,2,PX);

    double maxPower;
    int maxIdx;
    cv::minMaxIdx(PX, 0, &maxPower, 0, &maxIdx);

    double maxFr=8.0;
    heart_beat=(double)maxIdx*maxFr/(double)signal.rows;

  }

  ////// extract pulse signal

  // filtering

  // remove points moving too much

  // find projections

  // find pulse signal and increase sampling frequency

  // peak detection

  // calculate output parameters

  ////// compute final signal along with its power spectrum

  // compute fourier transform

  // extract pikes
};
