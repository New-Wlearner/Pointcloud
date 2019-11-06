#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <vector>


using namespace std;
using namespace cv;


void AverFiltering(const Mat &src,Mat &dst,int i,int j,int k);
void MidFiltering(const Mat &src,Mat &dst,int i,int j,int k);
void medianFilter(Mat& cap);



typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;


double fx = 1943.21, fy = 1943.21, cx = 500, cy = 350;
double baseline = 3.65;


int main(int argc, char** argv)
{
	cv::Mat shicha = cv::imread(argv[1],CV_16UC1); // 读入视差图
	cout<<shicha.size<<endl;
    cv::Rect m_select = Rect(0,0,1200,720);
    shicha = shicha(m_select); // 除去边缘空洞
	Mat depth(shicha.size(),CV_32FC1); //深度图定
	Mat rgb(shicha.size(),CV_8UC3);
	imshow("yuanshicha",shicha);
	rgb = imread(argv[2]);
	cv::Rect m_selectrgb = Rect(351,133,641,590);
    rgb = rgb(m_selectrgb); // 截取彩色图像

	// medianFilter(shicha);

    for(int row = 0; row < shicha.rows; row++) // 中值滤波
	{
		for (int col = 0; col < shicha.cols; col++)
        {
        if(shicha.at<uchar>(row,col) <= 150 )
            MidFiltering(shicha, shicha, row, col, 35);
        }
    }
	
	blur(shicha,shicha,Size(20,20),Point(-1,-1));

    // for(int row = 0; row < shicha.rows; row++) // 均值滤波
	// {
	// 	for (int col = 0; col < shicha.cols; col++)
    //     {
    //     if(shicha.at<uchar>(row,col) < 110)
    //         AverFiltering(shicha, shicha, row, col, 71);
    //     }
    // }

	PointCloud::Ptr cloud(new PointCloud); // 前三维为xyz,第四维为颜色
    for(int i=0;i<depth.rows;i++)
        for(int j=0;j<depth.cols;j++)
        {
            if(shicha.at<uchar>(i,j)==0) continue;
			depth.at<float>(i,j)=fy * baseline / shicha.at<uchar>(i,j);
			
			float d = depth.at<float>(i,j);
            // 根据双目模型计算 point 的位置
            double x = - (i - cx) / fx;
            double y = (j - cy) / fy;
			if(d == 0) continue;
			PointT p;
            p.z = d;
		    p.x = x*d;
		    p.y = y*d;
			p.b = rgb.ptr<uchar>(i)[j * 3];
		    p.g = rgb.ptr<uchar>(i)[j * 3 + 1];
		    p.r = rgb.ptr<uchar>(i)[j * 3 + 2];
 

            cloud->points.push_back(p);
        }
        // 设置并保存点云
	    cloud->height = 1;
	    cloud->width = cloud->points.size();
		// cloud->size = (cloud->height) * (cloud->width);
	    cout << "point cloud size = " << cloud->points.size() << endl;
	    cloud->is_dense = false;
	    try
        {
		    //保存点云图
		    pcl::io::savePCDFile("/home/stuw/coder/Coder/homework/pointcloud/build/a.pcd", *cloud);
	    }
	    catch (pcl::IOException &e)
        {
		    cout << e.what()<< endl;
	    }
	    cloud->points.clear();
	    cout << "Point cloud saved." << endl;
	Mat depth0;
	depth.convertTo(depth0,CV_8UC1);

    cv::imshow("shicha",shicha);
    cv::imshow("depth",depth0);
    
	waitKey(0);
        

    return 0;
    
}

void MidFiltering(const Mat &src,Mat &dst,int i,int j,int k) // 自定义中值滤波
{
	// if (!src.data) return;
    for(int c = k; c > 15; c = c - 2 )
	{
        uchar arr[c*c];
        int z=0;
        
			for(int a = i - (c-1)/2; a <= i + (c-1)/2; a++ )
            {
                for(int b = j - (c-1)/2; b <= j + (c-1)/2; b++)
                {
					if ((i - (c-1)/2 <= 0) || (j - (c-1)/2) <= 0 || (i + (c-1)/2)>src.rows || (j + (c-1)/2)>src.cols) continue;// 边缘检测 
                    arr[z] = src.at<uchar>(a,b);
                    z++;
                }
            } 
			for (int gap = c*c / 2; gap > 0; gap /= 2)//希尔排序
		    	for (int i = gap; i < c*c; ++i)
			    	for (int j = i - gap; j >= 0 && arr[j] > arr[j + gap]; j -= gap)
				    	swap(arr[j], arr[j + gap]);
	    	dst.at<uchar>(i,j) = arr[(c*c-1)/2];//返回中值 
        
	}
}

void medianFilter(Mat& cap)
{
  for(int w=20;w>2;w/=2)
  {
    int min_gray=150;
    for (int i=0;i<cap.rows;i++)
      for(int j=0;j<cap.cols;j++)
      {
		if(cap.at<uchar>(i,j)<min_gray+1)
		{
	  		vector<uchar> gray_list;
	  		int count=0;
	  		for(int k=i-w/2;k<=i+w/2;k++)
	    		for(int l=j-w/2;l<=j+w/2;l++)
	    		{
	      			if(k<0||k>cap.rows||l<0||l>cap.cols||cap.at<uchar>(k,l)==0)
					continue;
	     			count++;
	     			gray_list.push_back(cap.at<uchar>(k,l));
	    		}
	sort(gray_list.begin(),gray_list.end());
	cap.at<uchar>(i,j)=gray_list[count/2];
		}
    }
  }
}
//自定义均值滤波
void AverFiltering(const Mat &src,Mat &dst,int i,int j,int k) 
{
	if (!src.data) return;
    for(int c = k; c > 3; c = c - 2 )
	{
        int sum = 0;
		int minus;

			for(int a = i - (k-1)/2; a <= i + (k-1)/2; a++ )
            {
                for(int b = j - (k-1)/2; b <= j + (k-1)/2; b++)
                {
					if ((i - (c-1)/2 <= 0) || (j - (c-1)/2) <= 0 || (i + (c-1)/2)>src.rows || (j + (c-1)/2)>src.cols) 
					{
						minus++;
						continue;
					}

                    sum = sum + src.at<uchar>(a,b);
                }
            }  
        dst.at<uchar>(i,j) = sum / (c*c-minus);
		// else 
        // {
		// 	dst.at<uchar>(i, j) = 255; //边缘赋值
		// }
	}
}
