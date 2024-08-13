#pragma once
#include<iostream>
#include <numeric>
#include<opencv2/opencv.hpp>
#include<io.h>
#include <vector>
#include <utility>
#include <string>   

#define ORT_OLD_VISON 13 

// 存储人体姿态关键点的坐标和置信度
struct PoseKeyPoint {
	float x = 0;	//关键点x坐标
	float y = 0;	//关键点y坐标
	float confidence = 0;	//关键点的置信度



};

//存储检验结果
struct OutputParams {
	int id;			//结果类别id
	float confidence;	//结果置信度
	cv::Rect box;		//矩形框
	cv::RotatedRect rotatedBox;		//obb结果矩形框
	cv::Mat boxMask;		//矩形框内mask，节省内存空间和加快速度
	std::vector<PoseKeyPoint> keyPoints; //姿态关键点，存储人体姿态识别中的关键点

};

//存储掩码相关的参数
struct MaskParams {
	//int segChannels = 32;
	//int segWidth = 160;
	//int segHeight = 160;
	int netWidth = 640;	//网络输入宽度
	int netHeight = 640;	//网络输入高度
	float maskThreshold = 0.5;	//掩码阈值
	cv::Size srcImgShape;	//源图像的大小
	cv::Vec4d params;	//参数向量，用于存储相关的缩放比例和偏移量
};

//存储人体姿态估计相关参数
struct PoseParams {
	float kptThreshold = 0.5;	//关键点的置信度阈值
	int kptRadius = 5;		//关键点绘制时的半径
	bool isDrawKptLine = true; //是否绘制关键点之间的连线，默认是true
	cv::Scalar personColor = cv::Scalar(0, 0, 255);	//人物的颜色，默认是红色
	std::vector<std::vector<int>>skeleton = {	//骨架连线的定义，关键点的连接关系
		{16, 14} ,{14, 12},{17, 15},{15, 13},
		{12, 13},{6, 12},{7, 13},{6, 7},{6, 8},{7, 9},
		{8, 10},{9, 11},{2, 3},{1, 2},{1, 3},{2, 4},
		{3, 5},{4, 6},{5, 7}
	};
	std::vector<cv::Scalar> posePalette =//关键点颜色调色板
	{
	cv::Scalar(255, 128, 0) ,
	cv::Scalar(255, 153, 51),
	cv::Scalar(255, 178, 102),
	cv::Scalar(230, 230, 0),
	cv::Scalar(255, 153, 255),
	cv::Scalar(153, 204, 255),
	cv::Scalar(255, 102, 255),
	cv::Scalar(255, 51, 255),
	cv::Scalar(102, 178, 255),
	cv::Scalar(51, 153, 255),
	cv::Scalar(255, 153, 153),
	cv::Scalar(255, 102, 102),
	cv::Scalar(255, 51, 51),
	cv::Scalar(153, 255, 153),
	cv::Scalar(102, 255, 102),
	cv::Scalar(51, 255, 51),
	cv::Scalar(0, 255, 0),
	cv::Scalar(0, 0, 255),
	cv::Scalar(255, 0, 0),
	cv::Scalar(255, 255, 255),
	};
	//骨架连线的颜色索引
	std::vector<int> limbColor = { 9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16 };
	//关键点的颜色索引
	std::vector<int> kptColor = { 16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9 };
	//关键的名称
	std::map<unsigned int, std::string> kptBodyNames{
		{0,"Nose"},				{1,	"left_eye"},
		{2,	"right_eye"},		{3,"left_ear"},		
		{4,	"right_ear"},		{5,	"left_shoulder"},	
		{6,	"right_shoulder"},	{7,	"left_elbow"},		
		{8,	"right_elbow"},		{9,	"left_wrist"},		
		{10,"right_wrist"},		{11,"left_hip"},		
		{12,"right_hip"},		{13,"left_knee"},		
		{14,"right_knee"},		{15,"left_ankle"},		
		{16,"right_ankle"}
	};
};

//检查模型路径是否合法
bool CheckModelPath(std::string modelPath);

//检查参数是否合法
bool CheckParams(int netHeight, int netWidth, const int* netStride, int strideSize);

//在图像上绘制检测结果，包括矩形框和类别名称
void DrawPred(cv::Mat& img,
	std::vector<OutputParams> result,
	std::vector<std::string> classNames,
	std::vector<cv::Scalar> color,
	bool isVideo = false
);

//在图像上绘制姿态检测结果，包括关键点和骨架
void DrawPredPose(cv::Mat& img, std::vector<OutputParams> result, PoseParams& poseParams, bool isVideo = false);

//在图像上绘制旋转矩形框
void DrawRotatedBox(cv::Mat& srcImg, cv::RotatedRect box, cv::Scalar color, int thinkness);

//对图像进行LetterBox处理，使其适配网络输入尺寸，同时保持宽高比
void LetterBox(const cv::Mat& image, cv::Mat& outImage,
	cv::Vec4d& params, //[ratio_x,ratio_y,dw,dh]
	const cv::Size& newShape = cv::Size(640, 640),
	bool autoShape = false,
	bool scaleFill = false,
	bool scaleUp = true,
	int stride = 32,
	const cv::Scalar& color = cv::Scalar(114, 114, 114));

//从掩码提案和原始掩码中获取最终掩码
void GetMask(const cv::Mat& maskProposals, const cv::Mat& maskProtos, std::vector<OutputParams>& output, const MaskParams& maskParams);

//从掩码提案和原始掩码中获取单个输出的掩码
void GetMask2(const cv::Mat& maskProposals, const cv::Mat& maskProtos, OutputParams& output, const MaskParams& maskParams);

//将边界框转换为旋转矩形框
int BBox2Obb(float centerX, float centerY, float boxW, float boxH, float angle, cv::RotatedRect& rotatedRect);