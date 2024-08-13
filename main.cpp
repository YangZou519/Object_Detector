#include <iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
#include "yolov8_onnx.h"
#include<time.h>


using namespace cv;
using namespace dnn;

//用于圖像檢測的函數，接受圖像路徑和顔色向量作爲參數
void Image_Detect(const std::string& img_path, const std::vector<Scalar>& color);
//用於視頻檢測的函數，接受命令行參數，視頻路徑和顔色向量作爲參數
void Video_Detect(int argc, char** argv, const std::string& video_path, const std::vector<Scalar>& color);


int main(int argc, char** argv) {
	std::vector<Scalar> color; //存儲顏色值
	srand(time(0));
	//生成81個隨機顔色
	for (int i = 0; i < 81; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		//將生成的顔色值以Scalar類型添加到color向量中
		color.push_back(Scalar(b, g, r));
	}

	std::string img_path = "C:/Users/25442/Pictures/person2.jpg";
	std::string video_path = "C:/Users/25442/Pictures/video_4.mp4"; 
	std::string wrong_detect_path = "C:/Users/25442/Pictures/video_2.mp4";
	//Image_Detect(img_path, color);
	//Video_Detect(argc, argv, video_path, color);
	Video_Detect(argc, argv, wrong_detect_path, color);
	return 0;
}

void Image_Detect(const std::string& img_path, const std::vector<Scalar>& color) {
	std::string model_path_onnx = "./yolov8m.onnx";
	Yolov8Onnx task_detect_onnx; 
	cv::Mat src = imread(img_path);
	cv::Mat img = src.clone(); //克隆圖像src並存儲在img中，避免直接修改原圖
	//加載ONNX模型。如果模型加載成功，輸出“read net ok！”
	if (task_detect_onnx.ReadModel(model_path_onnx, false)) {
		std::cout << "read net ok!" << std::endl;
	}
	//失敗則返回
	else {
		std::cout << "read net error!" << std::endl;
		return;
	}
	std::vector<OutputParams> result;
	bool res = task_detect_onnx.OnnxDetect(src, result);

	//如果檢測成功，調用DrawPred函數在圖像上繪製檢測結果
	if (res) {
		DrawPred(src, result, task_detect_onnx._className, color);
	}
	//cv::imwrite("result.jpg", src);
	cv::imshow("result", src);
	cv::waitKey(0);
}
void Video_Detect(int argc, char** argv, const std::string& video_path, const std::vector<Scalar>& color) {
	VideoCapture cap;
	//檢查命令行參數的數量。
	//大於一則通過命令行提供了一個額外的輸入參數
	//argv[0]通常是程序的名稱，argv[1]就是提供的第一個參數
	if (argc > 1) {
		std::string input_source = argv[1];
		if (isdigit(input_source[0])) {
			int camera_index = stoi(input_source);
			cap.open(camera_index);
		}
		else {
			cap.open(input_source);
		}
	}
	else {
		cap.open(video_path);
	}
	if (!cap.isOpened()) {
		std::cerr << "Error: Unable to open the video source." << std::endl;
		return ;
	}

	std::string model_path_onnx = "./yolov8m.onnx";
	Yolov8Onnx task_detect_onnx;
	if (task_detect_onnx.ReadModel(model_path_onnx, false)) {
		std::cout << "Read net ok!" << std::endl;
	}
	else {
		std::cout << "Read net error!" << std::endl;
		return;
	}
	Mat frame;
	int frame_cout = 0;
	int skip_frames = 10;
	//逐幀處理視頻
	while (true) {
		//從視頻流中讀取一幀圖像，並將其存儲在frame變量中
		cap >> frame;
		if (frame.empty()) break;
		frame_cout++;
		//跳过某些帧
		//通過計算frame_cout與skip_frames的餘數，決定是否跳過當前幀的檢測。如果餘數不爲零，則跳過當前幀的處理。
		if (frame_cout % (skip_frames + 1) != 0) continue;

		std::vector<OutputParams> result;
		bool res = task_detect_onnx.OnnxDetect(frame, result);
		//如果檢測成功，則調用DrawPred函數在當前幀上繪製檢測結果
		if (res) {
			DrawPred(frame, result, task_detect_onnx._className, color);
		}
		cv::imshow("Video Detection", frame);
		if (waitKey(30) >= 0) break;
	}
	//釋放視頻捕獲對象，關閉視頻文件
	cap.release();
	destroyAllWindows;
}









