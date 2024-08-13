#include "yolov8_onnx.h"
using namespace Ort;

//读取ONNX模型
bool Yolov8Onnx::ReadModel(const std::string& modelPath, bool isCuda, int cudaID, bool warmUp) {
	//如果_batchSize小于1则将其设置为1
	if (_batchSize < 1) _batchSize = 1;
	       
	try
	{
		//检查路径是否合法
		if (!CheckModelPath(modelPath))
			return false;
		//设置会话选项的凸优化级别为ORT_ENABLE_EXTENDED
		_OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
		
		//根据平台创建ONNX运行时会话_OrtSession
#ifdef _WIN32
		std::wstring model_path(modelPath.begin(), modelPath.end());
		_OrtSession = new Ort::Session(_OrtEnv, model_path.c_str(),     _OrtSessionOptions);
#else
		_OrtSession = new Ort::Session(_OrtEnv, modelPath.c_str(), _OrtSessionOptions);
#endif
		//创建默认选项的分配器
		Ort::AllocatorWithDefaultOptions allocator;
		//获取输入节点数
		_inputNodesNum = _OrtSession->GetInputCount();

		//根据ONNX API 版本获取输入节点名称，并添加到_inputNodeNames向量
#if ORT_API_VERSION < ORT_OLD_VISON
		_inputName = _OrtSession->GetInputName(0, allocator);
		_inputNodeNames.push_back(_inputName);
#else
		_inputName = std::move(_OrtSession->GetInputNameAllocated(0, allocator));
		_inputNodeNames.push_back(_inputName.get());
#endif
		//获取输入节点的类型信息
		Ort::TypeInfo inputTypeInfo = _OrtSession->GetInputTypeInfo(0);
		//获取输入张量的类型和形状信息
		auto input_tensor_info = inputTypeInfo.GetTensorTypeAndShapeInfo();
		//获取输入节点的数据类型
		_inputNodeDataType = input_tensor_info.GetElementType();
		//获取输入张量的形状
		_inputTensorShape = input_tensor_info.GetShape();
		  
		if (_inputTensorShape[0] == -1)
		{
			_isDynamicShape = true;
			_inputTensorShape[0] = _batchSize;

		}
		if (_inputTensorShape[2] == -1 || _inputTensorShape[3] == -1) {
			_isDynamicShape = true;
			_inputTensorShape[2] = _netHeight;
			_inputTensorShape[3] = _netWidth;
		}
		//init output
		_outputNodesNum = _OrtSession->GetOutputCount();
#if ORT_API_VERSION < ORT_OLD_VISON
		_output_name0 = _OrtSession->GetOutputName(0, allocator);
		_outputNodeNames.push_back(_output_name0);
#else
		_output_name0 = std::move(_OrtSession->GetOutputNameAllocated(0, allocator));
		_outputNodeNames.push_back(_output_name0.get());
#endif
		//获取输出节点的类型信息
		Ort::TypeInfo type_info_output0(nullptr);
		type_info_output0 = _OrtSession->GetOutputTypeInfo(0);  //output0

		//获取输出张量的类型和形状信息
		auto tensor_info_output0 = type_info_output0.GetTensorTypeAndShapeInfo();
		_outputNodeDataType = tensor_info_output0.GetElementType();
		//获取输出张量的形状
		_outputTensorShape = tensor_info_output0.GetShape();

		//warm up
		//如果使用CUDA并且需要预热模型，则进行预热
		if (isCuda && warmUp) {
			//draw run
			std::cout << "Start warming up" << std::endl;
			//创建一个临时张量并进行模型若干次以进行预热
			size_t input_tensor_length = VectorProduct(_inputTensorShape);
			float* temp = new float[input_tensor_length];
			std::vector<Ort::Value> input_tensors;
			std::vector<Ort::Value> output_tensors;
			input_tensors.push_back(Ort::Value::CreateTensor<float>(
				_OrtMemoryInfo, temp, input_tensor_length, _inputTensorShape.data(),
				_inputTensorShape.size()));
			for (int i = 0; i < 3; ++i) {
				output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
					_inputNodeNames.data(),
					input_tensors.data(),
					_inputNodeNames.size(),
					_outputNodeNames.data(),
					_outputNodeNames.size());
			} 

			delete[]temp;
		}
	}
	//捕获所有异常，如果发生异常则返回false
	catch (const std::exception&) {
		return false;
	}
	return true;

}

int Yolov8Onnx::Preprocessing(const std::vector<cv::Mat>& srcImgs, std::vector<cv::Mat>& outSrcImgs, std::vector<cv::Vec4d>& params) {
	//清空输出图像向量
	outSrcImgs.clear();

	//对每张输入图像进行尺寸，保持纵横比并填充边框
	//将调整好的图像和对应的参数保存到输出向量中
	cv::Size input_size = cv::Size(_netWidth, _netHeight);
	for (int i = 0; i < srcImgs.size(); ++i) {
		cv::Mat temp_img = srcImgs[i];
		cv::Vec4d temp_param = { 1,1,0,0 };
		if (temp_img.size() != input_size) {
			cv::Mat borderImg;
			LetterBox(temp_img, borderImg, temp_param, input_size, false, false, true, 32);
			outSrcImgs.push_back(borderImg);
			params.push_back(temp_param);
		}
		else {
			outSrcImgs.push_back(temp_img);
			params.push_back(temp_param);
		}
	}

	//如果批处理大小大于出入图像数量，则填充空白图像
	int lack_num = _batchSize - srcImgs.size();
	if (lack_num > 0) {
		for (int i = 0; i < lack_num; ++i) {
			cv::Mat temp_img = cv::Mat::zeros(input_size, CV_8UC3);
			cv::Vec4d temp_param = { 1,1,0,0 };
			outSrcImgs.push_back(temp_img);
			params.push_back(temp_param);
		}
	}
	return 0;

}

bool Yolov8Onnx::OnnxDetect(cv::Mat& srcImg, std::vector<OutputParams>& output) {
	//接受一个图像
	std::vector<cv::Mat> input_data = { srcImg };
	std::vector<std::vector<OutputParams>> tenp_output;
	//调用OnnxBatchDetect进行批量检测，成功则将结果保存到output并返回true
	if (OnnxBatchDetect(input_data, tenp_output)) {
		output = tenp_output[0];
		return true;
	}
	else return false;
}
bool Yolov8Onnx::OnnxBatchDetect(std::vector<cv::Mat>& srcImgs, std::vector<std::vector<OutputParams>>& output) {
	std::vector<cv::Vec4d> params;
	std::vector<cv::Mat> input_images;
	cv::Size input_size(_netWidth, _netHeight);
	//preprocessing
	//进行预处理
	Preprocessing(srcImgs, input_images, params);

	//将图像转化为blob
	cv::Mat blob = cv::dnn::blobFromImages(input_images, 1 / 255.0, input_size, cv::Scalar(0, 0, 0), true, false);

	int64_t input_tensor_length = VectorProduct(_inputTensorShape);
	std::vector<Ort::Value> input_tensors;
	std::vector<Ort::Value> output_tensors;
	input_tensors.push_back(Ort::Value::CreateTensor<float>(_OrtMemoryInfo, (float*)blob.data, input_tensor_length, _inputTensorShape.data(), _inputTensorShape.size()));

	output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
		_inputNodeNames.data(),
		input_tensors.data(),
		_inputNodeNames.size(),
		_outputNodeNames.data(),
		_outputNodeNames.size()
	);
	//post-process
	float* all_data = output_tensors[0].GetTensorMutableData<float>();
	_outputTensorShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
	int net_width = _outputTensorShape[1];
	int socre_array_length = net_width - 4;
	int64_t one_output_length = VectorProduct(_outputTensorShape) / _outputTensorShape[0];
	for (int img_index = 0; img_index < srcImgs.size(); ++img_index) {
		cv::Mat output0 = cv::Mat(cv::Size((int)_outputTensorShape[2], (int)_outputTensorShape[1]), CV_32F, all_data).t();  //[bs,116,8400]=>[bs,8400,116]
		all_data += one_output_length;
		float* pdata = (float*)output0.data;
		int rows = output0.rows;
		std::vector<int> class_ids;//分类的id列表
		std::vector<float> confidences;//每个id对应的置信度列表
		std::vector<cv::Rect> boxes;//每个id的矩形框
		for (int r = 0; r < rows; ++r) {    //stride
			cv::Mat scores(1, socre_array_length, CV_32F, pdata + 4);
			cv::Point classIdPoint;
			double max_class_socre;
			minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
			max_class_socre = (float)max_class_socre;
			if (max_class_socre >= _classThreshold) {
				float x = (pdata[0] - params[img_index][2]) / params[img_index][0];  //x
				float y = (pdata[1] - params[img_index][3]) / params[img_index][1];  //y
				float w = pdata[2] / params[img_index][0];  //w
				float h = pdata[3] / params[img_index][1];  //h
				int left = MAX(int(x - 0.5 * w + 0.5), 0);
				int top = MAX(int(y - 0.5 * h + 0.5), 0);
				class_ids.push_back(classIdPoint.x);
				confidences.push_back(max_class_socre);
				boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
			}
			pdata += net_width;//下一行
		}

		std::vector<int> nms_result;
		//进行非极大值抑制
		cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold, nms_result);
		std::vector<std::vector<float>> temp_mask_proposals;
		cv::Rect holeImgRect(0, 0, srcImgs[img_index].cols, srcImgs[img_index].rows);
		std::vector<OutputParams> temp_output;
		for (int i = 0; i < nms_result.size(); ++i) {
			int idx = nms_result[i];
			OutputParams result;
			result.id = class_ids[idx];
			result.confidence = confidences[idx];
			result.box = boxes[idx] & holeImgRect;
			temp_output.push_back(result);
		}
		output.push_back(temp_output);
	}

	if (output.size())
		return true;
	else
		return false;
}