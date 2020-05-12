#include <stdio.h>
#include <direct.h>
#include <io.h>
#include "opencv.hpp"

void mkdirs(const char* dir)
{
	int len = strlen(dir);
	char str[256];
	strcpy(str, dir);

	for (int i = 0; i < len; i++)
	{
		if (str[i] == '\\' || str[i] == '/')
		{
			str[i] = '\0';
			if (_access(str, 0) != 0)
				_mkdir(str);
			str[i] = '/';
		}
	}
	if (_access(str, 0) != 0)
		_mkdir(str);
}

int main()
{
	std::string data_dir = "E:/myself/vs2019/py-test/cifar-10-batches-bin/";
	std::string train_o_dir = data_dir + "raw_train/";
	std::string test_o_dir = data_dir + "raw_test/";
//test
	std::string test_data_path = data_dir + "test_batch.bin";
	FILE* fp = fopen(test_data_path.c_str(), "rb");
	
	int count = 10000;
	unsigned char tempLabel;
	unsigned char tempRc[1024];
	unsigned char tempGc[1024];
	unsigned char tempBc[1024];

	for (int i = 0; i < count; i++)
	{
		fread(&tempLabel, sizeof(char), 1, fp);
		fread(tempRc, sizeof(char), 1024, fp);
		fread(tempGc, sizeof(char), 1024, fp);
		fread(tempBc, sizeof(char), 1024, fp);

		cv::Mat rc = cv::Mat(32, 32, CV_8UC1, tempRc);
		cv::Mat gc = cv::Mat(32, 32, CV_8UC1, tempGc);
		cv::Mat bc = cv::Mat(32, 32, CV_8UC1, tempBc);
		std::vector<cv::Mat> channels;
		channels.push_back(bc);
		channels.push_back(gc);
		channels.push_back(rc);
		cv::Mat bgr;
		cv::merge(channels, bgr);
		std::string o_dir = test_o_dir + std::to_string(tempLabel);
		mkdirs(o_dir.c_str());
		std::string img_name = std::to_string(tempLabel) + std::string("_") + std::to_string(i) + ".png";
		std::string img_path = o_dir + std::string("/") + img_name;
		cv::imwrite(img_path.c_str(), bgr);
	}
	printf("test_batch loaded.\n");
	fclose(fp);

//train
	for (int j = 1; j < 6; j++)
	{
		std::string data_path = data_dir + std::string("data_batch_");
		data_path = data_path + std::to_string(j) + std::string(".bin");
		FILE* fp = fopen(data_path.c_str(), "rb");

		int count = 10000;
		unsigned char tempLabel;
		unsigned char tempRc[1024];
		unsigned char tempGc[1024];
		unsigned char tempBc[1024];

		printf("train data is loading...\n");

		for (int i = 0; i < count; i++)
		{
			fread(&tempLabel, sizeof(char), 1, fp);
			fread(tempRc, sizeof(char), 1024, fp);
			fread(tempGc, sizeof(char), 1024, fp);
			fread(tempBc, sizeof(char), 1024, fp);

			cv::Mat rc = cv::Mat(32, 32, CV_8UC1, tempRc);
			cv::Mat gc = cv::Mat(32, 32, CV_8UC1, tempGc);
			cv::Mat bc = cv::Mat(32, 32, CV_8UC1, tempBc);
			std::vector<cv::Mat> channels;
			channels.push_back(bc);
			channels.push_back(gc);
			channels.push_back(rc);
			cv::Mat bgr;
			cv::merge(channels, bgr);
			std::string o_dir = train_o_dir + std::to_string(tempLabel);
			mkdirs(o_dir.c_str());
			std::string img_name = std::to_string(tempLabel) + std::string("_") + std::to_string(i + (j - 1) * 10000) + ".png";
			std::string img_path = o_dir + std::string("/") + img_name;
			cv::imwrite(img_path.c_str(), bgr);
		}
		printf("train_batch loaded.\n");
		fclose(fp);
	}

	return 0;
}


// format:1 byte label, 1024 r, 1024 g, 1024 b