#include <stdio.h>
#include <io.h>
#include <vector>
#include <iostream>

void list_dir(const char* dir,std::vector<std::string> &file_list)
{
	std::string new_dir = std::string(dir) + "*.*";
	intptr_t handle;
	_finddata_t findData;

	handle = _findfirst(new_dir.c_str(), &findData);
	if (handle == -1)
		return;

	do {
		if (findData.attrib & _A_SUBDIR)
		{
			if (strcmp(findData.name, ".") == 0 || strcmp(findData.name, "..") == 0)
				continue;
			std::string sub_dir = std::string(dir) + std::string(findData.name) + "/";
			list_dir(sub_dir.c_str(), file_list);
		}
		else
		{
			std::string file = std::string(dir) + std::string(findData.name);
			file_list.push_back(file);
		}
	} while (_findnext(handle, &findData) == 0);
	
	_findclose(handle);
}

void gen_txt(char* txt_path, char* img_dir, char* flag)
{
	FILE* fp = fopen(txt_path, "w");

	std::vector<std::string> file_list;
	list_dir(img_dir, file_list);

	for (int i = 0; i < file_list.size(); i++)
	{
		std::string file = file_list[i];
		size_t pos1 = file.find_last_of("\\/");
		size_t pos2 = file.find_last_of("_");
		std::string label = file.substr(pos1 + 1, pos2 - pos1 - 1);
		fprintf(fp, "%s %s %s\n", file.c_str(), label.c_str(), flag);
	}

	fclose(fp);
}

int main()
{
	char train_dir[256] = "E:/program/test/py-test/cifar-10-batches-bin/raw_train/";
	char train_txt_path[256] = "E:/program/test/py-test/cifar-10-batches-bin/train.txt";

	char test_dir[256] = "E:/program/test/py-test/cifar-10-batches-bin/raw_test/";
	char test_txt_path[256] = "E:/program/test/py-test/cifar-10-batches-bin/test.txt";

	char flag1[10] = "train";
	char flag2[10] = "test";
	gen_txt(train_txt_path, train_dir, flag1);
	gen_txt(test_txt_path, test_dir, flag2);

	printf("generate done.\n");

	return 0;
}