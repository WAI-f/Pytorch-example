#include "torch/torch.h"
#include "torch/script.h"
#include <iostream>

int main()
{
	// Construct a 5x3 matrix, uninitialized:
	torch::Tensor x = torch::empty({ 5,3 });
	std::cout << x << std::endl;

	// Construct a randomly initialized matrix :
	torch::Tensor y = torch::rand({ 5,3 });
	std::cout << y << std::endl;

	// Construct a matrix filled zerosand of dtype long:
	torch::Tensor z = torch::zeros({ 5,3 }, torch::kInt64);
	std::cout << z << std::endl;

	// Construct a tensor directly from data :
	std::vector<float> m = { 5.5,3 };
	torch::Tensor t = torch::tensor(m);
	std::cout << t << std::endl;

	return 0;
}