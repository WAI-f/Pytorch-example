#include "torch/script.h"
#include <iostream>

int main()
{
	// Addition:syntax 1
	torch::Tensor x = torch::ones({ 5,3 });
	torch::Tensor y = torch::rand({ 5,3 });
	std::cout << x + y << std::endl;

	// Addition: syntax 2
	torch::Tensor result;
	result = torch::add(x, y);
	std::cout << result << std::endl;

	// Addition: syntax 3, in - place
	y.add_(x);
	std::cout << y << std::endl;

	return 0;
}