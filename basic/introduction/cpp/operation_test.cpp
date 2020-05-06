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

	// Resizing: If you want to resize / reshape tensor :
	torch::Tensor m = torch::randn({ 4, 4 });
	torch::Tensor n = m.view(16);
	torch::Tensor k = m.view({ -1,8 });
	std::cout << m.sizes() << std::endl;
	std::cout << n.sizes() << std::endl;
	std::cout << k.sizes() << std::endl;

	// one element tensor, use .item() to get the value
	torch::Tensor r = torch::randn(1);
	std::cout << r << std::endl;
	std::cout << r.item() << std::endl;

	return 0;
}