#include "torch/script.h"
#include <iostream>

int main()
{
	// Create a tensorand set requires_grad = True to track computation with it
	torch::Tensor x = torch::ones({ 2, 2 }, torch::requires_grad());
	std::cout << x << std::endl;

	// Do a tensor operation
	torch::Tensor y = x + 2;
	std::cout << y << std::endl;

	// change requires_grad flag in - place
	torch::Tensor a = torch::randn({ 2, 2 });
	std::cout << a.grad_fn() << std::endl;
	a = ((a * 3) / (a - 1));
	std::cout << a.grad_fn() << std::endl;
	std::cout << a.requires_grad() << std::endl;
	a.requires_grad_(true);
	std::cout << a.requires_grad() << std::endl;
	a = ((a * 3) / (a - 1));
	std::cout << a.grad_fn() << std::endl;

	return 0;
}