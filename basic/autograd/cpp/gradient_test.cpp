#include "torch/script.h"
#include "torch/torch.h"
#include <iostream>

int main()
{
	// network output a scalar
	torch::Tensor x = torch::randn(3, torch::requires_grad());
	torch::Tensor y = x * 2;
	torch::Tensor z = y.sum();
	z.backward();
	std::cout << x.grad() << std::endl;

	// network output a vector
	x = torch::randn(3, torch::requires_grad());
	y = x * 2;
	torch::Tensor v = torch::ones(3, torch::dtype(torch::kFloat32));
	y.backward(v);
	std::cout << x.grad() << std::endl;

	return 0;
}