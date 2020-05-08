#include "torch/script.h"
#include "torch/torch.h"
#include <iostream>

// basic model define
struct  Net : torch::nn::Module
{
	torch::nn::Conv2d conv1{ nullptr };
	torch::nn::Conv2d conv2{ nullptr };
	torch::nn::Linear fc1{ nullptr };
	torch::nn::Linear fc2{ nullptr };
	torch::nn::Linear fc3{ nullptr };

	Net()
	{
		conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 6, 3)));
		conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(6, 16, 3)));
		fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(16 * 6 * 6, 120)));
		fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(120, 84)));
		fc3 = register_module("fc3", torch::nn::Linear(torch::nn::LinearOptions(84, 10)));
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::relu(conv1->forward(x));
		x = torch::max_pool2d(x, 2);
		x = torch::relu(conv2->forward(x));
		x = torch::max_pool2d(x, 2);

		x = x.view({ -1,16 * 6 * 6 });
		x = torch::relu(fc1->forward(x));
		x = torch::relu(fc2->forward(x));
		x = fc3->forward(x);

		return x;
	}
};

int main()
{
	// see net structure
	auto net = Net();
	std::cout << net << std::endl;

	// view trainable parameters
	for (auto& param : net.parameters())
	{
		std::cout << param.sizes() << std::endl;
	}

	// random input
	auto input = torch::randn({ 1,1,32,32 });
	auto out = net.forward(input);
	std::cout << out << std::endl;

	// calculate MSE Loss
	auto target = torch::randn(10);
	target = target.view({ 1, -1 });
	auto criterion = torch::nn::MSELoss();
	auto loss = criterion(out, target);
	std::cout << loss << std::endl;

	// check conv1 bias gradients before and after the backward
	net.zero_grad();
	std::cout << net.conv1.get()->bias.grad() << std::endl; // before backward
	loss.backward();
	std::cout << net.conv1.get()->bias.grad() << std::endl; // after backward

	return 0;
}