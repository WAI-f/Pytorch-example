#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Options {
    int image_size = 32;
    size_t train_batch_size = 4;
    size_t test_batch_size = 4;
    size_t iterations = 2;
    size_t log_interval = 2000;
    // path must end in delimiter
    std::string datasetPath = "E:/program/test/py-test/cifar-10-batches-bin/";
    std::string infoFilePath = "E:/program/test/py-test/cifar-10-batches-bin/total.txt";
    torch::DeviceType device = torch::kCPU;
};

static Options options;

using Data = std::vector<std::pair<std::string, long long>>;

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {
    using Example = torch::data::Example<>;

    Data data;

public:
    CustomDataset(const Data& data) : data(data) {}

    Example get(size_t index) {
        std::string path = data[index].first;
        auto mat = cv::imread(path);
        //cv::imshow(path, mat);
        //cv::waitKey(0);
        assert(!mat.empty());

        cv::resize(mat, mat, cv::Size(options.image_size, options.image_size));
        std::vector<cv::Mat> channels(3);
        cv::split(mat, channels);

        auto R = torch::from_blob(
            channels[2].ptr(),
            { options.image_size, options.image_size },
            torch::kUInt8);
        auto G = torch::from_blob(
            channels[1].ptr(),
            { options.image_size, options.image_size },
            torch::kUInt8);
        auto B = torch::from_blob(
            channels[0].ptr(),
            { options.image_size, options.image_size },
            torch::kUInt8);

        auto tdata = torch::cat({ R, G, B })
            .view({ 3, options.image_size, options.image_size })
            .to(torch::kFloat);
        tdata = tdata / 255;
        auto tlabel = torch::from_blob(&data[index].second, { 1 }, torch::kLong);
        
        return { tdata, tlabel };
    }

    torch::optional<size_t> size() const {
        return data.size();
    }
};

std::pair<Data, Data> readInfo() {
    Data train, test;

    std::ifstream stream(options.infoFilePath);
    assert(stream.is_open());

    long long label;
    std::string path, type;

    while (true) {
        stream >> path >> label >> type;
        //std::cout << path << "\t" << label << std::endl;

        if (type == "train")
            train.push_back(std::make_pair(path, label));
        else if (type == "test")
            test.push_back(std::make_pair(path, label));
        else
            assert(false);

        if (stream.eof())
            break;
    }

    std::random_shuffle(train.begin(), train.end());
    return std::make_pair(train, test);
}

struct NetworkImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{ nullptr };
    torch::nn::Conv2d conv2{ nullptr };
    torch::nn::MaxPool2d pool1{ nullptr };
    torch::nn::MaxPool2d pool2{ nullptr };
    torch::nn::Linear fc1{ nullptr };
    torch::nn::Linear fc2{ nullptr };
    torch::nn::Linear fc3{ nullptr };

    NetworkImpl() {
        conv1 = register_module("conv1", torch::nn::Conv2d(3, 6, 5));
        pool1 = register_module("pool1", torch::nn::MaxPool2d(2));
        pool2 = register_module("pool2", torch::nn::MaxPool2d(2));
        conv2 = register_module("conv2", torch::nn::Conv2d(6, 16, 5));
        fc1 = register_module("fc1", torch::nn::Linear(16 * 5 * 5, 120));
        fc2 = register_module("fc2", torch::nn::Linear(120, 84));
        fc3 = register_module("fc3", torch::nn::Linear(84, 10));
    }
    torch::Tensor forward(torch::Tensor x)
    {
        x = pool1(torch::relu(conv1->forward(x)));
        x = pool2(torch::relu(conv2->forward(x)));
        x = x.view({ -1, 16 * 5 * 5 });
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        x = fc3(x);
        return x;
    }
};
TORCH_MODULE(Network);

template <typename DataLoader>
void train(
    Network& network,
    DataLoader& loader,
    torch::optim::Optimizer& optimizer,
    size_t epoch,
    size_t data_size) {
    size_t index = 0;
    network->train();
    float Loss = 0, Acc = 0;

    for (auto& batch : loader) {
        auto data = batch.data.to(options.device);
        auto targets = batch.target.to(options.device).squeeze_();
        //std::cout << data << std::endl;
        //std::cout << targets << std::endl;
        auto output = network->forward(data);
        auto criterion = torch::nn::CrossEntropyLoss();
        //std::cout << output << std::endl;
        auto loss = criterion->forward(output, targets);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        Loss += loss.template item<float>();

        if (index++ % options.log_interval == 1999) {
            auto end = std::min(data_size, (index + 1) * options.train_batch_size);

            std::cout << "Train Epoch: " << epoch << " " << end << "/" << index
                << "\tLoss: " << Loss / 2000 << std::endl;
            Loss = 0;
        }
    }
}

template <typename DataLoader>
void test(Network& network, DataLoader& loader, size_t data_size) {
    size_t index = 0;
    network->eval();
    torch::NoGradGuard no_grad;
    float Loss = 0, Acc = 0;

    for (const auto& batch : loader) {
        auto data = batch.data.to(options.device);
        auto targets = batch.target.to(options.device).squeeze_();

        auto output = network->forward(data);
        auto criterion = torch::nn::CrossEntropyLoss();
        auto loss = criterion->forward(output, targets);
        auto acc = output.argmax(1).eq(targets).sum();

        Loss += loss.template item<float>();
        Acc += acc.template item<float>();
    }

    if (index++ % options.log_interval == 0)
        std::cout << "Test Loss: " << Loss / data_size << "\tAcc: " << Acc / data_size << std::endl;
}

int main() {
    torch::manual_seed(1);

    if (torch::cuda::is_available())
        options.device = torch::kCUDA;
    std::cout << "Running on: "
        << (options.device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

    auto data = readInfo();

    auto train_set =
        CustomDataset(data.first).map(torch::data::transforms::Normalize<>({ 0.5,0.5,0.5 }, { 0.5,0.5,0.5 })).map(torch::data::transforms::Stack<>());
    auto train_size = train_set.size().value();
    auto train_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(train_set), options.train_batch_size);

    auto test_set =
        CustomDataset(data.second).map(torch::data::transforms::Normalize<>({ 0.5,0.5,0.5 }, { 0.5,0.5,0.5 })).map(torch::data::transforms::Stack<>());
    auto test_size = test_set.size().value();
    auto test_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(test_set), options.test_batch_size);

    Network network;
    network->to(options.device);

    torch::optim::SGD optimizer(
        network->parameters(), torch::optim::SGDOptions(0.001).momentum(0.9));

    for (size_t i = 0; i < options.iterations; ++i) {
        train(network, *train_loader, optimizer, i + 1, train_size);
    }

    test(network, *test_loader, test_size);

    return 0;
}