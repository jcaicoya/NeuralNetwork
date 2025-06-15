#include "neural_net/network.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

// Training data for XOR
const std::vector<std::vector<double>> inputs = {
    {0, 0}, {0, 1}, {1, 0}, {1, 1}
};

const std::vector<std::vector<double>> targets = {
    {0}, {1}, {1}, {0}
};

int main() {
    nn::Network net({2, 4, 4, 1});

    const std::string model_path = "xor_weights.bin";

    // Try loading existing weights
    try {
        net.load(model_path);
        std::cout << "Model loaded from file.\n";
    } catch (...) {
        std::cout << "No saved model found. Training from scratch.\n";

        constexpr int epochs = 20'000;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                constexpr double learning_rate = 0.1;
                net.train(inputs[i], targets[i], learning_rate);
            }

            if (epoch % 1'000 == 0) {
                double loss = 0;
                for (size_t i = 0; i < inputs.size(); ++i) {
                    auto out = net.forward(inputs[i]);
                    loss += std::pow(out[0] - targets[i][0], 2);
                }
                std::cout << "Epoch " << epoch << ", loss = " << loss << "\n";
            }
        }

        net.save(model_path);
        std::cout << "Model saved to file.\n";
    }

    // Test output
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = net.forward(inputs[i]);
        std::cout << inputs[i][0] << " XOR " << inputs[i][1] << " = " << output[0] << "\n";
    }

    return 0;
}

