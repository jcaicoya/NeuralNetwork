#include "neural_net/network.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#endif

// XOR training dataset
const std::vector<std::vector<double>> inputs = {
    {0, 0}, {0, 1}, {1, 0}, {1, 1}
};

const std::vector<std::vector<double>> targets = {
    {0}, {1}, {1}, {0}
};

// 🔧 Configuration flags
constexpr bool load_from_file = true;
constexpr bool train_network = false;
constexpr bool show_training_loss = false;

// Configuration values
const std::vector<int> layers {2, 4, 4, 1 };
constexpr double learning_rate = 0.01;
constexpr int epochs = 20'000;


void print_line() {
    constexpr std::size_t line_length = 60;
    constexpr char c = '-';
    std::cout << std::string(line_length, c) << std::endl;
}

void train_xor(nn::ActivationType activation_type) {
    std::cout << "Training XOR using ";
    nn::Network net(layers, activation_type);
    std::string model_path;
    switch (activation_type) {
        case nn::ActivationType::Sigmoid:
            std::cout << "Sigmoid";
            model_path = "xor_weights_sigmoid.bin";
            break;
        case nn::ActivationType::Tanh:
            std::cout << "Tanh";
            model_path = "xor_weights_tanh.bin";
            break;
        case nn::ActivationType::ReLU:
            std::cout << "ReLU";
            model_path = "xor_weights_relu.bin";
            break;
        case nn::ActivationType::LeakyReLU:
            std::cout << "LeakyReLU";
            model_path = "xor_weights_leaky_relu.bin";
            break;
    }
    std::cout << " activation_type\n";

    bool starting_from_scratch = false;
    if constexpr (load_from_file) {
        try {
            net.load(model_path);
            std::cout << "✅ Model loaded from " << model_path << " file.\n";
        } catch (...) {
            std::cout << "⚠️ No model file found (" << model_path << "). Starting from scratch.\n";
            starting_from_scratch = true;
        }
    } else {
        std::cout << "🚫 Skipping model load. Starting from scratch.\n";
        starting_from_scratch = true;
    }

    if (train_network or starting_from_scratch) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                net.train(inputs[i], targets[i], learning_rate);
            }

            if constexpr (show_training_loss) {
                if (epoch % 1'000 == 0) {
                    double loss = 0.0;
                    for (size_t i = 0; i < inputs.size(); ++i) {
                        auto out = net.forward(inputs[i]);
                        loss += std::pow(out[0] - targets[i][0], 2);
                    }
                    std::cout << "Epoch " << epoch << ", loss = " << loss << "\n";
                }
            }
        }

        net.save(model_path);
        std::cout << "💾 Model saved to file.\n";
    } else {
        std::cout << "🧪 Skipping training. Using existing weights.\n";
    }

    // Test model on XOR inputs
    std::cout << "\n🧠 XOR Results:\n";
    for (const auto & input : inputs) {
        auto output = net.forward(input);
        const double result = output[0];
        std::cout << input[0] << " XOR " << input[1]
                  << " = " << result << "\n";
    }
}


int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    const std::vector<nn::ActivationType> activation_types = {
        nn::ActivationType::Sigmoid,
        nn::ActivationType::Tanh,
        nn::ActivationType::LeakyReLU
    };

    for (const auto activation_type : activation_types) {
        print_line();
        train_xor(activation_type);
        print_line();
    }

    return 0;
}
