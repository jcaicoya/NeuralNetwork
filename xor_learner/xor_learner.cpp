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
constexpr bool train_network = true;
constexpr bool show_training_loss = false;

// Configuration values
constexpr int epochs = 20'000;
constexpr double learning_rate = 0.01;
const std::string model_path = "xor_weights.bin";


int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    nn::Network net({2, 4, 4, 1}, nn::ActivationType::Sigmoid);

    if constexpr (load_from_file) {
        try {
            net.load(model_path);
            std::cout << "✅ Model loaded from file.\n";
        } catch (...) {
            std::cout << "⚠️ No model file found. Starting from scratch.\n";
        }
    } else {
        std::cout << "🚫 Skipping model load. Starting from scratch.\n";
    }

    if constexpr (train_network) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                net.train(inputs[i], targets[i], learning_rate);
            }

            if constexpr (show_training_loss) {
                if (epoch % 1000 == 0) {
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
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = net.forward(inputs[i]);
        const double result = output[0];
        std::cout << inputs[i][0] << " XOR " << inputs[i][1]
                  << " = " << result << "\n";
    }

    return 0;
}
