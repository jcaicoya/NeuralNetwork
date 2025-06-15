#include "neural_net/network.hpp"
#include <random>
#include <cmath>
#include <cassert>
#include <fstream>
#include "activation.hpp"

namespace nn {

// ======= Constructor =======
Network::Network(std::vector<int> layers, ActivationType activation)
    : layer_sizes_(std::move(layers)), activation_type_(activation) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0, 1);

    for (size_t i = 1; i < layer_sizes_.size(); ++i) {
        int curr = layer_sizes_[i];
        int prev = layer_sizes_[i - 1];

        biases_.emplace_back(curr);
        for (double& b : biases_.back()) {
            b = dist(gen);
        }

        weights_.emplace_back(prev, std::vector<double>(curr));
        for (auto& from : weights_.back()) {
            for (double& w : from) {
                w = dist(gen);
            }
        }
    }

    // Set function pointers
    switch (activation_type_) {
        case ActivationType::Sigmoid:
            activation_ = Activation::sigmoid;
            activation_derivative_ = Activation::sigmoid_derivative;
            break;
        case ActivationType::Tanh:
            activation_ = Activation::tanh;
            activation_derivative_ = Activation::tanh_derivative;
            break;
        case ActivationType::ReLU:
            activation_ = Activation::relu;
            activation_derivative_ = Activation::relu_derivative;
            break;
        case ActivationType::LeakyReLU:
            activation_ = Activation::leakey_relu;
            activation_derivative_ = Activation::leakey_relu_derivative;
            break;
        default:
            throw std::runtime_error("Unknown activation type");
    }
}

// ======= Forward =======
std::vector<double> Network::forward(const std::vector<double>& input) {
    std::vector<double> layer_input = input;

    for (size_t l = 0; l < weights_.size(); ++l) {
        std::vector<double> layer_output(layer_sizes_[l + 1]);

        for (int j = 0; j < layer_sizes_[l + 1]; ++j) {
            double z = biases_[l][j];
            for (int i = 0; i < layer_sizes_[l]; ++i) {
                z += weights_[l][i][j] * layer_input[i];
            }
            layer_output[j] = activation_(z);
        }

        layer_input = std::move(layer_output);
    }

    return layer_input;
}

// ======= Train =======
void Network::train(const std::vector<double>& input,
                    const std::vector<double>& target,
                    double learning_rate) {
    // Forward pass: record activations and pre-activations
    std::vector<std::vector<double>> activations = { input };
    std::vector<std::vector<double>> zs;

    for (size_t l = 0; l < weights_.size(); ++l) {
        std::vector<double> z(layer_sizes_[l + 1]);
        std::vector<double> a(layer_sizes_[l + 1]);

        for (int j = 0; j < layer_sizes_[l + 1]; ++j) {
            z[j] = biases_[l][j];
            for (int i = 0; i < layer_sizes_[l]; ++i) {
                z[j] += weights_[l][i][j] * activations[l][i];
            }
            a[j] = activation_(z[j]);
        }

        zs.push_back(std::move(z));
        activations.push_back(std::move(a));
    }

    // Backward pass
    std::vector<std::vector<double>> delta(weights_.size());

    // Output layer delta
    size_t last = weights_.size() - 1;
    delta[last].resize(layer_sizes_.back());
    for (int i = 0; i < layer_sizes_.back(); ++i) {
        double z = zs[last][i];
        double a = activations.back()[i];
        delta[last][i] = (a - target[i]) * activation_derivative_(z);
    }

    // Hidden layers
    for (int l = static_cast<int>(weights_.size()) - 2; l >= 0; --l) {
        delta[l].resize(layer_sizes_[l + 1]);
        for (int i = 0; i < layer_sizes_[l + 1]; ++i) {
            double sum = 0.0;
            for (int j = 0; j < layer_sizes_[l + 2]; ++j) {
                sum += weights_[l + 1][i][j] * delta[l + 1][j];
            }
            double z = zs[l][i];
            delta[l][i] = sum * activation_derivative_(z);
        }
    }

    // Gradient descent
    for (size_t l = 0; l < weights_.size(); ++l) {
        for (int i = 0; i < layer_sizes_[l]; ++i) {
            for (int j = 0; j < layer_sizes_[l + 1]; ++j) {
                weights_[l][i][j] -= learning_rate * delta[l][j] * activations[l][i];
            }
        }
        for (int j = 0; j < layer_sizes_[l + 1]; ++j) {
            biases_[l][j] -= learning_rate * delta[l][j];
        }
    }
}

void nn::Network::save(const std::string& filepath) const {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for saving: " + filepath);
    }

    // Save layer sizes
    const int size = static_cast<int>(layer_sizes_.size());
    out.write(reinterpret_cast<const char*>(&size), sizeof(int));
    out.write(reinterpret_cast<const char*>(layer_sizes_.data()), sizeof(int) * size);

    // Save biases
    for (const auto& layer_biases : biases_) {
        out.write(reinterpret_cast<const char*>(layer_biases.data()), sizeof(double) * layer_biases.size());
    }

    // Save weights
    for (const auto& layer_weights : weights_) {
        for (const auto& from_weights : layer_weights) {
            out.write(reinterpret_cast<const char*>(from_weights.data()), sizeof(double) * from_weights.size());
        }
    }

    out.close();
}


void nn::Network::load(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for loading: " + filepath);
    }

    // Load layer sizes
    int size;
    in.read(reinterpret_cast<char*>(&size), sizeof(int));

    std::vector<int> loaded_layers(size);
    in.read(reinterpret_cast<char*>(loaded_layers.data()), sizeof(int) * size);
    layer_sizes_ = std::move(loaded_layers);

    // Reset containers
    biases_.clear();
    weights_.clear();

    // Load biases
    for (size_t i = 1; i < layer_sizes_.size(); ++i) {
        int curr = layer_sizes_[i];
        std::vector<double> b(curr);
        in.read(reinterpret_cast<char*>(b.data()), sizeof(double) * curr);
        biases_.push_back(std::move(b));
    }

    // Load weights
    for (size_t i = 1; i < layer_sizes_.size(); ++i) {
        int prev = layer_sizes_[i - 1];
        int curr = layer_sizes_[i];
        std::vector<std::vector<double>> w(prev, std::vector<double>(curr));
        for (int from = 0; from < prev; ++from) {
            in.read(reinterpret_cast<char*>(w[from].data()), sizeof(double) * curr);
        }
        weights_.push_back(std::move(w));
    }

    in.close();
}

} // namespace nn
