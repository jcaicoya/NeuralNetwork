#include "neural_net/network.hpp"
#include <random>
#include <cmath>
#include <cassert>
#include <fstream>

namespace nn {

static double sigmoid_scalar(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

static double sigmoid_derivative_scalar(double x) {
    double s = sigmoid_scalar(x);
    return s * (1 - s);
}

Network::Network(const std::vector<int>& layers) : layer_sizes_(layers) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1.0, 1.0);

    // Initialize weights and biases
    for (size_t i = 1; i < layers.size(); ++i) {
        int prev = layers[i - 1];
        int curr = layers[i];

        // Biases
        std::vector<double> b(curr);
        for (double& val : b) val = dist(gen);
        biases_.push_back(b);

        // Weights
        std::vector<std::vector<double>> w(prev, std::vector<double>(curr));
        for (auto& row : w)
            for (double& val : row) val = dist(gen);
        weights_.push_back(w);
    }
}

std::vector<double> Network::forward(const std::vector<double>& input) {
    activations_.clear();
    activations_.push_back(input);

    std::vector<double> current = input;

    for (size_t l = 0; l < weights_.size(); ++l) {
        const auto& w = weights_[l];
        const auto& b = biases_[l];
        std::vector<double> next(b.size(), 0.0);

        for (size_t j = 0; j < next.size(); ++j) {
            for (size_t i = 0; i < current.size(); ++i) {
                next[j] += current[i] * w[i][j];
            }
            next[j] += b[j];
            next[j] = sigmoid_scalar(next[j]);
        }

        activations_.push_back(next);
        current = next;
    }

    return current;
}

void Network::train(const std::vector<double>& input, const std::vector<double>& target, double learning_rate) {
    assert(target.size() == layer_sizes_.back());

    std::vector<double> output = forward(input);

    // Compute output error (delta)
    std::vector<std::vector<double>> deltas(weights_.size());
    std::vector<double> delta(layer_sizes_.back());

    for (size_t i = 0; i < delta.size(); ++i) {
        double a = activations_.back()[i];
        delta[i] = (a - target[i]) * a * (1 - a);  // sigmoid derivative
    }
    deltas.back() = delta;

    // Backpropagate errors
    for (int l = (int)weights_.size() - 2; l >= 0; --l) {
        const auto& next_w = weights_[l + 1];
        const auto& next_delta = deltas[l + 1];
        std::vector<double> curr_delta(layer_sizes_[l + 1]);

        for (int i = 0; i < layer_sizes_[l + 1]; ++i) {
            double sum = 0.0;
            for (int j = 0; j < layer_sizes_[l + 2]; ++j) {
                sum += next_w[i][j] * next_delta[j];
            }
            double a = activations_[l + 1][i];
            curr_delta[i] = sum * a * (1 - a);
        }

        deltas[l] = curr_delta;
    }

    // Update weights and biases
    for (size_t l = 0; l < weights_.size(); ++l) {
        auto& w = weights_[l];
        auto& b = biases_[l];
        const auto& delta = deltas[l];
        const auto& a = activations_[l];

        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < delta.size(); ++j) {
                w[i][j] -= learning_rate * a[i] * delta[j];
            }
        }

        for (size_t j = 0; j < delta.size(); ++j) {
            b[j] -= learning_rate * delta[j];
        }
    }
}

    void nn::Network::save(const std::string& filepath) const {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for saving: " + filepath);
    }

    // Save layer sizes
    int size = static_cast<int>(layer_sizes_.size());
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
