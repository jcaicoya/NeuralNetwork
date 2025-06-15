#include "neural_net/network.hpp"

namespace nn {

Network::Network(std::vector<int> layers) : layer_sizes_(std::move(layers)) {
    // Initialize weights and biases here
}

std::vector<double> Network::forward(const std::vector<double>& input) {
    // Placeholder forward propagation
    return input;
}

}
