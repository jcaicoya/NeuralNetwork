#include "neural_net/network.hpp"
#include <iostream>

int main() {
    nn::Network net({3, 5, 2});
    auto result = net.forward({1.0, 0.5, -0.1});
    for (double val : result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    return 0;
}
