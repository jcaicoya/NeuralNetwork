#ifndef NEURAL_NET_NETWORK_HPP
#define NEURAL_NET_NETWORK_HPP

#include <vector>
#include <string>

namespace nn {

    class Network {
    public:
        explicit Network(const std::vector<int>& layers);

        // Forward pass
        std::vector<double> forward(const std::vector<double>& input);

        // Training (backpropagation + gradient descent)
        void train(const std::vector<double>& input, const std::vector<double>& target, double learning_rate);

        void save(const std::string& filepath) const;
        void load(const std::string& filepath);

    private:
        std::vector<int> layer_sizes_;

        std::vector<std::vector<double>> activations_;  // per layer
        std::vector<std::vector<double>> biases_;       // biases[layer][neuron]
        std::vector<std::vector<std::vector<double>>> weights_; // weights[layer][from][to]

        std::vector<double> sigmoid(const std::vector<double>& v) const;
        std::vector<double> sigmoid_derivative(const std::vector<double>& v) const;

        std::vector<double> dot(const std::vector<double>& vec, const std::vector<std::vector<double>>& mat, int col_count) const;
    };

} // namespace nn

#endif
