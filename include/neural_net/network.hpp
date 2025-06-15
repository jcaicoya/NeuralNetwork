#ifndef NEURAL_NET_NETWORK_HPP
#define NEURAL_NET_NETWORK_HPP

#include <vector>
#include <string>

namespace nn {

    enum class ActivationType { Sigmoid, Tanh, ReLU, LeakyReLU };

    class Network {
    public:
        Network(std::vector<int> layers, ActivationType activation);

        std::vector<double> forward(const std::vector<double>& input);
        void train(const std::vector<double>& input, const std::vector<double>& target, double learning_rate);

        void save(const std::string& filepath) const;
        void load(const std::string& filepath);

    private:
        std::vector<int> layer_sizes_;
        std::vector<std::vector<double>> biases_;               // biases[layer][neuron]
        std::vector<std::vector<std::vector<double>>> weights_; // weights[layer][from][to]

        ActivationType activation_type_;

        double (*activation_)(double x);
        double (*activation_derivative_)(double x);
    };

} // namespace nn

#endif
