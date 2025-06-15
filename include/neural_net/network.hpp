#ifndef NEURAL_NET_NETWORK_HPP
#define NEURAL_NET_NETWORK_HPP

#include <vector>

namespace nn {

class Network {
public:
    Network(std::vector<int> layers);
    std::vector<double> forward(const std::vector<double>& input);

private:
    std::vector<int> layer_sizes_;
};

} // namespace nn

#endif // NEURAL_NET_NETWORK_HPP
