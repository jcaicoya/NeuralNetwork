#include <gtest/gtest.h>
#include "neural_net/network.hpp"

TEST(NetworkTest, ForwardPassIdentity) {
    nn::Network net({3, 3, 3});
    std::vector<double> input = {1.0, 2.0, 3.0};
    std::vector<double> result = net.forward(input);
    EXPECT_EQ(result, input);
}
