#include <gtest/gtest.h>
#include "neural_net/network.hpp"
#include <cmath>
#include <fstream>

// Simple helper to check closeness
bool almost_equal(double a, double b, double epsilon) {
    return std::abs(a - b) < epsilon;
}

TEST(NetworkTest, ForwardConsistency) {
    nn::Network net({2, 2, 1}, nn::ActivationType::Sigmoid);

    // Run forward twice and check outputs are the same
    auto out1 = net.forward({1.0, 0.0});
    auto out2 = net.forward({1.0, 0.0});

    ASSERT_EQ(out1.size(), out2.size());
    for (size_t i = 0; i < out1.size(); ++i) {
        constexpr double epsilon = 1e-6;
        EXPECT_TRUE(almost_equal(out1[i], out2[i], epsilon));
    }
}

TEST(NetworkTest, TrainingXORReducesError) {
    nn::Network net({2, 2, 1}, nn::ActivationType::Sigmoid);
    std::vector<std::vector<double>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    std::vector<std::vector<double>> targets = {
        {0}, {1}, {1}, {0}
    };

    double before = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto out = net.forward(inputs[i]);
        before += std::pow(out[0] - targets[i][0], 2);
    }

    for (int epoch = 0; epoch < 5000; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            net.train(inputs[i], targets[i], 0.1);
        }
    }

    double after = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto out = net.forward(inputs[i]);
        after += std::pow(out[0] - targets[i][0], 2);
    }

    EXPECT_LT(after, before);
}

TEST(NetworkTest, SaveAndLoadRoundTrip) {
    nn::Network net1({2, 2, 1}, nn::ActivationType::Sigmoid);
    auto original_output = net1.forward({1.0, 1.0});

    net1.save("test_model.bin");

    nn::Network net2({}, nn::ActivationType::Sigmoid);
    net2.load("test_model.bin");
    auto loaded_output = net2.forward({1.0, 1.0});

    std::cout << "Original output: " << original_output[0] << "\n";
    std::cout << "Loaded output:   " << loaded_output[0] << "\n";

    ASSERT_EQ(original_output.size(), loaded_output.size());
    for (size_t i = 0; i < original_output.size(); ++i) {
        constexpr double epsilon = 1e-6;
        EXPECT_TRUE(almost_equal(original_output[i], loaded_output[i], epsilon));
    }

    std::remove("test_model.bin");
}

const std::vector<std::vector<double>> inputs = {
    {0, 0}, {0, 1}, {1, 0}, {1, 1}
};

const std::vector<std::vector<double>> targets = {
    {0}, {1}, {1}, {0}
};

constexpr int epochs = 20'000;
const std::vector<int> configuration = { 2, 6, 1 };
constexpr double learning_rate = 0.1;

TEST(NetworkTest, XORWithSigmoid) {
    nn::Network net(configuration, nn::ActivationType::Sigmoid);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            net.train(inputs[i], targets[i], learning_rate);
        }
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        constexpr double epsilon = 5e-2;
        auto output = net.forward(inputs[i]);
        EXPECT_TRUE(almost_equal(output[0], targets[i][0], epsilon))
            << "Failure with input [" << inputs[i][0] << ", " << inputs[i][1]
            << "]: expected " << targets[i][0]
            << ", got " << output[0];
    }
}

TEST(NetworkTest, XORWithTanh) {
    nn::Network net(configuration, nn::ActivationType::Tanh);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            net.train(inputs[i], targets[i], learning_rate);
        }
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        constexpr double epsilon = 1e-2;
        auto output = net.forward(inputs[i]);
        EXPECT_TRUE(almost_equal(output[0], targets[i][0], epsilon))
            << "Failure with input [" << inputs[i][0] << ", " << inputs[i][1]
            << "]: expected " << targets[i][0]
            << ", got " << output[0];
    }
}

TEST(NetworkTest, XORWithLeakyReLU) {
    nn::Network net(configuration, nn::ActivationType::LeakyReLU);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            net.train(inputs[i], targets[i], learning_rate);
        }
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        constexpr double epsilon = 1e-2;
        auto output = net.forward(inputs[i]);
        EXPECT_TRUE(almost_equal(output[0], targets[i][0], epsilon))
            << "Failure with input [" << inputs[i][0] << ", " << inputs[i][1]
            << "]: expected " << targets[i][0]
            << ", got " << output[0];
    }
}
