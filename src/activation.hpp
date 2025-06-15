#pragma once

#include <cmath>

namespace nn {

    struct Activation {

        static double sigmoid(double x) {
            return 1.0 / (1.0 + std::exp(-x));
        }

        static double sigmoid_derivative(double x) {
            double s = sigmoid(x);
            return s * (1 - s);
        }

        static double tanh(double x) {
            return std::tanh(x);
        }

        static double tanh_derivative(double x) {
            double t = tanh(x);
            return 1 - t * t;
        }

        static double relu(double x) {
            return x > 0 ? x : 0.0;
        }

        static double relu_derivative(double x) {
            return x > 0 ? 1.0 : 0.0;
        }

        static double leakey_relu(double x) {
            return x > 0 ? x : leakey_alpha * x;
        }

        static double leakey_relu_derivative(double x) {
            return x > 0 ? 1.0 : leakey_alpha;
        }

    private:
        static constexpr double leakey_alpha = 0.01;
    };

}  // namespace nn
