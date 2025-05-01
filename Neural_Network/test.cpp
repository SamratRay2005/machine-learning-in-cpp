#include <iostream>
#include <vector>
#include <iomanip>    // for std::setprecision
#include <random>
#include <algorithm>
#include "Network.hpp"

int main() {
    // Seed for reproducibility
    std::mt19937 rng(123);

    // Create a network: 2 inputs —> 1 output (no hidden layer), learning rate 0.1
    Network nn({2, 1}, /*learning_rate=*/0.1, SIGMOID);

    // AND gate dataset (linearly separable)
    std::vector<std::vector<double>> inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    std::vector<std::vector<double>> targets = {
        {0.0},  // 0 AND 0 = 0
        {0.0},  // 0 AND 1 = 0
        {0.0},  // 1 AND 0 = 0
        {1.0}   // 1 AND 1 = 1
    };

    const int epochs = 5000;
    std::vector<int> order(inputs.size());
    std::iota(order.begin(), order.end(), 0);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        // Shuffle training order each epoch
        std::shuffle(order.begin(), order.end(), rng);

        double total_loss = 0.0;
        int correct = 0;

        for (int idx : order) {
            auto x = inputs[idx];
            auto t = targets[idx];

            // Train
            nn.backpropagation(x, t);
            double out = nn.forward_pass(x)[0];

            double err = t[0] - out;
            total_loss += err * err;

            if ((out >= 0.5 && t[0] == 1.0) || (out < 0.5 && t[0] == 0.0)) {
                ++correct;
            }
        }

        if (epoch % 500 == 0) {
            double accuracy = 100.0 * correct / inputs.size();
            std::cout << "Epoch " << std::setw(4) << epoch
                      << " | Loss: " << std::fixed << std::setprecision(6) << total_loss
                      << " | Acc: " << std::setprecision(1) << accuracy << "%\n";
        }
    }

    // Final evaluation on AND outputs
    std::cout << "\nAND gate results after training:\n";
    for (size_t i = 0; i < inputs.size(); ++i) {
        double out = nn.forward_pass(inputs[i])[0];
        std::cout << "[" << inputs[i][0] << ", " << inputs[i][1] << "] -> "
                  << std::fixed << std::setprecision(4) << out
                  << " (" << (out >= 0.5 ? 1 : 0) << ")\n";
    }

    return 0;
}
