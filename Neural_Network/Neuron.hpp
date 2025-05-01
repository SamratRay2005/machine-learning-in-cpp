#ifndef _NEURON_HPP_
#define _NEURON_HPP_

#include <iostream> 
#include <vector>
#include <numeric>
#include <cmath>
#include <random>

using namespace std;

enum acty
{
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU,
    LINEAR
};

// each neuron -> f(wx + b)  f->taken from user
class Neuron
{
public:
    vector<double> w;
    double b;
    acty activation;

    Neuron(int n, acty act = SIGMOID)
    {
        w.resize(n);
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> d(0, 1);

        for (auto& weight : w) {
            weight = d(gen);
        }

        b = 0.0;
        activation = act;
    }

    double sigmoid_cal(double x)
    {
        return 1.0 / (1 + exp(-x));
    }
    double tanh_cal(double x)
    {
        return tanh(x);
    }
    double relu_cal(double x)
    {
        if (x>0) return x;
        else return 0;
    }
    double leaky_relu_cal(double x)
    {
        if (x>0) return x;
        else return (0.01*x);
    }
    double linear_cal(double x)
    {
        return x;
    }
    double activate(double x)
    {
        switch (activation)
        {
            case SIGMOID: 
            return sigmoid_cal(x);
            case TANH: 
            return tanh_cal(x);
            case RELU: 
            return relu_cal(x);
            case LEAKY_RELU: 
            return leaky_relu_cal(x);
            case LINEAR: 
            return linear_cal(x);
            default: 
            return x;
        }
    }

    double forward_pass(vector<double>& inputs)
    {
        double sum = inner_product(w.begin(), w.end(), inputs.begin(), 0.0) + b;
        sum = activate(sum);
        return sum;
    }
};

#endif