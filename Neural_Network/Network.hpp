#ifndef _NETWORK_HPP_
#define _NETWORK_HPP_

#include <vector>
#include <cmath>
#include "Layer.hpp"

using namespace std;

class Network
{
    public:
    vector<Layer> ls;
    acty activation;
    double learning_rate;
    
    Network(vector<int> Layer_size , double learning_rate = 0.01 ,  acty activation = SIGMOID)
    {
        this-> activation = activation;
        this-> learning_rate = learning_rate;
        for (int i = 1 ; i< Layer_size.size() ; i++)
        {
            ls.emplace_back(Layer(Layer_size[i] , Layer_size[i-1] , activation));
        }
    }

    vector<double> forward_pass(vector<double> inputs)
    {
        vector<double> outputs = inputs;
        for (auto &l: ls)
        {
            outputs = l.forward_pass(outputs);
        }
        return outputs;
    }

    void backpropagation(const vector<double>& inputs, const vector<double>& target) 
    {
        vector<double> outputs = forward_pass(inputs);

        vector<double> err(outputs.size());
        for (int i = 0; i < outputs.size(); ++i) 
        {
            double e = outputs[i] - target[i];
            err[i] = e * dact_dx(outputs[i]);
        }

        for (int i = int(ls.size()) - 1; i >= 0; --i) 
        {
            Layer& l = ls[i];
            vector<double> lin;
            if (i==0)
            {
                lin = inputs;
            }
            else
            {
                lin = ls[i-1].get_outputs();
            }

            for (int j = 0; j < l.ns.size(); ++j) 
            {
                Neuron& neuron = l.ns[j];
                double delta = err[j];
                for (int k = 0; k < neuron.w.size(); ++k) {
                    neuron.w[k] -= learning_rate * delta * lin[k];
                }
                neuron.b -= learning_rate * delta;
            }

            if (i > 0) 
            {
                vector<double> new_err(lin.size(), 0.0);
                for (int j = 0; j < lin.size(); ++j) 
                {
                    double sum = 0.0;
                    for (int k = 0; k < l.ns.size(); ++k) 
                    {
                        sum += l.ns[k].w[j] * err[k];
                    }
                    new_err[j] = sum * dact_dx(lin[j]);
                }
                err = move(new_err);
            }
        }
    }




    private:
    double dact_dx(double x)
    {
        switch (activation)
        {
            case SIGMOID: 
            return x * (1 - x); 
            case TANH: 
            return 1 - x * x; 
            case RELU: 
            return (x > 0) ? 1 : 0; 
            case LEAKY_RELU: 
            return (x > 0) ? 1 : 0.01; 
            case LINEAR: 
            return 1; 
            default: 
            return 1; 
        }
    }

};

#endif