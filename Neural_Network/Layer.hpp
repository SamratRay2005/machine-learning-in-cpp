#ifndef _LAYER_HPP_
#define _LAYER_HPP_

#include <vector>
#include "Neuron.hpp"

using namespace std;


class Layer
{
    public:
    vector<Neuron> ns;
    vector<double> o;
    Layer(int m , int n , acty activation = SIGMOID)
    {
        ns.reserve(m);
        for (int i = 0 ; i<m ; i++)
        {
            ns.emplace_back(Neuron(n , activation));
        }
    }
    vector<double> forward_pass(vector<double>& inputs)
    {
        o.clear();
        o.reserve(ns.size());
        for (auto &i : ns)
        {
            o.push_back(i.forward_pass(inputs));
        }
        return o;
    }
    vector<double> get_outputs()
    {
        return o;
    }
};

#endif