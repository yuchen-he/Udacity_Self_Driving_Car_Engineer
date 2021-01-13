"""
A simple artificial neuron depends on three components:

   * input, x
   * weight, w
   * bias, b

The output, o, is just the weighted sum of the inputs plus the bias:

Remember, by varying the weights, you can vary the amount of influence any given input has on the output. The learning
aspect of neural networks takes place during a process known as backpropagation. In backpropogation, the network
modifies the weights to improve the network's output accuracy. You'll be applying all of this shortly.

For this quiz, I want you to build a linear neuron that generates an output by applying a simplified version of
Equation (1). Linear should take a list of inputs of length n, a list of weights of length n, and a bias.

Instructions:

   1. Open nn.py below (this file). Read through the neural network to see the expected output of Linear.
   2. Open miniflow.py below. Modify Linear, which is a subclass of Neuron, to generate an output with Equation (1).

(Hint: you could use numpy to solve this quiz if you'd like, but it's very possible to solve this with vanilla Python.)
"""

from miniflow import Input
from miniflow import Linear
from miniflow import topological_sort
from miniflow import forward_pass

x, y, z = Input(), Input(), Input()
inputs = [x, y, z]

weight_x, weight_y, weight_z = Input(), Input(), Input()
weights = [weight_x, weight_y, weight_z]

bias = Input()

f = Linear(inputs, weights, bias)

feed_dict = {
    x: 6,
    y: 14,
    z: 3,
    weight_x: 0.5,
    weight_y: 0.25,
    weight_z: 1.4,
    bias: 2
}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

print('Linear output: ', output)  # should be 12.7 with this example
