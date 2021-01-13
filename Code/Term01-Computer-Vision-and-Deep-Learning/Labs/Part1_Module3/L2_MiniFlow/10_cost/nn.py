"""
Test your MSE method with this script!

No changes necessary, but feel free to play
with this script to test your network.
"""

import numpy as np
from miniflow import Input
from miniflow import Linear
from miniflow import Sigmoid
from miniflow import topological_sort
from miniflow import forward_pass
from miniflow import MSE

inputs, weights, bias = Input(), Input(), Input()

f = Linear(inputs, weights, bias)
g = Sigmoid(f)

x = np.array([[-1., -2.], [-1, -2]])
w = np.array([[2., -3], [2., -3]])
b = np.array([-3., -5])

feed_dict = {inputs: x, weights: w, bias: b}

graph = topological_sort(feed_dict)
output = forward_pass(g, graph)

ideal_output = np.array(
    [[1.23394576e-04, 9.82013790e-01],
     [1.23394576e-04, 9.82013790e-01]])

cost = MSE(output, ideal_output, 1) # there's only 1 input

"""
Output should be on the order of 1e-22.
"""
print(cost)