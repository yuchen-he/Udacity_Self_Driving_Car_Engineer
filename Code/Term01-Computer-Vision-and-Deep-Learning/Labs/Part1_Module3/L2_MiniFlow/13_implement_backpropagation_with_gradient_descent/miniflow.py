''"""
Your code goes in this file! Add the Sigmoid#backward method.
Scroll down to find it.

Make sure to check out the Linear#backward and MSE#backward
methods for inspiration.
"""

import numpy as np

class Layer:
    """
    Base class for layers in the network.

    Arguments:

        `inbound_layers`: A list of layers with edges into this layer.
    """
    def __init__(self, inbound_layers=[]):
        """
        Layer's constructor (runs when the object is instantiated). Sets
        properties that all layers need.
        """
        # A list of layers with edges into this layer.
        self.inbound_layers = inbound_layers
        # The eventual value of this layer. Set by running
        # the forward() method.
        self.value = None
        # A list of layers that this layer outputs to.
        self.outbound_layers = []
        # New property! Keys are the inputs to this layer and
        # their values are the partials of this layer with
        # respect to that input.
        self.gradients = {}
        # Sets this layer as an outbound layer for all of
        # this layer's inputs.
        for layer in inbound_layers:
            layer.outbound_layers.append(self)

    def forward(self):
        """
        Every layer that uses this class as a base class will
        need to define its own `forward` method.
        """
        raise NotImplementedError

    def backward(self):
        """
        Every layer that uses this class as a base class will
        need to define its own `backward` method.
        """
        raise NotImplementedError


class Input(Layer):
    """
    A generic input into the network.
    """
    def __init__(self):
        # The base class constructor has to run to set all
        # the properties here.
        #
        # The most important property on an Input is value.
        # self.value is set during `topological_sort` later.
        Layer.__init__(self)

    def forward(self):
        # Do nothing because nothing is calculated.
        pass

    def backward(self):
        # An Input layer has no inputs so the gradient (derivative)
        # is zero.
        # The key, `self`, is reference to this object.
        self.gradients = {self: 0}
        # Weights and bias may be inputs, so you need to sum
        # the gradient from output gradients.
        for n in self.outbound_layers:
            self.gradients[self] += n.gradients[self]


class Linear(Layer):
    """
    Represents a layer that performs a linear transform.
    """
    def __init__(self, inbound_layer, weights, bias):
        # The base class (Layer) constructor. Weights and bias
        # are treated like inbound layers.
        Layer.__init__(self, [inbound_layer, weights, bias])

    def forward(self):
        """
        Performs the math behind a linear transform.
        """
        # X
        inputs = self.inbound_layers[0].value

        # W
        weights = self.inbound_layers[1].value

        # b
        bias = self.inbound_layers[2].value

        # Z = XW + b
        self.value = np.dot(inputs, weights) + bias

    def backward(self):
        """
        Calculates the gradient based on the output values.
        """
        # Initialize a partial for each of the inbound_layers.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_layers}
        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_layers:
            # Get the partial of the cost with respect to this layer.
            grad_cost = n.gradients[self]

            """
                d_C/d_X = (d_C/d_Z) * W_T

                The derivative of C with respect to X is the dot product of the
                derivative of C with respect to Z and the transpose of the matrix W.

                https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5820f6c0_partial-x/partial-x.png
            """
            # partial derivative of X: Set the partial of the loss with respect to this layer's inputs.
            self.gradients[self.inbound_layers[0]] += np.dot(grad_cost, self.inbound_layers[1].value.T)


            """
                d_C/d_W = X_T * (d_C/d_Z)

                The derivative of the cost C with respect to W is the dot product of the transpose of the matrix X
                and the derivative of C with respect to Z.

                https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5820f6c0_partial-w/partial-w.png
            """
            # partial derivative of W: Set the partial of the loss with respect to this layer's weights.
            self.gradients[self.inbound_layers[1]] += np.dot(self.inbound_layers[0].value.T, grad_cost)


            """
                d_C/d_b_l = Σi->m (1 * d_C/d_Z_il)

                The derivative of C with respect to b at the lth element is the summation i to m of the
                derivative of C with respect to Z at the ith row, lth column.

                https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5820f6bc_partial-bl3/partial-bl3.png
            """
            # b: Set the partial of the loss with respect to this layer's bias.
            self.gradients[self.inbound_layers[2]] += np.sum(grad_cost, axis=0, keepdims=False)


class Sigmoid(Layer):
    """
    Represents a layer that performs the sigmoid activation function.
    """
    def __init__(self, layer):
        # The base class constructor.
        Layer.__init__(self, [layer])

    def _sigmoid(self, x):
        """
        This method is separate from `forward` because it
        will be used with `backward` as well.

        `x`: A numpy array-like object.
        """
        # One of one plust "e" to the negative "x"
        return 1. / (1. + np.exp(-x))

    def forward(self):
        """
        Perform the sigmoid function and set the value.
        """
        input_value = self.inbound_layers[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        """
        Calculates the gradient using the derivative of
        the sigmoid function.
        """
        # Initialize the gradients to 0.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_layers}
        """
        Your code goes here!

        Set the gradients property to the gradients with respect to each input.

        See the linear layer and MSE layer for examples.
        """
        print(self.gradients)

        """
            The derivative of the sigmoid function actually includes the original function, hence the decision to
            keep the _sigmoid method separate earlier. To make your life easier, I'll give it to you. However, if you
            want a fun calculus challenge, I recommend proving that the following equation is true. (σ denotes sigmoid.)

            σ'(x) = σ(x) * (1 - σ(x))

            Equation (4)
        """
        sig = (self.value * (1. - self.value))

        for n in self.outbound_layers:
            grad_cost = n.gradients[self]
            self.gradients[self.inbound_layers[0]] += sig*grad_cost


class MSE(Layer):
    def __init__(self, inbound_layer):
        """
        The mean squared error cost function.
        Should be used as the last layer for a network.

        Arguments:
            `inbound_layer`: A layer with an activation function.
        """
        # Call the base class' constructor.
        Layer.__init__(self, [inbound_layer])
        """
        These two properties are set during topological_sort()
        """
        # The ideal_output for forward().
        self.ideal_output = None
        # The number of inputs for forward().
        self.n_inputs = None

    def forward(self):
        """
        Calculates the mean squared error.
        """
        # Save the computed output for backward.
        self.computed_output = self.inbound_layers[0].value
        first_term = 1. / (2. * self.n_inputs)
        norm = np.linalg.norm(self.ideal_output - self.computed_output)
        self.value = first_term * np.square(norm)

    def backward(self):
        """
        Calculates the gradient of the cost.
        """

        """
            d_C/d_x = -2(y(x) - a)

            Equation (15)

            https://d17h27t6h515a5.cloudfront.net/topher/2016/October/58121a71_codecogseqn-26/codecogseqn-26.gif
        """

        self.gradients[self.inbound_layers[0]] = -2 * (self.ideal_output - self.computed_output)


def topological_sort(feed_dict, ideal_output):
    """
    Sort the layers in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Layer and the value is the respective value feed to that Layer.
    `ideal_output`: The correct output value for the last activation layer.

    Returns a list of sorted layers.
    """

    input_layers = [n for n in feed_dict.keys()]

    G = {}
    layers = [n for n in input_layers]
    while len(layers) > 0:
        n = layers.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_layers:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            layers.append(m)

    L = []
    S = set(input_layers)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]
        if isinstance(n, MSE):
            n.ideal_output = ideal_output
            # there is only 1 input in this example
            n.n_inputs = 1

        L.append(n)
        for m in n.outbound_layers:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_and_backward(feed_dict, ideal_output, trainables=[]):
    """
    Performs a forward pass and a backward pass through a list of sorted Layers.

    Returns a list of the gradients on the trainables.

    Arguments:

        `feed_dict`: A dictionary where the key is a `Input` Layer and the value is the respective value feed to that Layer.
        `ideal_output`: The correct output value for the last activation layer.
        `trainables`: Inputs that need to be modified by gradient descent.
    """

    sorted_layers = topological_sort(feed_dict, ideal_output)

    # Forward pass
    for n in sorted_layers:
        n.forward()

    # Backward pass
    reversed_layers = sorted_layers[::-1] # see: https://docs.python.org/2.3/whatsnew/section-slices.html

    for n in reversed_layers:
        n.backward()

    # Returns a list of the gradients on the weights and bias (the trainables).
    return [n.gradients[n] for n in trainables]