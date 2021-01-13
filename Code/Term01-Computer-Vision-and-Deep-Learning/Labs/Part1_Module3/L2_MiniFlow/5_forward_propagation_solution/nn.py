"""
This script builds and runs a graph with miniflow.

There is no need to change anything to solve this quiz!

However, feel free to play with the network! Can you also
build a network that solves the equation below?

(x + y) + y
"""

from miniflow import *

x, y = Input(), Input()

f = Add(x, y)

feed_dict = {x: 10, y: 5}

sorted_neurons = topological_sort(feed_dict)
output = forward_pass(f, sorted_neurons)

print("{} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], output))


# Cheat version to built a network off of an existing network's output.
p = Input()
r = Add(p, y)

feed_dict = {p: output, x: feed_dict[x], y: feed_dict[y]}

sorted_neurons = topological_sort(feed_dict)
output = forward_pass(r, sorted_neurons)

print("({} + {}) + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[y], output))


# Full network implementation to Add (x + y) + y

x, y = Input(), Input()

f = Add(Add(x, y), y)

feed_dict = {x: 10, y: 5}

sorted_neurons = topological_sort(feed_dict)
output = forward_pass(f, sorted_neurons)

print("({} + {}) + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[y], output))