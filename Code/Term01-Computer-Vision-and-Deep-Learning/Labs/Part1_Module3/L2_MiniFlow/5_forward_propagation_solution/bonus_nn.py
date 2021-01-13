"""
Bonus Challenges!

These are ungraded challenges as they are more of a test of your python skills than neural network skills.

    1. Can you make Add accept any number of inputs? Eg. Add(x, y, z).
    2. Can you make a Mul class that multiplies n inputs?


No need to change anything here!

If all goes well, this should work after you
modify the Add class in miniflow.py.
"""

from miniflow import *

x, y, z = Input(), Input(), Input()

f = Add(x, y, z)

feed_dict = {x: 4, y: 5, z: 10}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

# should output 19
print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))


# Multiply

x, y, z = Input(), Input(), Input()

f = Mul(x, y, z)

feed_dict = {x: 4, y: 5, z: 10}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

# should output 200
print("{} * {} * {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))