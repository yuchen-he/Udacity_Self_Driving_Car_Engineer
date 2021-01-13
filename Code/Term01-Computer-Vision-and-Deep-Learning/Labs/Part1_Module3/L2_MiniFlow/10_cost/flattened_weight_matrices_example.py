import numpy as np

w1 = np.array([[1, 2], [3, 4]])
w2 = np.array([[5, 6], [7, 8]])

w1_flat = np.reshape(w1, -1)
w2_flat = np.reshape(w2, -1)

w = np.concatenate([w1_flat, w2_flat])

print(w)