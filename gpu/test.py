import python_interface
import numpy as np

net = python_interface.CudaBinaryNetwork(1, 1, np.array([1], dtype=np.uint32))
hidden_weights = np.ones([32, 32], dtype=np.bool)
hidden_weights[0, :] = 0
net.set_hidden_params(0, hidden_weights, np.zeros([32], dtype=np.int32))
net.set_output_params(np.ones([32, 32], dtype=np.bool), np.zeros([32], dtype=np.int32))

inputs = np.ones(32, dtype=np.bool)
print("Output:", net.forward(inputs))
