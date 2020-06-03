from genome.binary_networks import inference
import numpy as np

layers = [64, 64, 1]
debug = inference.CompiledNetwork(layers, inference.InferenceStrategy.DEBUG)
cpu = inference.CompiledNetwork(layers, inference.InferenceStrategy.CPU64)
gpu = inference.CompiledNetwork(layers, inference.InferenceStrategy.GPU)

# weights = [
#     [np.zeros([32, 32], np.bool), np.zeros(32, np.int32)],
#     [np.zeros([32, 1], np.bool), np.zeros(1, np.int32)],
# ]
# weights[0][0][0, :] = 1
# # weights[0][0] = weights[0][0].T

weights = [
    (np.random.uniform(size=[64, 64]) > 0.5, np.zeros(64, np.int32)),
    (np.random.uniform(size=[64, 1]) > 0.5, np.zeros(1, np.int32)),
]
debug.set_params(weights)
cpu.set_params(weights)
gpu.set_params(weights)

inputs = np.random.uniform(size=64) > 0.5

print("debug:", debug.forward(inputs))
print("cpu:", cpu.forward(inputs))
print("gpu:", gpu.forward(inputs))

weights = [
    (np.random.uniform(size=[64, 64]) > 0.5, np.zeros(64, np.int32)),
    (np.random.uniform(size=[64, 1]) > 0.5, np.zeros(1, np.int32)),
]
debug.set_params(weights)
cpu.set_params(weights)
gpu.set_params(weights)

inputs = np.random.uniform(size=64) > 0.5

print("debug:", debug.forward(inputs))
print("cpu:", cpu.forward(inputs))
print("gpu:", gpu.forward(inputs))
