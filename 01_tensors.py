import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f'tensor {x_data}')

x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f'Ones Tensor: \n {x_ones} \n')

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f'Random Tensor: \n {x_rand} \n')

# create a tensor of shape
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f'Rand Tensor: \n {rand_tensor} \n')
print(f'Ones Tensor: \n {ones_tensor} \n')
print(f'Zeros Tensor: \n {zeros_tensor} \n')

tensor = torch.rand(3, 4)
print(f'Shape of tensor: {tensor.shape}')
print(f'Datatype of tensor: {tensor.dtype}')
print(f'Device tensor is stored on device: {tensor.device} \n')

# numpy-like indexing and slicing:
tensor = torch.ones(4, 4)
print(tensor)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column: ', tensor[..., -1])
tensor[:, 1] = 0
print(tensor)

# joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print('Joined tensors horizontally')
print(t1)

# Arithmetic operaitons
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

# this computes the element-wise product. z1, z2, z3 will have the same value
z0 = tensor @ tensor
print(f'multiplied using @ {z0}')
z1 = tensor * tensor
print(f'multiplied using * {z1}')
z2 = tensor.mul(tensor)
print(f'multiplied using tensor.mul {z2} \n')
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# sum of elements in tensor matrix
agg = tensor.sum()
print(f'tensor.sum {agg} of tensor {tensor}')
agg_item = agg.item()
print(f' sum item {agg_item} of type {type(agg_item)} \n')

# functions that store values in-place are prefixed with _
# meaning actual value becomes changed
print(tensor, "\n")
tensor.add_(5)
print(f'added 5 in-place to tensor {tensor} \n')

# Tensors on CPU and NumPy arrays can share underlying memory, changing one will change the other
t = torch.ones(5)
print(f't torch tensor: {t}')
n = t.numpy()
print(f'n numpy tensor: {n} \n')

# change in tensor reflects numpy array
t.add_(1)
print(f'added 1 in place to t, torch changed: {t}')
print(f'added 1 in place to n, numpy changed: {n} \n')

# NumPy to Tensor
print('Changes in NumPy array reflects in the tensor')
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f't: {t}')
print(f'n: {n} \n')

