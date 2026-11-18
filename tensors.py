import torch
import numpy as np

# data = [[1,2], [3, 4]] #2D tensor
# x_data = torch.tensor(data)

tensor = torch.rand(3, 4) #2D tensor, 3 rows and 4 columns

# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(f"Device tensor is stored on: {tensor.device}")

#MOVING TENSOR TO GPU
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
tensor = tensor.to(device)
# print(f"Device tensor is stored on: {tensor.device}")

#STANDARD INDEXING AND SLICING
tensor = torch.ones(4, 4)
tensor[:,1] = 0
# print(tensor)

#JOIN MULTIPLE TENSORS
# t1 = torch.cat([tensor, tensor, tensor], dim=1) # dim=1 tells it to concatinate columns wise
# print(t1)

#MULTIPLY TENSORS: This computes the element-wise product
# print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")

#MATRIX MULTIPLICATION BETWEEN TWO TENSORS
# print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")

#IN-PLACE OPERATIONS
# tensor.add_(5) #adds 5 to all values in the tensor
# print(tensor)

#TENSOR TO NumPy ARRAY
t = torch.ones(5)
# print(f"t: {t}")
n = t.numpy()
# print(f"n: {n}")

# t.add_(1) #change in tensor reflects on the numpy as-well
# print(f"t: {t}")
# print(f"n: {n}")

#NumPy ARRAY TO TENSOR
n = np.ones(5)
t = torch.from_numpy(n)
# print(f"n: {n}")
# print(f"t: {t}")

np.add(n, 1, out=n) #change in numpy reflects on the tensor as-well
print(f"n: {n}")
print(f"t: {t}")