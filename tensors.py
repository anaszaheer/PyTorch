import torch
import numpy as np

# data = [[1,2], [3, 4]]
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
print(tensor, "\n")
tensor.add_(5)
print(tensor)
