# https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

import torch
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT) #loading a pretrained resnet18 model
data = torch.rand(1, 3, 64, 64) #creating random image data (no of images, no of color channels, height, width)
labels = torch.rand(1, 1000) #creating lables (no of images, no of ImageNet classes)

prediction = model(data) # forward pass

loss = (prediction - labels).sum()
loss.backward() # backward pass

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

optim.step() #gradient descent