import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# get device fpr training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


# define neural network by subclassing nn.Module and initalise
# the neural network layers in __init_
# every nn.Module subclass implements operations on input data in
# the forward method
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# move neural network to a device
model = NeuralNetwork().to(device)
print(f'model: {model} \n')

# to use model we pass input data
# calling the model on the input returns a 10-dimensional tensor with
# raw predicted values for each class.
# we get the prediction probabilities by passing it through an
# instance of the nn.Softmax module
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f'Predicted class: {y_pred} \n')

# lets break down the layers
input_image = torch.rand(3, 28, 28)
print(f'image size: {input_image.size()} \n')

# nn.Flatten
# initialise nn.Flatten layer to convert each 2D 28x28 image into
# contiguous array of 784 pixel values
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(f'flattened image: {flat_image} \n')

# nn.Linear
# the linear layer is a module that applies a linear
# transformation on the input using its stored
# weights and biases
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(f'hidden: {hidden1.size()} \n')

# nn.ReLU
# non-linear activations are what create complex mappings between
# the model inputs and outputs
# they are applied after linear transformations to introduce non-linearity
# helping neural networks learn a wide variety of phenomena
# here we use nnReLU, but there's other activations to introduce non-linearity
print(f'Before ReLU: {hidden1} \n')
hidden1 = nn.ReLU()(hidden1)
print(f'After ReLU: {hidden1} \n')

# nn.Sequential
# sequential is an ordered container of modules. the data is passed through
# all the modules in the same order as defined. you can use sequential
# containers to put together a quick network like seq_modules
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

# nn.Softmax
# the last linear layer of the neural network returns lgits - raw values
# in [-infinity, infinity] - wihich are passed to the nn.Softmax module
# the logits are scaled to values [0, 1] representing the model's predicted
# probabilities for each class. dim parameter indicates the dimension along
# which the values must sum to 1
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print(f'predicted probability {pred_probab}')

# Model parameters
# many layers inside a neural network are parameterized i.e.
# have associated weights and biases that are optimized during training.
# Subclassing nn.Module automatically tracks all fields defined in your
# model object, and makes all parameters accessible using your model's
# parameters() or named_parameters() methods.
# here we iterate over each parameter and print its size and preview values
print(f'Model structure: {model}')
for name, param in model.named_parameters():
    print(f'Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n')
