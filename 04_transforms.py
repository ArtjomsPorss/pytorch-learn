import torch
from torchvision import datasets
from torchvision.transforms import Lambda
from torchvision.transforms import ToTensor


# Data does not always come in its final processed form
# that is required for training machine learning algorithms.
# We use transforms to perform some manipulation of the data and
# make it suitable for training.
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float)
                            .scatter_(0, torch.tensor(y), value=1))
)


target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
