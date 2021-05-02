import torch

x = torch.rand(5, 3)
print(x)
print(f'Cuda available{torch.cuda.is_available()}')
print(f'Torch zeros cuda{torch.zeros(1).cuda()}')
