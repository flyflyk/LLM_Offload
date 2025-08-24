import torch
from accelerate import Accelerator

print(f'Before Accelerator init: {torch.cuda.is_available()}')
acc = Accelerator()
print(f'After Accelerator init: {torch.cuda.is_available()}')
