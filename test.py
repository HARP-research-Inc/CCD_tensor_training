import json
from run_regression import two_word_regression
import torch
import numpy as np
import random

t = torch.arange(5)
ls = list(range(5)) 

seed_value = 111
random.seed(seed_value)
np.random.seed(seed_value)

# Create a list of indices and shuffle them
indices = list(range(len(ls)))
random.seed(seed_value)
np.random.shuffle(indices)

# Apply the shuffled indices to both the tensor and the list
t = t[torch.tensor(indices)]
ls = [ls[i] for i in indices]

print(t)
print(ls)
