import torch
from util import get_embedding_in_parallel

# if torch.cuda.is_available():
#     device = torch.device('cuda')
#     print('GPU is available')
# else:
#     device = torch.device('cpu')
#     print('GPU is not available, using CPU instead')

# print(f'Using device: {device}')

# num_gpus = torch.cuda.device_count()
# print(f'Number of GPUs available: {num_gpus}')

print(get_embedding_in_parallel("test"))