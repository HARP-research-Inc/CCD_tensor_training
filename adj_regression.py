from full_rank_regression import two_word_regression
from util import get_embedding_in_parallel
import torch

if __name__ == "__main__":
    dependent_data = torch.load("adj_dependent_data.pt")
    
    