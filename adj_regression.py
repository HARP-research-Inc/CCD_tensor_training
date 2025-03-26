from full_rank_regression import two_word_regression
from util import get_embedding_in_parallel
import torch

if __name__ == "__main__":
    dependent_data = torch.load("data/adj_dependent_data.pt", weights_only=False)

    empirical_data = torch.load("data/adj_empirical_embeddings.pt", weights_only=False)
    
    two_word_regression("data/adj_weights.pt", 
                        dependent_data, empirical_data, num_epochs=25)