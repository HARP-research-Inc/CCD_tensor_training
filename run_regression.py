from regression import two_word_regression
import torch

def noun_adjective_pair_regression(destination):
    dependent_data = torch.load("data/adj_dependent_data.pt", weights_only=False)

    empirical_data = torch.load("data/adj_empirical_embeddings.pt", weights_only=False)
    
    two_word_regression(destination, 
                        dependent_data, empirical_data, num_epochs=1)

def transitive_verb_regression(destination):
    t = torch.load("data/hybrid_empirical_embeddings.pt", weights_only=False)
    s_o = torch.load("data/hybrid_dependent_data.pt", weights_only=False) # List of tuples of tensors

    two_word_regression(destination, s_o, t)

if __name__ == "__main__":
    noun_adjective_pair_regression("data/adj_weights_one_epoch.pt")
    transitive_verb_regression("data/hybrid_weights.pt")