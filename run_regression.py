from regression import FullRankTensorRegression, k_word_regression, two_word_regression
import torch
import json

def update_version_tracking_json():
    pass

def noun_adjective_pair_regression(destination, epochs = 100):
    """
    
    """
    dependent_data = torch.load("data/adj_dependent_data.pt", weights_only=False)

    empirical_data = torch.load("data/adj_empirical_embeddings.pt", weights_only=False)
    
    module = FullRankTensorRegression(300, 300)
    k_word_regression(destination, dependent_data, empirical_data, 2, module, word_dim=300, sentence_dim=300, num_epochs=epochs, shuffle=True)
    

def transitive_verb_regression(destination, epochs):
    t = torch.load("data/hybrid_empirical_embeddings.pt", weights_only=False)
    s_o = torch.load("data/hybrid_dependent_data.pt", weights_only=False) # List of tuples of tensors

    module = FullRankTensorRegression(300, 300)
    k_word_regression(destination, s_o, t, 2, module, word_dim=300, sentence_dim=300, num_epochs=epochs, shuffle=True)

def concatenated_three_word_regression(destination, epochs):
    pass

if __name__ == "__main__":
    noun_adjective_pair_regression("models/adj_weights.pt", epochs=5)
    #transitive_verb_regression("data/hybrid_weights.pt", epochs=10)

    print("Regression complete.")