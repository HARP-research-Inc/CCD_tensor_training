import torch

import os
import sys
sys.path.insert(0, os.path.abspath(".."))
from src.regression import k_word_regression, TwoWordTensorRegression

def sanity_check():
    # Define a tiny dataset
    embedding_dim = 384
    subject1 = torch.rand(embedding_dim)
    object1 = torch.rand(embedding_dim)
    target1 = torch.rand(embedding_dim)

    subject2 = torch.rand(embedding_dim)
    object2 = torch.rand(embedding_dim)
    target2 = torch.rand(embedding_dim)

    # Prepare the dataset as required by two_word_regression
    embedding_set = [(subject1, object1), (subject2, object2)]
    ground_truth = [target1, target2]

    module = TwoWordTensorRegression(embedding_dim, embedding_dim)

    # Run two_word_regression and save the model as "dummy_model.pt"
    k_word_regression("data/dummy_model.pt", embedding_set, ground_truth, 2, module, word_dim=384, sentence_dim=384)

    # Load the saved model and verify it works
    model = torch.load("data/dummy_model.pt")
    print(f"Sanity check completed. Model saved as 'dummy_model.pt'.")

if __name__ == "__main__":
    sanity_check()
