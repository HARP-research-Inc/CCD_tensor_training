import torch
from torch import nn
import torch.optim as optim
from util import FullRankTensorRegression
from full_rank_regression import two_word_regression

def sanity_check():
    # Define a tiny dataset
    embedding_dim = 300
    subject1 = torch.rand(embedding_dim)
    object1 = torch.rand(embedding_dim)
    target1 = torch.rand(embedding_dim)

    subject2 = torch.rand(embedding_dim)
    object2 = torch.rand(embedding_dim)
    target2 = torch.rand(embedding_dim)

    # Prepare the dataset as required by two_word_regression
    embedding_set = [(subject1, object1), (subject2, object2)]
    ground_truth = [target1, target2]

    # Run two_word_regression and save the model as "dummy_model.pt"
    two_word_regression("dummy_model.pt", embedding_set, ground_truth, num_epochs=100, embedding_dim=embedding_dim)

    # Load the saved model and verify it works
    model = torch.load("dummy_model.pt")
    print(f"Sanity check completed. Model saved as 'dummy_model.pt'.")

if __name__ == "__main__":
    sanity_check()
