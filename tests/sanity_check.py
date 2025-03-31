import torch

import os
import sys
sys.path.insert(0, os.path.abspath(".."))
from regression import two_word_regression, three_word_regression, k_word_regression

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
    k_word_regression("../data/dummy_model.pt", embedding_set, ground_truth, 2, word_dim=300, sentence_dim=300)

    # Load the saved model and verify it works
    model = torch.load("../data/dummy_model.pt")
    print(f"Sanity check completed. Model saved as 'dummy_model.pt'.")

def sanity_check_three():
    noun_dim = 100
    sentence_dim = 300

    nounA1 = torch.rand(noun_dim)
    nounA2 = torch.rand(noun_dim)
    nounA3 = torch.rand(noun_dim)
    targetA = torch.rand(sentence_dim)

    nounB1 = torch.rand(noun_dim)
    nounB2 = torch.rand(noun_dim)
    nounB3 = torch.rand(noun_dim)
    targetB = torch.rand(sentence_dim)

    # Prepare the dataset as required by three_word_regression
    embedding_set = [(nounA1, nounA2, nounA3), (nounB1, nounB2, nounB3)]
    ground_truth = [targetA, targetB]

    # Run two_word_regression and save the model as "dummy_model.pt"
    three_word_regression("../models/dummy_model.pt", embedding_set, ground_truth, num_epochs=100)

    # Load the saved model and verify it works
    model = torch.load("../data/dummy_model.pt")
    print(f"Sanity check completed. Model saved as 'dummy_model.pt'.")

if __name__ == "__main__":
    sanity_check()
