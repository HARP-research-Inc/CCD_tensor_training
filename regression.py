import torch
from torch import nn
import numpy as np
import torch.optim as optim

class FullRankTensorRegression(nn.Module):
    def __init__(self, noun_dim, sent_dim):
        super().__init__()
        self.sent_dim = sent_dim
        self.noun_dim = noun_dim

        self.V = nn.Parameter(torch.randn(sent_dim, noun_dim, noun_dim))
        self.bias = nn.Parameter(torch.zeros(sent_dim))

        # Optional: initialize slice-by-slice
        for i in range(sent_dim):
            torch.nn.init.xavier_uniform_(self.V[i])

    def forward(self, s, o):
        # Optionally normalize inputs
        # s = F.normalize(s, dim=1)
        # o = F.normalize(o, dim=1)

        Vs_o = torch.einsum('ljk,bj,bk->bl', self.V, s, o)
        return Vs_o + self.bias

def two_word_regression(model_destination, embedding_set, ground_truth, num_epochs = 50, embedding_dim = 300):
    """
    Regression for datasets of paired words.

    Args:

    Returns:
    """
    t = ground_truth
    s_o = embedding_set

    if len(t) != len(s_o):
        raise Exception("Mismatched data dimensions")

    num_nouns = len(t)
    test_size = num_nouns // 5  # 20% of the data
    train_size = num_nouns - test_size  # Remaining 80%

    # Allocating space for testing and training tensors
    print(">allocating space for testing and training tensors...")
    s_o_tensor = torch.zeros((train_size, 2, embedding_dim))
    test_s_o_tensor = torch.zeros((test_size, 2, embedding_dim))
    
    ground_truth = torch.zeros((train_size, embedding_dim))
    ground_truth_test = torch.zeros((test_size, embedding_dim))
    print(">done!\n\n\n")

    # Partitioning between testing and training sets
    print(">Partitioning between testing and training sets...")
    noun_pairs_test = s_o[:test_size]
    sentence_test = t[:test_size]

    verb1_noun_pairs = s_o[test_size:]
    verb1_sentences = t[test_size:]
    print(">done!\n\n\n")

    # Assembling training tensors
    print(">Assembling training tensors...")
    for i, noun_tup in enumerate(verb1_noun_pairs):
        s_o_tensor[i][0] = noun_tup[0]
        s_o_tensor[i][1] = noun_tup[1]
        ground_truth[i] = torch.Tensor(verb1_sentences[i])
    print(">done!\n\n\n")
    
    # Assembling test tensors
    print(">Assembling testing tensors...")
    for i, noun_tup in enumerate(noun_pairs_test):
        test_s_o_tensor[i][0] = noun_tup[0]
        test_s_o_tensor[i][1] = noun_tup[1]
        ground_truth_test[i] = torch.Tensor(sentence_test[i])
    print(">done!\n\n\n")
    
    model = FullRankTensorRegression(embedding_dim, embedding_dim)
    optimizer = optim.Adadelta(model.parameters(), lr=0.001)

    print(">Running regression...")
    subjects = s_o_tensor[:, 0, :]
    objects = s_o_tensor[:, 1, :]

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        predicted = model(subjects, objects)

        # Compute loss (Mean Squared Error)
        loss = torch.mean((predicted - ground_truth)**2)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Print loss for each epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.20f}')

        # Debugging: Check if weights are being updated
        if epoch == 0 or epoch == num_epochs - 1:
            print(f"Sample weights at epoch {epoch + 1}: {list(model.parameters())[0][0][:5]}")

    print(f'>done! Final In-Sample Loss: {loss.item():.20f}\n\n\n')

    # Save model weights
    torch.save(model.state_dict(), model_destination)
    print(f"Model weights saved to: {model_destination}")

    """************************testing************************"""
    print(">************************testing************************")
    subjects = test_s_o_tensor[:, 0, :]
    objects = test_s_o_tensor[:, 1, :]    
    predicted_test = model(subjects, objects)
    loss = torch.mean((predicted_test - ground_truth_test)**2)

    print(f'>done! Test Sample Loss: {loss.item():.4f}\n\n\n')   

    torch.save(model.state_dict(), model_destination)

def three_word_regression():
    # May or may not be implemented
    pass

if __name__ == "__main__":

    t = torch.load("data/hybrid_empirical_embeddings.pt", weights_only=False)
    s_o = torch.load("data/hybrid_dependent_data.pt", weights_only=False) # List of tuples of tensors

    two_word_regression("data/hybrid_weights.pt", s_o, t)






