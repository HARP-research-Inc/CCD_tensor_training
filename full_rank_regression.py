import torch
from torch import nn
import numpy as np
import torch.optim as optim
from util import FullRankTensorRegression


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

    
    s_o_tensor = torch.zeros((int(0.8*num_nouns), 2, embedding_dim))
    test_s_o_tensor = torch.zeros((int(0.2*num_nouns), 2, embedding_dim))
    
    ground_truth = torch.zeros((int(0.8*num_nouns), embedding_dim))
    ground_truth_test = torch.zeros((int(0.2*num_nouns), embedding_dim))

    print(len(t))
    print(len(s_o))
    
    noun_pairs_test = s_o[:int(0.2*num_nouns)]
    sentence_test = t[:int(0.2*num_nouns)]

    # 
    verb1_noun_pairs = s_o[int(0.2*num_nouns):num_nouns]
    verb1_sentences = t[int(0.2*num_nouns):num_nouns]

    print("shape of verb1_sentences: ", verb1_sentences[0].shape)

    print(len(verb1_sentences))
    print(ground_truth.shape)

    #assembling training tensors
    for i, noun_tup in enumerate(verb1_noun_pairs):
        s_o_tensor[i][0] = noun_tup[0]
        s_o_tensor[i][1] = noun_tup[1]
        ground_truth[i] = torch.Tensor(verb1_sentences[i])
    
    #assembling test tensors
    for i, noun_tup in enumerate(noun_pairs_test):
        test_s_o_tensor[i][0] = noun_tup[0]
        test_s_o_tensor[i][1] = noun_tup[1]
        ground_truth_test[i] = torch.Tensor(sentence_test[i])
    
    model = FullRankTensorRegression(embedding_dim, embedding_dim)
    optimizer = optim.Adadelta(model.parameters(), lr=1) 

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
    
    print(f'Final In-Sample Loss: {loss.item():.20f}')

    """************************testing************************"""

    subjects = test_s_o_tensor[:, 0, :]
    objects = test_s_o_tensor[:, 1, :]    
    predicted_test = model(subjects, objects)
    loss = torch.mean((predicted_test - ground_truth_test)**2)

    print(f'Test Sample Loss: {loss.item():.4f}')   

    torch.save(model.state_dict(), model_destination)

def three_word_regression():
    #May or may not be implemented
    pass

if __name__ == "__main__":

    t = torch.load("data/hybrid_empirical_embeddings.pt", weights_only=False)
    s_o = torch.load("data/hybrid_dependent_data.pt", weights_only=False) #list of tuples of tensors

    two_word_regression("data/hybrid_weights.pt", s_o, t)

    

    


