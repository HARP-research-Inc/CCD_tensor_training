import torch
from torch import nn
import torch.optim as optim
import random

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

class TwoWordTensorRegression(nn.Module):
    def __init__(self, noun_dim, sent_dim):
        """
        Regression initialization
        """
        super().__init__()
        self.sent_dim = sent_dim
        self.noun_dim = noun_dim

        self.V = nn.Parameter(torch.randn(sent_dim, noun_dim, noun_dim))
        self.bias = nn.Parameter(torch.zeros(sent_dim))

        # Optional: initialize slice-by-slice
        for i in range(sent_dim):
            torch.nn.init.xavier_uniform_(self.V[i])

    def forward(self, s, o):
        """
        Optionally normalize inputs
        
        Args:
            s = F.normalize(s, dim=1)
            o = F.normalize(o, dim=1)
        
        Returns:

        """

        Vs_o = torch.einsum('ljk,bj,bk->bl', self.V, s, o)
        return Vs_o + self.bias

class ThreeWordTensorRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Regression initialization for mapping three input vectors to one output vector.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Tensor for mapping three input vectors to one output vector
        self.V = nn.Parameter(torch.randn(output_dim, input_dim, input_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))

        # Initialize slice-by-slice
        for i in range(output_dim):
            torch.nn.init.xavier_uniform_(self.V[i])

    def forward(self, x1, x2, x3):
        """
        Forward pass for three input vectors.

        Args:
            x1, x2, x3: Input tensors of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        Vx1x2x3 = torch.einsum('lijk,bi,bj,bk->bl', self.V, x1, x2, x3)
        return Vx1x2x3 + self.bias

class OneWordTensorRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Regression initialization for mapping one input vector to one output vector.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Tensor for mapping one input vector to one output vector
        self.V = nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))

        # Initialize slice-by-slice
        for i in range(output_dim):
            torch.nn.init.xavier_uniform_(self.V[i])

    def forward(self, x):
        """
        Forward pass for one input vector.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        Vx = torch.einsum('ik,bk->bi', self.V, x)
        return Vx + self.bias


def parallel_shuffle(data1, data2):
    """
    Randomly shuffle two datasets in parallel.

    Args:
        data1: First dataset to shuffle
        data2: Second dataset to shuffle
    Modifies:
        data1, data2
    """
    if len(data1) != len(data2):
        raise ValueError("Datasets must have the same length to shuffle in parallel.")

    indices = list(range(len(data1)))
    random.shuffle(indices)

    # Shuffle both datasets using the same indices
    if isinstance(data1, torch.Tensor):
        data1[:] = torch.stack([data1[i] for i in indices])
    else:
        data1[:] = [data1[i] for i in indices]

    if isinstance(data2, torch.Tensor):
        data2[:] = torch.stack([data2[i] for i in indices])
    else:
        data2[:] = [data2[i] for i in indices]

#############################################
############# k-word regression #############
#############################################

def k_word_regression(model_destination, embedding_set, ground_truth, tuple_len, module: nn.Module,
                    num_epochs = 50, word_dim = 100, sentence_dim = 300, lr = 0.1, shuffle = False):
    """
    Regression function meant to handle differnt word len regressions. Produces
    linear map between k embeddings and one ground_truth tensor. Thus far only
    supports 1, 2, and 3 word regressions.

    Args: 
        model_destination: file path of final regression model (.pt file preferred)
        embedding_set: list of tuples of raw dependent data embeddings
        ground_truth: tensor containing empirical contextual sentence embedding 
        num_epochs: number of epochs to train
        word_dim: dimension word embeddings 
        sentence_dim: dimension of sentence embeddings

    Throws:
        Exception if ground truth data is of different len to word embedding data len
    """

    t = ground_truth # t stores ground truth data
    s_o = embedding_set #s_o stores word data
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    if device == "cuda:2":
        torch.cuda.empty_cache() 
    module.to(device)

    if len(t) != len(s_o):
        raise Exception("Mismatched data dimensions")
    
    print(">shuffling data...")
    if shuffle:
        parallel_shuffle(s_o, t)
    print(">done!\n\n\n")

    num_nouns = len(t)
    test_size = num_nouns // 5  # 20% of the data
    train_size = num_nouns - test_size  # Remaining 80%

    # Allocating space for testing and training tensors
    print(">allocating space for testing and training tensors...")
    s_o_tensor = torch.zeros((train_size, tuple_len, word_dim)).to(device)
    test_s_o_tensor = torch.zeros((test_size, tuple_len, word_dim)).to(device)
    
    ground_truth = torch.zeros((train_size, sentence_dim)).to(device)
    ground_truth_test = torch.zeros((test_size, sentence_dim)).to(device)
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
        for tup_i in range(0, tuple_len):
            s_o_tensor[i][tup_i] = noun_tup[tup_i]
        ground_truth[i] = torch.Tensor(verb1_sentences[i])
    print(">done!\n\n\n")
    
    # Assembling test tensors
    print(">Assembling testing tensors...")
    for i, noun_tup in enumerate(noun_pairs_test):
        for tup_i in range(0, tuple_len):
            test_s_o_tensor[i][tup_i] = noun_tup[tup_i]
        ground_truth_test[i] = torch.Tensor(sentence_test[i])
    print(">done!\n\n\n")
    
    #utilizing Adadelta regularization
    optimizer = optim.Adadelta(module.parameters(), lr=lr)

    print(">Running regression...")

    nouns_train = list()
    for i in range(tuple_len):
        #print(i, tuple_len, s_o_tensor.shape)
        nouns_train.append(s_o_tensor[:, i, :])

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        #to-do: make more dynamic w/ list
        if tuple_len == 3:
            predicted = module(nouns_train[0],nouns_train[1], nouns_train[2])
        elif tuple_len == 2:
            predicted = module(nouns_train[0],nouns_train[1])
        elif tuple_len == 1:
            predicted = module(nouns_train[0])
        else:
            raise Exception("Unsupported tuple length")

        # Compute loss (Mean Squared Error)
        loss = torch.mean((predicted - ground_truth)**2)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Print loss for each epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.20f}')

        # Debugging: Check if weights are being updated
        # if epoch == 0 or epoch == num_epochs - 1:
        #     print(f"Sample weights at epoch {epoch + 1}: {list(model.parameters())[0][0][:5]}")

    print(f'>done! Final In-Sample Loss: {loss.item():.20f}\n\n\n')

    # Save model weights
    torch.save(module.state_dict(), model_destination)
    print(f"Model weights saved to: {model_destination}")


    """************************testing************************"""


    print(">************************testing************************")

    nouns_test = list()
    for i in range(tuple_len):
        nouns_test.append(test_s_o_tensor[:, i, :])

    if tuple_len == 3:
        predicted_test = module(nouns_test[0],nouns_test[1], nouns_test[2])
    elif tuple_len == 2:
        predicted_test = module(nouns_test[0],nouns_test[1])
    elif tuple_len == 1:
        predicted_test = module(nouns_test[0])
    else:
        raise Exception("Unsupported tuple length")

    

    loss = torch.mean((predicted_test - ground_truth_test)**2)

    print(f'>done! Test Sample Loss: {loss.item():.4f}\n\n\n')   

    torch.save(module.state_dict(), model_destination)






