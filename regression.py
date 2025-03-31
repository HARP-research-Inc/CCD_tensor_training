import torch
from torch import nn
import numpy as np
import torch.optim as optim

class FullRankTensorRegression(nn.Module):
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

def two_word_regression(model_destination, embedding_set, ground_truth, 
                        num_epochs = 50, embedding_dim = 300, lr = 0.1):
    """
    Regression for datasets of paired words e.g. adjective-noun pairs. Produces
    linear map between paired embeddings and ground_truth tensor.

    Args: 
        model_destination: file path of final regression model (.pt file preferred)
        embedding_set: list of tuples of raw dependent data embeddings
        ground_truth: tensor containing empirical contextual sentence embedding 
        num_epochs: number of epochs to train
        embedding_dim: dimension of embeddings (effectively enforces share dimensionality between both datasets)
        lr: learning rate

    Throws:
        Exception if ground truth data is of different len to word embedding data len
    """

    # to-do rename variables from original transitive verb training names

    t = ground_truth # t stores 
    s_o = embedding_set #

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

    #utilizing Adadelta regularization
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

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
        # if epoch == 0 or epoch == num_epochs - 1:
        #     print(f"Sample weights at epoch {epoch + 1}: {list(model.parameters())[0][0][:5]}")

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

def three_word_regression(model_destination, embedding_set, ground_truth, 
                          num_epochs = 50, word_dim = 100, sentence_dim = 300, lr = 0.1):
    """
    Regression for datasets of tripled words e.g. SVO triplets. Produces
    linear map between triplet embeddings and ground_truth tensor.

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

    if len(t) != len(s_o):
        raise Exception("Mismatched data dimensions")

    num_nouns = len(t)
    test_size = num_nouns // 5  # 20% of the data
    train_size = num_nouns - test_size  # Remaining 80%

    # Allocating space for testing and training tensors
    print(">allocating space for testing and training tensors...")
    s_o_tensor = torch.zeros((train_size, 3, word_dim))
    test_s_o_tensor = torch.zeros((test_size, 3, word_dim))
    
    ground_truth = torch.zeros((train_size, sentence_dim))
    ground_truth_test = torch.zeros((test_size, sentence_dim))
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
        s_o_tensor[i][2] = noun_tup[2]
        ground_truth[i] = torch.Tensor(verb1_sentences[i])
    print(">done!\n\n\n")
    
    # Assembling test tensors
    print(">Assembling testing tensors...")
    for i, noun_tup in enumerate(noun_pairs_test):
        test_s_o_tensor[i][0] = noun_tup[0]
        test_s_o_tensor[i][1] = noun_tup[1]
        test_s_o_tensor[i][2] = noun_tup[2]
        ground_truth_test[i] = torch.Tensor(sentence_test[i])
    print(">done!\n\n\n")
    
    model = ThreeWordTensorRegression(word_dim, sentence_dim)

    #utilizing Adadelta regularization
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    print(">Running regression...")
    nouns1 = s_o_tensor[:, 0, :]
    nouns2 = s_o_tensor[:, 1, :]
    nouns3 = s_o_tensor[:, 2, :]

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        predicted = model(nouns1, nouns2, nouns3)

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
    torch.save(model.state_dict(), model_destination)
    print(f"Model weights saved to: {model_destination}")


    """************************testing************************"""


    print(">************************testing************************")
    nouns1 = test_s_o_tensor[:, 0, :]
    nouns2 = test_s_o_tensor[:, 1, :]
    nouns3 = test_s_o_tensor[:, 2, :]
    predicted_test = model(nouns1, nouns2, nouns3)

    loss = torch.mean((predicted_test - ground_truth_test)**2)

    print(f'>done! Test Sample Loss: {loss.item():.4f}\n\n\n')   

    torch.save(model.state_dict(), model_destination)

def k_word_regression(model_destination, embedding_set, ground_truth, tuple_len,
                          num_epochs = 50, word_dim = 100, sentence_dim = 300, lr = 0.1):
    """
    Regression for datasets of tripled words e.g. SVO triplets. Produces
    linear map between triplet embeddings and ground_truth tensor.

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

    if len(t) != len(s_o):
        raise Exception("Mismatched data dimensions")

    num_nouns = len(t)
    test_size = num_nouns // 5  # 20% of the data
    train_size = num_nouns - test_size  # Remaining 80%

    # Allocating space for testing and training tensors
    print(">allocating space for testing and training tensors...")
    s_o_tensor = torch.zeros((train_size, tuple_len, word_dim))
    test_s_o_tensor = torch.zeros((test_size, tuple_len, word_dim))
    
    ground_truth = torch.zeros((train_size, sentence_dim))
    ground_truth_test = torch.zeros((test_size, sentence_dim))
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
    
    
    if tuple_len == 3:
        model = ThreeWordTensorRegression(word_dim, sentence_dim)
    elif tuple_len == 2:
        model = FullRankTensorRegression(word_dim, sentence_dim)

    #utilizing Adadelta regularization
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    print(">Running regression...")
    nouns1 = s_o_tensor[:, 0, :]
    nouns2 = s_o_tensor[:, 1, :]
    if tuple_len == 3:
        nouns3 = s_o_tensor[:, 2, :]

    

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        if tuple_len == 3:
            predicted = model(nouns1, nouns2, nouns3)
        elif tuple_len == 2:
            predicted = model(nouns1, nouns2)

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
    torch.save(model.state_dict(), model_destination)
    print(f"Model weights saved to: {model_destination}")


    """************************testing************************"""


    print(">************************testing************************")
    nouns1 = test_s_o_tensor[:, 0, :]
    nouns2 = test_s_o_tensor[:, 1, :]
    if tuple_len == 3:
        nouns3 = test_s_o_tensor[:, 2, :]
    
    if tuple_len == 3:
            predicted_test = model(nouns1, nouns2, nouns3)
    elif tuple_len == 2:
        predicted_test = model(nouns1, nouns2)

    loss = torch.mean((predicted_test - ground_truth_test)**2)

    print(f'>done! Test Sample Loss: {loss.item():.4f}\n\n\n')   

    torch.save(model.state_dict(), model_destination)






