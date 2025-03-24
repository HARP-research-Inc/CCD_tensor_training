import tltorch
import torch
from torch import nn
import numpy as np
from util import cp_decompose
import torch.optim as optim


class FullRankTensorRegression(nn.Module):
    def __init__(self, noun_dim, sent_dim):
        super().__init__()
        self.sent_dim = sent_dim
        self.noun_dim = noun_dim
        # Full-rank tensor parameter with Xavier initialization
        self.V = nn.Parameter(torch.randn(sent_dim, noun_dim, noun_dim) * np.sqrt(2.0 / (noun_dim + sent_dim)))

    def forward(self, s, o):
        """
        s: (batch_size, noun_dim)
        o: (batch_size, noun_dim)
        """
        # Perform tensor contraction explicitly
        # Using Einstein notation for clarity: result[b,l] = sum_jk V_ljk * s_j * o_k
        Vs_o = torch.einsum('ljk,bj,bk->bl', self.V, s, o)
        return Vs_o




if __name__ == "__main__":
    num_verbs = 1
    num_s_o = 50
    embedding_dim = 300

    t = torch.load("data/hybrid_empirical_embeddings.pt", weights_only=False)
    s_o = torch.load("data/hybrid_dependent_data.pt", weights_only=False) #list of tuples of tensors

    #do just one verb for now

    
    s_o_tensor = torch.zeros((num_s_o**2, 2, embedding_dim))
    ground_truth = torch.zeros((num_s_o**2, embedding_dim))

    print(len(t))
    print(len(s_o))
    
    noun_pairs_test = s_o[:int(0.2*num_s_o**2)]
    sentence_test = t[:int(0.2*num_s_o**2)]

    # 
    verb1_noun_pairs = s_o[int(0.2*num_s_o**2):num_s_o**2]
    verb1_sentences = t[int(0.2*num_s_o**2):num_s_o**2]



    print("shape of verb1_sentences: ", verb1_sentences[0].shape)

    print(len(verb1_sentences))
    print(ground_truth.shape)

    #assembling training tensor
    for i, noun_tup in enumerate(verb1_noun_pairs):
        print(i)
        # print(f"Index: {i}")
        # print(f"Shape of noun_tup: {noun_tup[0].shape}, {noun_tup[1].shape}")
        # print(f"Shape of verb1_sentences[{i}]: {verb1_sentences[i].shape}")
        # print(f"Shape of ground_truth[{i}]: {ground_truth[i].shape}")
        
        s_o_tensor[i][0] = noun_tup[0]
        s_o_tensor[i][1] = noun_tup[1]
        ground_truth[i] = torch.Tensor(verb1_sentences[i])
    

    

    num_epochs = 1  # Number of epochs for training

    model = FullRankTensorRegression(embedding_dim, embedding_dim)
    optimizer = optim.Adadelta(model.parameters(), lr=0.1)  # Adjust learning rate

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
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    print(f'Final In-Sample Loss: {loss.item():.4f}')
    #testing 

    
    for i, noun_tup in enumerate(noun_pairs_test):
        subject = noun_tup[0]
        object = noun_tup[1]
        sentence = torch.Tensor(sentence_test[i])

    # predicted_test = model(subjects_test, objects_test)
    # loss = torch.mean((predicted_test - ground_truth_test)**2)

    print(f'Test Sample Loss: {loss.item():.4f}')   

    torch.save(model.state_dict(), "data/hybrid_weights.pt")

    


