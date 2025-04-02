from regression import FullRankTensorRegression, k_word_regression, two_word_regression
import torch
from util import get_embedding_in_parallel
from sentence_transformers import SentenceTransformer
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
    """
    Regression for SVO sentences using 1 word regression with word embeddings
    concactenated together.
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    file_in = open("data/top_transitive.json")
    data = json.load(file_in)

    dependent_data = list()
    empirical_data = list()

    for verb in data:
        nouns = data[verb]
        for subject in nouns[0]:
            for object in nouns[1]:

                subject_embedding = get_embedding_in_parallel(subject)
                object_embedding = get_embedding_in_parallel(object)
                verb_embedding = get_embedding_in_parallel(verb)
                
                if subject_embedding is None or object_embedding is None or verb_embedding is None:
                    continue
                
                full_tensor = torch.cat((subject_embedding, verb_embedding, object_embedding), dim=0)
                dependent_data.append((full_tensor))

                sentence = subject + " " + verb + " " + object
                empirical_data.append(model.encode(sentence))

        


if __name__ == "__main__":
    noun_adjective_pair_regression("models/adj_weights.pt", epochs=5)
    #transitive_verb_regression("data/hybrid_weights_dummy.pt", epochs=400)

    #concatenated_three_word_regression("models/three_word_weights.pt", epochs=10)

    print("Regression complete.")