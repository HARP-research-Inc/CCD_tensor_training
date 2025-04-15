from regression import FullRankTensorRegression, k_word_regression, two_word_regression
import torch
from util import get_embedding_in_parallel
from sentence_transformers import SentenceTransformer
from transitive_build_embeddings import build_one_verb, BERT_only_no_PCA
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

def build_trans_verb_model(src, destination, model, epochs):
    """
    
    """
    response = input("WARNING: building this model will take up over 30 gb of space. Type \'YES\' to continue, type anything else to exit: ")
    if response != "YES":
        return
    file_in = open(src)
    data = json.load(file_in)

    big_BERT = None

    for verb in data:
        #print(data[verb])
        pca, empirical_embeddings, s_o_embeddings = build_one_verb(data, verb, model)
        module = FullRankTensorRegression(300, 300)
        print(len(s_o_embeddings), len(empirical_embeddings))
        k_word_regression(destination+f"/{verb}", s_o_embeddings, empirical_embeddings, 
                          2, module, num_epochs=epochs, word_dim=300, lr=0.5, shuffle=True)

    
def bert_on_bert(src, destination, model, epochs):
    with open(src) as file_in:
        data = json.load(file_in)
    
    for verb in data:
        empirical_embeddings, s_o_embeddings = BERT_only_no_PCA(data, verb, model)
        module = FullRankTensorRegression(384, 384)
        print(empirical_embeddings.shape, len(s_o_embeddings))
        k_word_regression(destination+f"/{verb}", s_o_embeddings, empirical_embeddings, 
                          2, module, num_epochs=epochs, sentence_dim=384, word_dim=384, lr=0.5, shuffle=True)
        


    
    
    


if __name__ == "__main__":
    #noun_adjective_pair_regression("models/adj_weights.pt", epochs=5)
    #transitive_verb_regression("models/hybrid_weights_dummy", epochs=400)
    #concatenated_three_word_regression("models/three_word_weights.pt", epochs=10)


    # model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # build_trans_verb_model("data/top_transitive.json","transitive_verb_model/", model, epochs=50)
    

    bert_on_bert("data/one_verb.json", "models/", SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"), epochs=500)



    print("Regression complete.")