from regression import FullRankTensorRegression
import torch
from util import cosine_sim
from transformers import BertModel, BertTokenizer

def get_embedding(word):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    inputs = tokenizer(word, return_tensors='pt')
    outputs = model(**inputs)
    
    # Get the embeddings for the [CLS] token
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    
    return cls_embedding

if __name__ == "__main__":
    model = FullRankTensorRegression(768, 768)
    model.load_state_dict(torch.load("data/1_verb_weights.pt"))

    model.eval()

    s = get_embedding("bat")
    o = get_embedding("ball")
    sentence_embedding_empirical = get_embedding("bat strike ball")

    

    sentence_embedding = model(s, o)
    print(sentence_embedding)
    print(sentence_embedding.shape)

    print(cosine_sim(sentence_embedding.detach().numpy(), sentence_embedding_empirical.detach().numpy()))