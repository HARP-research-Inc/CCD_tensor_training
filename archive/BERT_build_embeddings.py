import torch
from transformers import BertModel, BertTokenizer
import json

num_nouns = 50
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

if __name__ == "__main__":

    file_in = open("data/one_verb.json")
    data = json.load(file_in)

    num_sentences = len(data)

    empirical_embeddings = list()
    
    s_o_embeddings = list()

    for v_i, verb in enumerate(data):
        nouns = data[verb]

        subjects = nouns[0]
        objects = nouns[1]

        #embedding = Verb(verb, N = max_nouns)
        
        
        for s_i, subject in enumerate(subjects, start=0):
            for o_i, object in enumerate(objects, start=0):
                sentence = subject + " " + verb + " " + object

                #tokenize
                sentence_tokenized = tokenizer(sentence, return_tensors="pt")
                subject_tokenized = tokenizer(subject, return_tensors="pt")
                object_tokenized = tokenizer(object, return_tensors="pt")

                #get sentence embedding
                with torch.no_grad():
                    sentence_embedding = model(**sentence_tokenized)
                    subject_embedding = model(**subject_tokenized)
                    object_embedding = model(**object_tokenized)

                
                #get last hidden state of the first token
                sentence_embedding = sentence_embedding.last_hidden_state[:, 0, :] 
                subject_embedding = subject_embedding.last_hidden_state[:, 0, :]    
                object_embedding = object_embedding.last_hidden_state[:, 0, :]




                #embedding.update_tensor_at(s_i, o_i, sentence_embedding) 
                
                #print("SENTENCE " + sentence + "\n" , sentence_embedding.shape)

                s_o_embeddings.append((subject_embedding, object_embedding))
                empirical_embeddings.append(sentence_embedding)
                
    
            #empirical_embeddings.append(embedding.tensor)

    # for embedding in embeddings:
    #     print(embedding.name)
    #     print(embedding)

    # torch.save(empirical_embeddings, "data/empirical_embeddings.pt")
    # torch.save(s_o_embeddings, "data/dependent_data.pt")

    torch.save(empirical_embeddings, "data/1_verb_empirical_embeddings.pt")
    torch.save(s_o_embeddings, "data/1_verb_dependent_data.pt")