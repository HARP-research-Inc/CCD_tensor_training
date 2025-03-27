import torch
from transformers import BertModel, BertTokenizer
import json
import archive.trans_verb as trans_verb

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


if __name__ == "__main__":
    file = open("data/toy.json", 'r')
    data = json.load(file)
    
    for verb in data:
        nouns = data[verb]

        subjects = nouns[0]
        objects = nouns[1]
        print(nouns)
        for subject in subjects:
            for object in objects:
                sentence = subject + " " + verb + " " + object
                #sentence_tokenized = tokenizer(sentence, return_tensors="pt", padding="max_length", max_length=6, truncation=True)
                sentence_tokenized = tokenizer(sentence, return_tensors="pt")

                with torch.no_grad():
                    s_o = model(**sentence_tokenized)
                
                sentence_embedding = s_o.last_hidden_state

                print("SENTENCE " + sentence + "\n" , sentence_embedding.shape)
