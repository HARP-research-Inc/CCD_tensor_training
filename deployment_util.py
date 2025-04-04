import json
import torch

MODEL_PATH = "transitive_verb_model/"

def trans_verb(subject, verb, object):
    verb_tensor_path = MODEL_PATH + verb
    

def build_lookup(first_build = False):
    
    if first_build:
        file_in = open("data/top_transitive.json", 'r')
        file_out = open("temp.txt", 'w')
        data = json.load(file_in)
        for verb in data:
            file_out.write(verb + "\n")
    
        file_in.close()
        file_out.close()
    
    file_in = open("temp.txt")

def test():
    word = "fudge"
    

if __name__ == "__main__":
    build_lookup()