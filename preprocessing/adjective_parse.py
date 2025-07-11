import spacy
import json
import re
import time
from multiprocessing import Pool, cpu_count
import numpy as np

NOUN_NUM = 50

def worker(index, data):
    t = time.time()

    nlp = spacy.load("en_core_web_sm", disable=['lemmatizer', 'ner'])
    raw_dict = dict()

    for i, sentence in enumerate(data):
        if len(sentence.strip()) == 0:
            continue
        doc = nlp(str(sentence))

        for token in doc:
            if token.pos_ == "NOUN":
                adjectives = {child.text for child in token.children if child.pos_ == "ADJ"}
                if(len(adjectives) > 0):
                    for adj in adjectives:
                        if adj not in raw_dict:
                            raw_dict[adj] = {"nouns": {}}  
                        raw_dict[adj]["nouns"][token.text] = raw_dict[adj]["nouns"].get(token.text, 0) + 1
        
        if i % 1000 == 0:
            print(f'Thread {index}, {i}/{len(data)}', "sentences parsed,", int(time.time() - t), "seconds elapsed")
    
    print(f'Thread {index} completed')

    return raw_dict
    

if __name__ == "__main__":
    file_in = open("../data_raw/IMDB_Textblock.txt", 'r')
    file_out = open("../data/top_adjective.json", 'w')

    text = re.split(r"\.|\?|\!|\;", file_in.read())
    nlp = spacy.load("en_core_web_sm", disable=['lemmatizer', 'ner'])

    adj_dict: dict[str, set[str]] = {}
    raw_dict = dict()

    t = time.time()

    with Pool(processes=cpu_count()) as p:
        for rd in p.starmap(worker, enumerate(np.array_split(text, cpu_count()))):
            for adj, posDict in rd.items():
                if adj not in raw_dict:
                    raw_dict[adj] = posDict
                else:
                    for pos in posDict:
                        for tokenText in posDict[pos]:
                            raw_dict[adj][pos][tokenText] = raw_dict[adj][pos].get(tokenText, 0) + posDict[pos][tokenText]

    for adj in raw_dict:
        nouns = raw_dict[adj]["nouns"]
        sorted_nouns = sorted(nouns.items(), key=lambda item: item[1], reverse=True)[:NOUN_NUM]
        if len(sorted_nouns) >= NOUN_NUM:
            if adj not in adj_dict:
                adj_dict[adj] = set()
            adj_dict[adj].update(set(list(zip(*sorted_nouns[:NOUN_NUM]))[0]))

    json_ready_dict = {adj: list(adj_dict[adj]) for adj in adj_dict}
    for adj in json_ready_dict:
        print(adj, json_ready_dict[adj])
        
    json.dump(json_ready_dict, file_out)

    file_out.close()