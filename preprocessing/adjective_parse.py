import spacy
import json
import re

NOUN_NUM = 50

if __name__ == "__main__":
    file_in = open("../data_raw/IMDB_Textblock.txt", 'r')
    file_out = open("../data/top_adjective.json", 'w')

    text = re.split(r"\.|\?|\!|\;", file_in.read())
    nlp = spacy.load("en_core_web_sm")

    adj_dict: dict[str, set[str]] = {}
    raw_dict = dict()

    for i, sentence in enumerate(text):
        if len(sentence.strip()) == 0:
            continue
        doc = nlp(sentence)

        for token in doc:
            if token.pos_ == "NOUN":
                adjectives = {child.text for child in token.children if child.pos_ == "ADJ"}
                if(len(adjectives) > 0):
                    print(f"Token: {token.text}, Children: {adjectives}")
                    for adj in adjectives:
                        if adj not in raw_dict:
                            raw_dict[adj] = {"nouns": {}}  
                        raw_dict[adj]["nouns"][token.text] = raw_dict[adj]["nouns"].get(token.text, 0) + 1
                        
                        
                        # for noun in raw_dict[adj]["nouns"]:
                        #     if noun not in adj_dict:
                        #         adj_dict[noun] = set()
                        #     adj_dict[noun].add(adj)
        
        # if (i > 1000):
        #     break

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