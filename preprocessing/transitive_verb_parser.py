import spacy
import json
import time
from multiprocessing import Pool, cpu_count
import numpy as np

NOUN_NUM = 50

def worker(index, data):
    t = time.time()

    nlp = spacy.load("en_core_web_sm", disable=['ner'])
    raw_dict = dict() #meant to cache number of occurences of each noun

    for i, sentence in enumerate(data):
        if len(sentence.strip()) == 0:
            continue

        doc = nlp(str(sentence))
        
        for token in doc:
            if token.pos_ == "VERB":
                lemma = token.lemma_  # Get lemma of the verb
                obj = {child.lemma_ for child in token.children if child.dep_ == "dobj"}  # Get noun lemma
                subj = {child.lemma_ for child in token.children if child.dep_ in ("nsubj", "nsubjpass")}  # Get noun lemma

                if obj and subj:
                    if lemma not in raw_dict:
                        raw_dict[lemma] = {"subjects": {}, "objects": {}}  # Store noun counts as dictionaries
                    
                    for s in subj:
                        raw_dict[lemma]["subjects"][s] = raw_dict[lemma]["subjects"].get(s, 0) + 1
                    for o in obj:
                        raw_dict[lemma]["objects"][o] = raw_dict[lemma]["objects"].get(o, 0) + 1
        
        if i % 1000 == 0:
            print(f'Thread {index}, {i}/{len(data)}', "sentences parsed,", int(time.time() - t), "seconds elapsed")
    
    print(f'Thread {index} completed')

    return raw_dict

if __name__ == "__main__":
    file_in = open("../data_raw/IMDB_Textblock.txt", 'r')
    file_out = open("../data/top_transitive.json", 'w')

    text = file_in.read().split(".")
    nlp = spacy.load("en_core_web_sm", disable=['ner'])

    # Use sets to prevent duplicates
    verb_dict: dict[str, list[set[str], set[str]]] = {}
    raw_dict = dict() #meant to cache number of occurences of each noun

    with Pool(processes=cpu_count()) as p:
        for rd in p.starmap(worker, enumerate(np.array_split(text, cpu_count()))):
            for lemma, posDict in rd.items():
                if lemma not in raw_dict:
                    raw_dict[lemma] = posDict
                else:
                    for pos in posDict:
                        for tokenText in posDict[pos]:
                            raw_dict[lemma][pos][tokenText] = raw_dict[lemma][pos].get(tokenText, 0) + posDict[pos][tokenText]

    for verb in raw_dict:
        
        nouns = raw_dict[verb]
        sorted_subjects = sorted(nouns['subjects'].items(), key=lambda item: item[1], reverse=True)
        #print("subjects", sorted_subjects)

        sorted_objects = sorted(nouns['objects'].items(), key=lambda item: item[1], reverse=True)
        #print("objects", sorted_objects)
        
        if(len(sorted_subjects) >= NOUN_NUM and len(sorted_objects) >= NOUN_NUM):
            verb_dict[verb] = [set(), set()]
            verb_dict[verb][0].update(set(list(zip(*sorted_subjects[:NOUN_NUM]))[0]))

            verb_dict[verb][1].update(set(list(zip(*sorted_objects[:NOUN_NUM]))[0]))
    
    # print("----------------")
    # for thing in verb_dict:
    #     print(thing, verb_dict[thing])

    #Convert sets back to lists for JSON serialization
    json_ready_dict = {verb: [list(subjects), list(objects)] for verb, (subjects, objects) in verb_dict.items()}

    json.dump(json_ready_dict, file_out, indent=4)
    file_out.close()
