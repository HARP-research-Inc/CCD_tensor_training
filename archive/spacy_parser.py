import spacy
import json

nlp = spacy.load("en_core_web_sm")

if __name__ == "__main__":
    file_in = open("data_raw/IMDB_Textblock.txt", 'r')
    file_out = open("data/naive_transitive.json", 'w')

    text = file_in.read().split(".")

    # Use sets to prevent duplicates
    verb_dict: dict[str, list[set[str], set[str]]] = {}

    for sentence in text:
        if len(sentence.strip()) == 0:
            continue

        doc = nlp(sentence)

        for token in doc:
            if token.pos_ == "VERB":
                lemma = token.lemma_  # Get lemma of the verb
                obj = {child.lemma_ for child in token.children if child.dep_ == "dobj"}  # Get noun lemmas
                subj = {child.lemma_ for child in token.children if child.dep_ in ("nsubj", "nsubjpass")}  # Get noun lemmas

                if obj and subj:
                    #print(f"Verb: {token.text} (Lemma: {lemma}), Subject: {subj}, Object: {obj}")

                    if lemma not in verb_dict:
                        verb_dict[lemma] = [set(), set()]  # Store lemma instead of conjugated verb

                    nouns = verb_dict[lemma]
                    nouns[0].update(subj)  # Add subjects (no duplicates)
                    nouns[1].update(obj)   # Add objects (no duplicates)

    # Convert sets back to lists for JSON serialization
    json_ready_dict = {verb: [list(subjects), list(objects)] for verb, (subjects, objects) in verb_dict.items()}

    json.dump(json_ready_dict, file_out, indent=4)
    file_out.close()
