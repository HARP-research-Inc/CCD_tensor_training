import stanza
import json


if __name__ == "__main__":
    file_in = open("data_raw/textblock_tiny.txt", 'r')
    file_out = open("data/naive_transitive.json")

    text = file_in.read().split(".")

    #stanza.download('en')
    nlp = stanza.Pipeline('en')

    verb_dict : dict[str, tuple[set, set]] = {}
    
    for sentence in text:
        if len(sentence) == 0:
            continue

        processed = nlp(sentence)

        i = 1

        sentence_words = processed.sentences[0].words

        #should skip if one word sentence
        while(i < len(sentence_words)):
            if(sentence_words[i].upos == "VERB"):
                s_i = i - 1 #subject index
                o_i = i + 1 #object index

                verb = sentence_words[i].lemma

                if(o_i >= len(sentence_words) or s_i < 0):
                    i += 1 
                    continue
                #print(s_i, o_i, len(sentence_words))
                #print(sentence_words)
                if (sentence_words[s_i].upos == "NOUN" or sentence_words[s_i].upos == "PROPN") \
                and (sentence_words[o_i].upos == "NOUN" or sentence_words[o_i].upos == "PROPN" \
                or sentence_words[o_i].lemma == "the" or sentence_words[o_i].lemma == "a"):
                    
                    if((sentence_words[o_i].lemma == "the" or sentence_words[o_i].lemma == "a") and o_i + 1 < len(sentence_words)):
                        o_i += 1
                    elif(o_i + 1 >= len(sentence_words)):
                        i += 1 
                        continue

                    if(verb not in verb_dict):
                        verb_dict.update({verb : (set(), set())})
                    noun_tup = verb_dict.get(verb)

                    print(type(noun_tup))
                    print(type(noun_tup[0]))
                    noun_tup[0].add(sentence_words[s_i].lemma)
                    noun_tup[1].add(sentence_words[o_i].lemma)
                
                #print(sentence_words[i])

            i += 1 
    
    file_out.write(json.dumps(verb_dict))

