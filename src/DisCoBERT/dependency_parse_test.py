import spacy

nlp = spacy.load("en_core_web_trf")

text = "The men among the dogs"

doc = nlp(text)

give_children = doc[1].children

# for child in give_children:
#     print(child.text)

for token in doc:
    print("---------------------")
    print(token.text + f"({token.pos_}), ({token.dep_})" + ":")
    [print(child.text + f"({child.pos_}), ({child.dep_})") for child in token.children]    
