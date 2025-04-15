import spacy

nlp = spacy.load("en_core_web_lg")

with open("simple.txt", 'r') as file:
    text = file.read()

doc = nlp(text)

give_children = doc[1].children

# for child in give_children:
#     print(child.text)

for token in doc:
    print("---------------------")
    print(token.text + ":")
    [print(child.text, child.pos_) for child in token.children]    
