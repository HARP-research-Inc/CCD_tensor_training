import spacy

nlp = spacy.load("en_core_web_trf")

text = "Tarare did you eat a fucking baby?"

doc = nlp(text)

give_children = doc[1].children

# for child in give_children:
#     print(child.text)

for token in doc:
    print("---------------------")
    print(token.text + f"({token.pos_})" + ":")
    [print(child.text, child.pos_) for child in token.children]    
