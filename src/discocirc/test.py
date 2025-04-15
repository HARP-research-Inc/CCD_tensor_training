import spacy
nlp = spacy.load("training/coref")

doc = nlp("John Smith called from New York, he says it's raining in the city.")
# check the word clusters
print("=== word clusters ===")
word_clusters = [val for key, val in doc.spans.items() if key.startswith("coref_head")]
for cluster in word_clusters:
    print(cluster)
# check the expanded clusters
print("=== full clusters ===")
full_clusters = [val for key, val in doc.spans.items() if key.startswith("coref_cluster")]
for cluster in full_clusters:
    print(cluster)