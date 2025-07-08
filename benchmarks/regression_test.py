import nltk
from sklearn.linear_model import LogisticRegression
import torch
from src.DisCoBERT.categories import Box

def build_data(path):
    with open(path, 'r') as file:
        data = file.readlines()

    total_sentences = 0

    train_embeddings = list()
    train_classifications = list()

    test_embeddings = list()
    test_classifications = list()

    n = 0

    for line in data:
        if line[0] == '#' or len(line) < 4:
            continue

        point = line.strip().split("-", 1)

        y = int(point[0].strip())

        if y < 6:
            y = 0
        elif y >= 6:
            y = 1

        sentences = nltk.sent_tokenize(point[1].strip())
        total_sentences += len(sentences)
        for sentence in sentences:
            n += 1
            embedding = Box.model_cache.retrieve_BERT(sentence.strip()).tolist()[0]
            print(sentence[0:20], y)
            print(len(embedding))

            if n % 10 == 0:
                test_embeddings.append(embedding)
                test_classifications.append(y)
            else:
                train_embeddings.append(embedding)
                train_classifications.append(y)
        
    print(f"Total sentences: {total_sentences}")

    return train_embeddings, train_classifications, test_embeddings, test_classifications

def logisitic_regression():
    train_embeddings, train_classifications, test_embeddings, test_classifications = build_data("benchmarks/classification.txt")

    model = LogisticRegression(max_iter=1000)
    model.fit(train_embeddings, train_classifications)

    print("Model coefficients:", model.coef_)
    print("Model intercept:", model.intercept_)

    print("Training score:", model.score(train_embeddings, train_classifications))
    print("Testing score:", model.score(test_embeddings, test_classifications))

    return model

if __name__ == "__main__":
    logisitic_regression()

