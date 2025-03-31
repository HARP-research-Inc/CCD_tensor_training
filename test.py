import json
from run_regression import two_word_regression


with open("data/top_transitive.json") as file:
    data = json.load(file)
    print(len(data))

