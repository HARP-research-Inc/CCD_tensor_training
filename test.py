import json
from util import get_embedding_in_parallel

# file_in = open("data/top_adjective.json")

# data = json.load(file_in)

# print(len(data))

# for adj in data:
#     print(adj, data[adj])

print(get_embedding_in_parallel("rensselaer"))
