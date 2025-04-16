import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

import json
with open("data/one_verb.json") as file_in:
    data = json.load(file_in)

print("success")