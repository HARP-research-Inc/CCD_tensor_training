import gensim.downloader as api
from fastapi import FastAPI, HTTPException
import numpy as np
import torch

print("Loading fastText model...")
ft_model = api.load('fasttext-wiki-news-subwords-300')
print("Done loading fastText model.")

app = FastAPI()

#print(ft_model['neologism'].shape)  

@app.get("/embedding/{word}")
def read_item(word: str):
    try:
        return {"embedding": torch.tensor(ft_model[word]).unsqueeze(0).tolist()}
    except KeyError:
        raise HTTPException(status_code=404, detail="Word not found")



