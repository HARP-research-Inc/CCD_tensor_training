import torch
import sys
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import time

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.regression import TwoWordTensorRegression, CPTensorRegression

def get_embedding_in_parallel(word, model):
    word_embedding = model.encode(word, convert_to_tensor=True)
    word_embedding = word_embedding.cpu().numpy().reshape(1, -1)

    return torch.from_numpy(word_embedding)

if __name__ == "__main__":
	bert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

	model = TwoWordTensorRegression(384, 384)
	model.load_state_dict(torch.load("transitive_verb_model/abandon", weights_only=True))
	model.eval()
	model2 = TwoWordTensorRegression(384, 384)
	model2.load_state_dict(torch.load("transitive_verb_model/accompany", weights_only=True))
	model2.eval()
	model3 = TwoWordTensorRegression(384, 384)
	model3.load_state_dict(torch.load("transitive_verb_model/strike", weights_only=True))
	model3.eval()

	t = time.time()
      
	cosi = torch.nn.CosineSimilarity(dim=1) 

	a = model(get_embedding_in_parallel("man", bert), get_embedding_in_parallel("man", bert))
	b = model2(get_embedding_in_parallel("director", bert), get_embedding_in_parallel("family", bert))
	c = model2(get_embedding_in_parallel("man", bert), get_embedding_in_parallel("king", bert))
	d = model3(get_embedding_in_parallel("empire", bert), get_embedding_in_parallel("balance", bert))
      
	amen = CPTensorRegression([384], 384, 100)
	amen.load_state_dict(torch.load("intj_model/amen", weights_only=True))
	amen.eval()
      
	amen_brother = amen(get_embedding_in_parallel("brother", bert))
	amen_sister = amen(get_embedding_in_parallel("sister", bert))

	print()
	print('similarity("brother", "sister") =>', cosi(get_embedding_in_parallel("brother", bert), get_embedding_in_parallel("sister", bert)).item())
	print('similarity("amen brother", "brother") =>', cosi(amen_brother, get_embedding_in_parallel("brother", bert)).item())
	print('similarity("amen sister", "sister") =>', cosi(amen_sister, get_embedding_in_parallel("sister", bert)).item())
	print('similarity("amen brother", "amen sister") =>', cosi(amen_brother, amen_sister).item())
	print('similarity_bert("amen brother", "brother") =>', cosi(get_embedding_in_parallel("amen brother", bert), get_embedding_in_parallel("brother", bert)).item())
	print('similarity_bert("amen sister", "sister") =>', cosi(get_embedding_in_parallel("amen sister", bert), get_embedding_in_parallel("sister", bert)).item())
	print('similarity_bert("amen brother", "amen sister") =>', cosi(get_embedding_in_parallel("amen brother", bert), get_embedding_in_parallel("amen sister", bert)).item())
    
	andModel = CPTensorRegression([384, 384], 384, 100)
	andModel.load_state_dict(torch.load("cconj_noun_model/and", weights_only=True))
	andModel.eval()
      
	rice_and_beans = andModel(get_embedding_in_parallel("rice", bert), get_embedding_in_parallel("beans", bert))
	beans_and_rice = andModel(get_embedding_in_parallel("beans", bert), get_embedding_in_parallel("rice", bert))
      
	print('similarity("rice", "beans") =>', cosi(get_embedding_in_parallel("rice", bert), get_embedding_in_parallel("beans", bert)).item())
	print('similarity("rice and beans", "beans and rice") =>', cosi(rice_and_beans, beans_and_rice).item())
	print('similarity_bert("rice and beans", "beans and rice") =>', cosi(get_embedding_in_parallel("rice and beans", bert), get_embedding_in_parallel("beans and rice", bert)).item())

	print()
	print(cosi(a, b))
	print(cosi(b, c))
	print(cosi(a, c))
	print(cosi(a, d))
	print(cosi(b, d))
	print(cosi(c, d))

	print(f"Diff: {time.time() - t}")

