import torch
import sys
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import time

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.regression import CPTensorRegression

def get_embedding_in_parallel(word, model):
    word_embedding = model.encode(word, convert_to_tensor=True)
    word_embedding = word_embedding.cpu().numpy().reshape(1, -1)

    return torch.from_numpy(word_embedding)

if __name__ == "__main__":
	bert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

	vbd = CPTensorRegression([384], 384, 100)
	vbd.load_state_dict(torch.load("tense_model/VBD", weights_only=True))
	vbd.eval()
	vbg = CPTensorRegression([384], 384, 100)
	vbg.load_state_dict(torch.load("tense_model/VBG", weights_only=True))
	vbg.eval()
	vbn = CPTensorRegression([384], 384, 100)
	vbn.load_state_dict(torch.load("tense_model/VBN", weights_only=True))
	vbn.eval()

	t = time.time()
      
	cosi = torch.nn.CosineSimilarity(dim=1) 

	run = get_embedding_in_parallel("run", bert)
	ran = get_embedding_in_parallel("ran", bert)
	past_run = vbd(run)
	past_participle_run = vbn(run)
    
	print()
	print('similarity("run" (base), "ran") =>', cosi(run, past_run).item())
	print('similarity("run" (base), "run" (past participle)) =>', cosi(run, past_participle_run).item())
	print('similarity("ran", "run" (past participle)) =>', cosi(past_run, past_participle_run).item())
	print('similarity("ran", "ran") =>', cosi(ran, past_run).item())
      
	print(f"Diff: {time.time() - t}")

