from regression import TwoWordTensorRegression, OneWordTensorRegression, CPTensorRegression, k_word_regression, multi_word_regression, batch_word_regression
import torch
from util import get_embedding_in_parallel
from sentence_transformers import SentenceTransformer
from transitive_build_embeddings import build_one_verb
from torch.multiprocessing import Pool, Manager, cpu_count, set_start_method
import json
import os
import numpy as np
import time
import functools
import pickle

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

def embedding_generator(index: int, data, queue, wordQueue, cache: dict, counter: dict, numThreads: int):
	model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=f"cuda:{[0, 2, 3][index%3]}")

	print(f"Loaded embedding_generator {index}")

	while True:
		try:
			queue_obj = wordQueue.get()

			if not queue_obj:
				break

			t = time.time()
			words = data[queue_obj]

			embeddings = []
			sentence_embeddings = []

			if all(isinstance(item, list) for item in words):
				indices = [0 for _ in range(len(words))]
				variations = set()
				
				n = 0
				terminate = False
				while not terminate:
					if functools.reduce(lambda a, b: a * b, [len(item) for item in words]) < 1_000_000:
						variations.add(tuple(words[i][indices[i]] for i in range(len(words))))

						indices[-1] += 1

						for i in range(len(words) - 1, -1, -1):
							if indices[i] >= len(words[i]):
								if i > 0:
									indices[i] = 0
									indices[i - 1] += 1
								else:
									terminate = True
									break
							else:
								break
					else:
						variations.add(tuple(words[i][np.random.randint(0, len(words[i]))] for i in range(len(words))))

						if len(variations) > 100_000:
							terminate = True
					
					if n % 5000 == 4999:
						print(f'embedding_generator {index} generated {n + 1} variations')
					n += 1

				if len(variations) > 100_000:
					indices = np.random.choice(np.arange(len(variations)), 100_000, replace=False)
				
					var = list(variations)
					variations = [var[i] for i in indices]

				n = 0
				for item in variations:
					arr = []

					item = [queue_obj] + item
					for word in item:
						if word in cache:
							arr.append(cache[word])
						else:
							embedding = get_embedding_in_parallel(word, model)
							cache[word] = embedding
							arr.append(embedding)

					if n % 500 == 499:
						print(f'embedding_generator {index} parsed {n + 1} examples')
					n += 1

					embeddings.append(arr)
					sentence_embeddings.append(get_embedding_in_parallel(f'{word[0]} {queue_obj} {' '.join(word[1:])}'.strip(), model))
			else:
				for word in words:
					item = [word]
					arr = []

					item = [queue_obj] + item
					for word in item:
						if word in cache:
							arr.append(cache[word])
						else:
							embedding = get_embedding_in_parallel(word, model)
							cache[word] = embedding
							arr.append(embedding)
				
					embeddings.append(arr)
					sentence_embeddings.append(get_embedding_in_parallel(f'{word} {queue_obj}', model))
					#sentence_embeddings.append(get_embedding_in_parallel(f'{queue_obj} {word}', model))

			queue.put((embeddings, sentence_embeddings))
			print(f"embedding_generator {index} took {int(time.time() - t)} seconds, {counter["count"]} threads completed")
		except:
			import traceback
			print(traceback.format_exc())
			break

	counter["count"] += 1

	if counter["count"] == numThreads:
		print("Sending terminating signal")
		queue.put(None)
	
	print('embedding_generator completed')


def build_model(src: str, destination: str, epochs: int, producerThreads=3, reverse=False):
	with open(src) as file_in:
		data = json.load(file_in)

	os.makedirs(destination, exist_ok=True)

	manager = Manager()

	cache = manager.dict()

	manager2 = Manager()

	counter = manager2.dict()

	counter["count"] = 0

	manager3 = Manager()
	
	queue = manager3.Queue()

	manager4 = Manager()

	wordQueue = manager4.Queue()

	for key in (list(reversed(sorted(data.keys()))) if reverse else list(data.keys())):
		wordQueue.put(key)
	
	for _ in range(producerThreads):
		wordQueue.put(None)

	word_embeddings = []
	sentence_embeddings = []

	with Pool(processes=producerThreads) as p:
		res1 = p.starmap_async(embedding_generator, [(index, data, queue, wordQueue, cache, counter, producerThreads) for index in range(producerThreads)])

		m = 0
		try:
			while True:
				queue_obj = queue.get()

				if not queue_obj:
					break

				w_e, s_e = queue_obj
				word_embeddings += w_e
				sentence_embeddings += s_e

				m += 1
				print("Received", m, "packages")
		except:
			pass

		word_embeddings *= 1
		sentence_embeddings *= 1

		res1.get()

		p.terminate()
		p.join()
		
	module = CPTensorRegression([384 for _ in range(1 if not isinstance(word_embeddings[0], list) else len(word_embeddings[0]))], 384, 100)
		
	batch_word_regression(os.path.join(destination, "model"), word_embeddings, sentence_embeddings, 1 if not isinstance(word_embeddings[0], list) else len(word_embeddings[0]), module, num_epochs=epochs, sentence_dim=384, word_dim=384, lr=0.001, shuffle=True, device=0)

if __name__ == "__main__":
	#build_model("data/top_adj_amod.json", "general_adj_model", 200, 20)
	build_model("data/top_verb_nsubj.json", "general_intransitive_model", 200, 20)
	print("Regression complete.")