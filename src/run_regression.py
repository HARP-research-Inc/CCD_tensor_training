from regression import TwoWordTensorRegression, OneWordTensorRegression, CPTensorRegression, k_word_regression, multi_word_regression
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

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

def update_version_tracking_json():
	pass

def noun_adjective_pair_regression(destination, epochs = 100):
	"""
	
	"""
	dependent_data = torch.load("data/adj_dependent_data.pt", weights_only=False)

	empirical_data = torch.load("data/adj_empirical_embeddings.pt", weights_only=False)
	
	module = TwoWordTensorRegression(384, 384)


	k_word_regression(destination, dependent_data, empirical_data, 2, module, word_dim=384, sentence_dim=384, num_epochs=epochs, shuffle=True)
	

def transitive_verb_regression(destination, epochs):
	t = torch.load("data/hybrid_empirical_embeddings.pt", weights_only=False)
	s_o = torch.load("data/hybrid_dependent_data.pt", weights_only=False) # List of tuples of tensors

	module = TwoWordTensorRegression(384, 384)
	k_word_regression(destination, s_o, t, 2, module, word_dim=384, sentence_dim=384, num_epochs=epochs, shuffle=True)



def concatenated_three_word_regression(destination, epochs):
	"""
	Regression for SVO sentences using 1 word regression with word embeddings
	concactenated together.
	"""
	model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
	file_in = open("data/top_transitive.json")
	data = json.load(file_in)

	dependent_data = list()
	empirical_data = list()

	for verb in data:
		nouns = data[verb]
		for subject in nouns[0]:
			for object in nouns[1]:

				subject_embedding = get_embedding_in_parallel(subject)
				object_embedding = get_embedding_in_parallel(object)
				verb_embedding = get_embedding_in_parallel(verb)
				
				if subject_embedding is None or object_embedding is None or verb_embedding is None:
					continue
				
				full_tensor = torch.cat((subject_embedding, verb_embedding, object_embedding), dim=0)
				dependent_data.append((full_tensor))

				sentence = subject + " " + verb + " " + object
				empirical_data.append(model.encode(sentence))

def one_verb_worker(index: int, all_data, verbs: list[str], queue, cache: dict, counter: dict, numThreads: int):
	model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=f"cuda:{[0, 2, 3][index%3]}")

	print(f"Loaded one_verb_worker {index}")

	for verb in verbs:
		while queue.qsize() > numThreads * 2:
			time.sleep(1)

		_, empirical_embeddings, s_o_embeddings = build_one_verb(all_data, verb, model, cache)

		queue.put((verb, s_o_embeddings, empirical_embeddings))

	counter["count"] += 1

	if counter["count"] == numThreads:
		queue.put(None)

def trans_verb_worker(index: int, queue, destination, epochs):
	#model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=f"cuda:{[2, 3][index%2]}")

	print(f"Loaded trans_verb_worker {index}")

	t = time.time()
	i = 0
	while True:
		queue_obj = queue.get()

		if not queue_obj:
			queue.put(None)
			break

		verb, s_o_embeddings, empirical_embeddings = queue_obj

		print(f'Accepted {verb} from queue, {queue.qsize()} elements in backlog')

		module = CPTensorRegression([384, 384], 384, 100)
		
		multi_word_regression(os.path.join(destination, verb), s_o_embeddings, empirical_embeddings, 
						  2, module, num_epochs=epochs, sentence_dim=384, word_dim=384, lr=0.001, shuffle=True, device=[0, 2, 3][index%3])
	
		print(f'Thread {index}, {i} verbs parsed, {int(time.time() - t)} seconds elapsed')

		i += 1
	
	print(f'Thread {index} completed')


def build_trans_verb_model(src, destination, epochs):
	"""
	
	"""
	response = input("WARNING: building this model will take up over 30 gb of space. Type \'YES\' to continue, type anything else to exit: ")
	if response != "YES":
		return
	with open(src) as file_in:
		data = json.load(file_in)

	big_BERT = None

	os.makedirs(destination, exist_ok=True)

	manager = Manager()

	cache = manager.dict()

	manager2 = Manager()

	counter = manager2.dict()

	counter["count"] = 0

	manager3 = Manager()
	
	queue = manager3.Queue()

	numThreads = 6

	with Pool(processes=numThreads) as p:
		res1 = p.starmap_async(one_verb_worker, [(index, data, val, queue, cache, counter, numThreads) for index, val in enumerate(np.array_split(list(data.keys()), numThreads))])

		with Pool(processes=3) as p2:
			res2 = p2.starmap_async(trans_verb_worker, [(index, queue, destination, epochs) for index in range(3)])
			
			res1.get()
			res2.get()

def two_word_worker(index: int, data, keys: list[str], queue, cache: dict, counter: dict, numThreads: int):
	model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=f"cuda:{[0, 2, 3][index%3]}")

	print(f"Loaded two_word_worker {index}")

	for k in keys:
		while queue.qsize() > numThreads * 2:
			time.sleep(1)

		words = data[k]

		embeddings = []
		sentence_embeddings = []

		for word in words:
			if word in cache:
				embeddings.append(cache[word])
			else:
				embedding = get_embedding_in_parallel(word, model)
				cache[word] = embedding
				embeddings.append(embedding)
		
			sentence_embeddings.append(get_embedding_in_parallel(f'{k} {word}', model))

		queue.put((k, embeddings, sentence_embeddings))

	counter["count"] += 1

	if counter["count"] == numThreads:
		queue.put(None)
	
	print('two_word_worker completed')

def two_word_model_worker(index: int, queue, destination, epochs):
	#model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=f"cuda:{[2, 3][index%2]}")

	print(f"Loaded two_word_model_worker {index}")

	t = time.time()
	i = 0
	while True:
		queue_obj = queue.get()

		if not queue_obj:
			queue.put(None)
			break

		word, word_embeddings, sentence_embeddings = queue_obj

		print(f'Accepted {word} from queue, {queue.qsize()} elements in backlog')

		module = CPTensorRegression([384], 384, 100)
		
		multi_word_regression(os.path.join(destination, word), word_embeddings, sentence_embeddings, 1, module, num_epochs=epochs, sentence_dim=384, word_dim=384, lr=0.001, shuffle=True, device=[0, 2, 3][index%3])
	
		print(f'Thread {index}, {i} words parsed, {int(time.time() - t)} seconds elapsed')

		i += 1
	
	print(f'Thread {index} completed')


def build_two_word_model(src, destination, epochs):
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

	numThreads = 3

	with Pool(processes=numThreads) as p:
		res1 = p.starmap_async(two_word_worker, [(index, data, val, queue, cache, counter, numThreads) for index, val in enumerate(np.array_split(list(data.keys()), numThreads))])

		with Pool(processes=3) as p2:
			res2 = p2.starmap_async(two_word_model_worker, [(index, queue, destination, epochs) for index in range(3)])
			
			res1.get()
			res2.get()

def embedding_generator(index: int, data, queue, wordQueue, cache: dict, counter: dict, numThreads: int):
	model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=f"cuda:{[0, 2, 3][index%3]}")

	print(f"Loaded embedding_generator {index}")

	while True:
		try:
			while queue.qsize() > numThreads * 2:
				time.sleep(1)

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
					if word in cache:
						embeddings.append(cache[word])
					else:
						embedding = get_embedding_in_parallel(word, model)
						cache[word] = embedding
						embeddings.append(embedding)
				
					sentence_embeddings.append(get_embedding_in_parallel(f'{queue_obj} {word}', model))

			queue.put((queue_obj, embeddings, sentence_embeddings))
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

def model_worker(index: int, queue, destination, epochs):
	#model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=f"cuda:{[2, 3][index%2]}")

	print(f"Loaded model_worker {index}")

	t = time.time()
	i = 0
	while True:
		queue_obj = queue.get()
		t1 = time.time()

		if not queue_obj:
			print("Propagating signal")
			queue.put(None)
			queue.put(None)
			break

		word, word_embeddings, sentence_embeddings = queue_obj

		print(f'Accepted {word} from queue, {queue.qsize()} elements in backlog')

		module = CPTensorRegression([384 for _ in range(1 if not isinstance(word_embeddings[0], list) else len(word_embeddings[0]))], 384, 100)
		
		multi_word_regression(os.path.join(destination, word), word_embeddings, sentence_embeddings, 1 if not isinstance(word_embeddings[0], list) else len(word_embeddings[0]), module, num_epochs=epochs, sentence_dim=384, word_dim=384, lr=0.001, shuffle=True, device=[0, 2, 3][index%3])
	
		print(f'Thread {index}, {i} words parsed, {int(time.time() - t1)} seconds elapsed, {int(time.time() - t)} total seconds elapsed')

		i += 1
	
	print(f'Thread {index} completed')


def build_model(src: str, destination: str, epochs: int, producerThreads=3, consumerThreads=3, reverse=False):
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

	with Pool(processes=producerThreads) as p:
		res1 = p.starmap_async(embedding_generator, [(index, data, queue, wordQueue, cache, counter, producerThreads) for index in range(producerThreads)])

		with Pool(processes=consumerThreads) as p2:
			res2 = p2.starmap_async(model_worker, [(index, queue, destination, epochs) for index in range(consumerThreads)])
			
			res1.get()

			print("Completed producers")
			
			res2.get()

			print("Completed consumers")

def tense_embedding_generator(index: int, data, queue, wordQueue, cache: dict, counter: dict, numThreads: int):
	model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=f"cuda:{[0, 2, 3][index%3]}")

	print(f"Loaded embedding_generator {index}")

	while True:
		try:
			while queue.qsize() > numThreads * 2:
				time.sleep(1)

			queue_obj = wordQueue.get()

			if not queue_obj:
				break

			t = time.time()
			words = data[queue_obj]

			embeddings = []
			output_embeddings = []

			for lemma, tense in words:
				if lemma in cache:
					embeddings.append(cache[lemma])
				else:
					embedding = get_embedding_in_parallel(lemma, model)
					cache[lemma] = embedding
					embeddings.append(embedding)
			
				if tense in cache:
					output_embeddings.append(cache[tense])
				else:
					embedding = get_embedding_in_parallel(tense, model)
					cache[tense] = embedding
					output_embeddings.append(embedding)

			queue.put((queue_obj, embeddings, output_embeddings))
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

def tense_model_worker(index: int, queue, destination, epochs):
	#model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=f"cuda:{[2, 3][index%2]}")

	print(f"Loaded model_worker {index}")

	t = time.time()
	i = 0
	while True:
		queue_obj = queue.get()
		t1 = time.time()

		if not queue_obj:
			print("Propagating signal")
			queue.put(None)
			queue.put(None)
			break

		word, word_embeddings, output_embeddings = queue_obj

		print(f'Accepted {word} from queue, {queue.qsize()} elements in backlog')

		module = CPTensorRegression([384], 384, 100)
		
		multi_word_regression(os.path.join(destination, word), word_embeddings, output_embeddings, 1 if not isinstance(word_embeddings[0], list) else len(word_embeddings[0]), module, num_epochs=epochs, sentence_dim=384, word_dim=384, lr=0.001, shuffle=True, device=[0, 2, 3][index%3])
	
		print(f'Thread {index}, {i} words parsed, {int(time.time() - t1)} seconds elapsed, {int(time.time() - t)} total seconds elapsed')

		i += 1
	
	print(f'Thread {index} completed')


def build_tense_model(src: str, destination: str, epochs: int, producerThreads=3, consumerThreads=3, reverse=False):
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

	for key in (list(reversed(sorted(data.keys()))) if reverse else data.keys()):
		wordQueue.put(key)
	
	for _ in range(producerThreads):
		wordQueue.put(None)

	with Pool(processes=producerThreads) as p:
		res1 = p.starmap_async(tense_embedding_generator, [(index, data, queue, wordQueue, cache, counter, producerThreads) for index in range(producerThreads)])

		with Pool(processes=consumerThreads) as p2:
			res2 = p2.starmap_async(tense_model_worker, [(index, queue, destination, epochs) for index in range(consumerThreads)])
			
			res1.get()

			print("Completed producers")
			
			res2.get()

			print("Completed consumers")

def bert_on_bert(src, destination, epochs):
	build_trans_verb_model(src, destination, epochs)
		
if __name__ == "__main__":
	#noun_adjective_pair_regression("models/adj_weights.pt", epochs=5)
	#transitive_verb_regression("models/hybrid_weights_dummy", epochs=400)
	#concatenated_three_word_regression("models/three_word_weights.pt", epochs=10)


	# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
	#build_trans_verb_model("data/top_transitive.json", "transitive_verb_model", epochs=500)

	#build_two_word_model("data/top_aux.json", "aux_model", epochs=5000)

	# build_model("data/top_verb_nsubj.json", "intransitive_model", 5000, 10, 20)
	# build_model("data/top_adj_amod.json", "adj_model", 5000, 10, 20)
	# build_trans_verb_model("data/top_transitive.json", "transitive_model", epochs=5000)
	# build_model("data/top_adv.json", "adv_model", 5000, 6, 3)
	# build_model("data/top_pron_NOUN_poss.json", "pron_model", 5000, 6, 3)
	# build_model("data/top_verb_nsubj_PROPN_PRON_NOUN_dative_PROPN_PRON_NOUN_dobj_PROPN_PRON_NOUN.json", "ditransitive_model", 2000, 12, 3)
	# build_model("data/top_adp_prep_PROPN_PRON_NOUN_pobj_PROPN_PRON_NOUN.json", "prep_model", 5000, 6, 3)
	# build_model("data/top_adp_prep_dative_VERB_pobj_PROPN_PRON_NOUN.json", "prep_verb_model", 5000, 6, 3)
	# build_model("data/top_adp_prep_AUX_pobj_PROPN_PRON_NOUN.json", "prep_aux_model", 5000, 6, 3)
	# build_model("data/top_adp_prep_AUX_pobj_PROPN_PRON_NOUN.json", "prep_aux_model", 5000, 6, 3)
	# build_model("data/top_intj.json", "intj_model", 20000, 6, 3)
	# build_model("data/top_cconj_cc_PROPN_PRON_NOUN_conj_PROPN_PRON_NOUN.json", "cconj_noun_model", 5000, 6, 3)
	# build_model("data/top_sconj_mark_VERB_advcl_VERB.json", "sconj_model", 5000, 6, 3)
	# build_model("data/top_cconj_cc_ADJ_conj_ADJ.json", "cconj_adj_model", 5000, 6, 3)
	# build_model("data/top_cconj_cc_VERB_conj_VERB.json", "cconj_verb_model", 5000, 6, 3)
	# build_model("data/top_det_NOUN_det.json", "det_model", 5000, 20, 10)
	build_model("data/top_aux.json", "aux_model", 5000, 6, 3)
	# bert_on_bert("data/one_verb.json", "models", epochs=500)
	# build_tense_model("data/top_tense.json", "tense_model", 5000, 6, 3)

	print("Regression complete.")