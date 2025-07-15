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
from multiprocessing import shared_memory
import struct

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

def embedding_generator(index: int, data, shm_info, wordQueue, cache: dict, counter: dict, numThreads: int):
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

				print(f"{len(variations)} total variations")

				if len(variations) > 100_000:
					indices = np.random.choice(np.arange(len(variations)), 100_000, replace=False)
				
					var = list(variations)
					variations = [var[i] for i in indices]

				n = 0
				for item in variations:
					arr = []

					item = [queue_obj] + list(item)
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

			# Write to shared memory instead of queue
			write_to_shared_memory(embeddings, sentence_embeddings, shm_info, index)
			
			print(f"embedding_generator {index} took {int(time.time() - t)} seconds, {counter['count']} threads completed")
		except:
			import traceback
			print(traceback.format_exc())
			break

	counter["count"] += 1
	print('embedding_generator completed')


def write_to_shared_memory(embeddings, sentence_embeddings, shm_info, worker_id):
	"""Write embeddings to shared memory"""
	# Convert embeddings to numpy arrays
	if embeddings:
		if isinstance(embeddings[0], list):
			# Multi-dimensional embeddings
			word_emb_array = np.array(embeddings, dtype=np.float32)
		else:
			word_emb_array = np.array(embeddings, dtype=np.float32)
	else:
		word_emb_array = np.array([], dtype=np.float32)
	
	sent_emb_array = np.array(sentence_embeddings, dtype=np.float32)
	
	# Get shared memory blocks
	word_shm = shared_memory.SharedMemory(name=shm_info['word_shm_names'][worker_id])
	sent_shm = shared_memory.SharedMemory(name=shm_info['sent_shm_names'][worker_id])
	meta_shm = shared_memory.SharedMemory(name=shm_info['meta_shm_names'][worker_id])
	
	# Write word embeddings
	word_bytes = word_emb_array.tobytes()
	word_shm.buf[:len(word_bytes)] = word_bytes
	
	# Write sentence embeddings
	sent_bytes = sent_emb_array.tobytes()
	sent_shm.buf[:len(sent_bytes)] = sent_bytes
	
	# Write metadata (shapes and sizes)
	meta_data = struct.pack('QQQQ', 
		len(word_bytes), 
		len(sent_bytes),
		len(word_emb_array.shape),
		len(sent_emb_array.shape)
	)
	meta_data += struct.pack('Q' * len(word_emb_array.shape), *word_emb_array.shape)
	meta_data += struct.pack('Q' * len(sent_emb_array.shape), *sent_emb_array.shape)
	
	meta_shm.buf[:len(meta_data)] = meta_data
	
	# Signal completion
	shm_info['completion_flags'][worker_id] = True


def read_from_shared_memory(shm_info, worker_id):
	"""Read embeddings from shared memory"""
	# Get shared memory blocks
	word_shm = shared_memory.SharedMemory(name=shm_info['word_shm_names'][worker_id])
	sent_shm = shared_memory.SharedMemory(name=shm_info['sent_shm_names'][worker_id])
	meta_shm = shared_memory.SharedMemory(name=shm_info['meta_shm_names'][worker_id])
	
	# Read metadata
	meta_data = bytes(meta_shm.buf[:1024])  # Assuming metadata is < 1024 bytes
	word_size, sent_size, word_ndim, sent_ndim = struct.unpack('QQQQ', meta_data[:32])
	
	offset = 32
	word_shape = struct.unpack('Q' * word_ndim, meta_data[offset:offset + 8 * word_ndim])
	offset += 8 * word_ndim
	sent_shape = struct.unpack('Q' * sent_ndim, meta_data[offset:offset + 8 * sent_ndim])
	
	# Read word embeddings
	word_bytes = bytes(word_shm.buf[:word_size])
	word_embeddings = np.frombuffer(word_bytes, dtype=np.float32).reshape(word_shape)
	
	# Read sentence embeddings
	sent_bytes = bytes(sent_shm.buf[:sent_size])
	sentence_embeddings = np.frombuffer(sent_bytes, dtype=np.float32).reshape(sent_shape)
	
	return word_embeddings.tolist(), sentence_embeddings.tolist()


def create_shared_memory_blocks(num_workers, max_embeddings_per_worker=100000, embedding_dim=384):
	"""Create shared memory blocks for all workers"""
	shm_info = {
		'word_shm_names': [],
		'sent_shm_names': [],
		'meta_shm_names': [],
		'word_shms': [],
		'sent_shms': [],
		'meta_shms': [],
		'completion_flags': Manager().list([False] * num_workers)
	}
	
	for i in range(num_workers):
		# Estimate memory needed (conservative estimate)
		# For word embeddings: assume max 10 words per embedding, each 384 dim
		word_mem_size = max_embeddings_per_worker * 10 * embedding_dim * 4  # 4 bytes per float32
		sent_mem_size = max_embeddings_per_worker * embedding_dim * 4
		meta_mem_size = 1024  # For metadata
		
		# Create shared memory blocks
		word_shm = shared_memory.SharedMemory(create=True, size=word_mem_size)
		sent_shm = shared_memory.SharedMemory(create=True, size=sent_mem_size)
		meta_shm = shared_memory.SharedMemory(create=True, size=meta_mem_size)
		
		shm_info['word_shm_names'].append(word_shm.name)
		shm_info['sent_shm_names'].append(sent_shm.name)
		shm_info['meta_shm_names'].append(meta_shm.name)
		
		shm_info['word_shms'].append(word_shm)
		shm_info['sent_shms'].append(sent_shm)
		shm_info['meta_shms'].append(meta_shm)
	
	return shm_info


def cleanup_shared_memory(shm_info):
	"""Clean up shared memory blocks"""
	for shm in shm_info['word_shms'] + shm_info['sent_shms'] + shm_info['meta_shms']:
		try:
			shm.close()
			shm.unlink()
		except:
			pass


def build_model(src: str, destination: str, epochs: int, producerThreads=3, reverse=False):
	with open(src) as file_in:
		data = json.load(file_in)

	os.makedirs(destination, exist_ok=True)

	manager = Manager()
	cache = manager.dict()
	
	manager2 = Manager()
	counter = manager2.dict()
	counter["count"] = 0

	manager4 = Manager()
	wordQueue = manager4.Queue()

	# Create shared memory blocks
	shm_info = create_shared_memory_blocks(producerThreads)

	for key in (list(reversed(sorted(data.keys()))) if reverse else list(data.keys()))[:10]:
		wordQueue.put(key)
	
	for _ in range(producerThreads):
		wordQueue.put(None)

	word_embeddings = []
	sentence_embeddings = []

	try:
		with Pool(processes=producerThreads) as p:
			res1 = p.starmap_async(embedding_generator, [(index, data, shm_info, wordQueue, cache, counter, producerThreads) for index in range(producerThreads)])

			# Wait for all workers to complete
			while counter["count"] < producerThreads:
				time.sleep(0.1)
			
			print("All workers completed, reading from shared memory...")
			
			# Read results from shared memory
			for i in range(producerThreads):
				if shm_info['completion_flags'][i]:
					w_e, s_e = read_from_shared_memory(shm_info, i)
					word_embeddings += w_e
					sentence_embeddings += s_e
					print(f"Read embeddings from worker {i}")

			res1.get()
			p.terminate()
			p.join()
			
	finally:
		# Clean up shared memory
		cleanup_shared_memory(shm_info)
		
	print("Length:", len(word_embeddings[0]))
	module = CPTensorRegression([384 for _ in range(1 if not isinstance(word_embeddings[0], list) else len(word_embeddings[0]))], 384, 100)
		
	batch_word_regression(os.path.join(destination, "model"), word_embeddings, sentence_embeddings, 1 if not isinstance(word_embeddings[0], list) else len(word_embeddings[0]), module, num_epochs=epochs, sentence_dim=384, word_dim=384, lr=0.001, shuffle=True, device=2)

if __name__ == "__main__":
	#build_model("data/top_adj_amod.json", "general_adj_model", 200, 20)
	#build_model("data/top_verb_nsubj.json", "general_intransitive_model", 200, 20)
	#build_model("data/top_transitive.json", "general_transitive_model", 200, 20)
	build_model("data/top_verb_nsubj_PROPN_PRON_NOUN_dative_PROPN_PRON_NOUN_dobj_PROPN_PRON_NOUN.json", "general_ditransitive_model", 200, 10)
	# build_model("data/top_adv_advmod.json", "general_adv_model", 5000, 200, 20)
	print("Regression complete.")