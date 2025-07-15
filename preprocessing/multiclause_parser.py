import spacy
import json
import re
import time
import multiprocessing as mp
from multiprocessing import Process, Queue, Pool, cpu_count
import numpy as np
import contractions
import traceback

mp.set_start_method('spawn', force=True)

NOUN_NUM = 50


def worker(index, data, queue: Queue, target_pos: str):
	start_t = time.time()

	nlp = spacy.load("en_core_web_sm", disable=['ner'])
	output = dict()

	counter = 0

	for i, sentence in enumerate(data):
		if len(sentence.strip()) == 0:
			continue
		try:
			sentence = contractions.fix(sentence).replace("(", " ").replace(")", " ")
		except:
			print(traceback.format_exc())
			print(sentence)
			continue

		doc = nlp(str(sentence))

		for n, token in enumerate(doc):
			if any(child.dep_ == "cc" and child.pos_ == target_pos for child in token.children) and any(child.dep_ == "conj" for child in token.children):
				cc = [child for child in token.children if child.dep_ == 'cc'][0]
				conj = [child for child in token.children if child.dep_ == 'conj'][0]

				if token.text.lower() not in output:
					output[token.text.lower()] = []
				
				token_indices = []
				conj_indices = []
				for k, t in enumerate(doc):
					if token.is_ancestor(t) or t == token and not conj.is_ancestor(t) and not cc.is_ancestor(t):
						token_indices.append(k)
					if conj.is_ancestor(t) or t == conj and not token.is_ancestor(t) and not cc.is_ancestor(t):
						conj_indices.append(k)
				
				token_indices.sort()
				conj_indices.sort()


				if len(token_indices) > 0 and len(conj_indices) > 0:
					output[token.text.lower()].append((' '.join(doc[k].text.lower() for k in token_indices), ' '.join(doc[k].text.lower() for k in conj_indices)))
					counter += 1
					print(f'{counter} {f' <{token.text.lower()}> '.join(output[token.text.lower()])}')
		
		if i % 1000 == 0:
			print(f'Thread {index}, {i}/{len(data)}', "sentences parsed,", counter, "matches,", int(time.time() - start_t), "seconds elapsed")
	
	queue.put(output)
	
	print(f'Thread {index} completed')

	return True

def parse(data: str, target_pos: str):
	print("Loading file...")
	file_in = open(data, 'r')

	text = re.split(r"\.|\?|\!|\;", file_in.read())

	print("Loaded!")

	output_dict: dict[str, set[str]] = {}
	output = dict()

	queue = Queue()

	processes = [Process(target=worker, args=pair + (queue, target_pos,)) for pair in enumerate(np.array_split(text, cpu_count()))]

	for p in processes:
		p.start()

	print("Processes started")
	
	for i in range(cpu_count()):
		obj = queue.get()
		print(f"Received item {i} from queue")

		for token, arr in obj.items():
			output[token] = output.get(token, []) + arr
	
	for p in processes:
		p.join()
	
	queue.close()
	queue.join_thread()

	for token in output:
		if token not in output_dict:
			output_dict[token] = set()
			
		output_dict[token].update(set(output[token]))

	json_ready_dict = { token: list(output_dict[token]) for token in output if len(output_dict[token]) > 10 }
	for token in json_ready_dict:
		print(token, json_ready_dict[token])
	
	print("Writing...")
	file_out = open(f"data/top_{target_pos.lower()}.json", 'w')
	json.dump(json_ready_dict, file_out)

	print("Written out to", f"data/top_{target_pos.lower()}.json")

	file_out.close()

if __name__ == "__main__":
	#parse("AUX", Conjunction(POS("VERB"), DEP("aux")), "parent")
	#parse("VERB", Conjunction(POS("AUX"), DEP("aux")), "child")
	#parse("AUX", Conjunction(POS("AUX"), DEP("aux")), "parent")
	#parse("ADV")
	#parse("data_raw/wikitext_textblock.txt", "INTJ")
	parse("data_raw/wikitext_textblock.txt", "CCONJ")