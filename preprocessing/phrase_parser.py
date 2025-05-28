import spacy
import json
import re
import time
from multiprocessing import Process, Queue, Pool, cpu_count
import numpy as np
import contractions
import traceback

NOUN_NUM = 50


def worker(index, data, queue: Queue, target_pos: str):
	t = time.time()

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
			if token.pos_ == target_pos:
				if token.text.lower() not in output:
					output[token.text.lower()] = []
				
				if len(doc[n+1:]) > 0:
					output[token.text.lower()].append(' '.join(t.text.lower() for t in doc[n+1:]))
					counter += 1
		
		if i % 1000 == 0:
			print(f'Thread {index}, {i}/{len(data)}', "sentences parsed,", counter, "matches,", int(time.time() - t), "seconds elapsed")
	
	queue.put(output)
	
	print(f'Thread {index} completed')

	return True

def parse(data: str, target_pos: str):
	file_in = open(data, 'r')
	file_out = open(f"data/top_{target_pos.lower()}.json", 'w')

	text = re.split(r"\.|\?|\!|\;", file_in.read())

	output_dict: dict[str, set[str]] = {}
	output = dict()

	queue = Queue()

	processes = [Process(target=worker, args=pair + (queue, target_pos,)) for pair in enumerate(np.array_split(text, cpu_count()))]

	for p in processes:
		p.start()
	
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
		
	json.dump(json_ready_dict, file_out)

	file_out.close()

if __name__ == "__main__":
	#parse("AUX", Conjunction(POS("VERB"), DEP("aux")), "parent")
	#parse("VERB", Conjunction(POS("AUX"), DEP("aux")), "child")
	#parse("AUX", Conjunction(POS("AUX"), DEP("aux")), "parent")
	#parse("ADV")
	parse("data_raw/wikitext_textblock.txt", "INTJ")