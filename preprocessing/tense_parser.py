import spacy
import json
import re
import time
from multiprocessing import Process, Queue, Pool, cpu_count, Manager
import numpy as np
import contractions
import traceback
from lemminflect import getLemma, getInflection

NOUN_NUM = 50

def worker(index, data, shared: dict):
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
			if token.pos_ == "VERB":
				lemma = getLemma(token.text.lower(), upos='VERB')[0]

				if lemma not in output:
					output[lemma] = {}

				output[lemma][token.tag_] = token.text.lower()
				counter += 1
		
		if i % 1000 == 0:
			print(f'Thread {index}, {i}/{len(data)}', "sentences parsed,", counter, "matches,", int(time.time() - t), "seconds elapsed")
	
	for lemma in output:
		val = shared.get(lemma, {})

		val.update(output[lemma])
		shared[lemma] = val

	print(f'Thread {index} completed')

	return True

def parse(data: str):
	try:
		tags = ["VBD", "VBG", "VBN", "VBP", "VBZ"]

		file_in = open(data, 'r')
		file_out = open(f"data/top_tense.json", 'w')

		text = re.split(r"\.|\?|\!|\;", file_in.read())

		output = dict()

		manager = Manager()

		shared = manager.dict()

		processes = [Process(target=worker, args=pair + (shared,)) for pair in enumerate(np.array_split(text, cpu_count()))]

		for p in processes:
			p.start()
		
		for p in processes:
			p.join()
		
		for lemma, tenses in shared.items():
			for tag in tags:
				if tag not in tenses:
					inf = getInflection(lemma, tag=tag)
					
					if inf:
						tenses[tag] = inf[0]

			for tense, text in tenses.items():
				if text != lemma:
					output[tense] = output.get(tense, []) + [(lemma, text)]

		json.dump(output, file_out)

		file_out.close()
	except:
		print(traceback.format_exc())

if __name__ == "__main__":
	parse("data_raw/wikitext_textblock.txt")