import spacy
import json
import re
import time
from multiprocessing import Process, Pool, cpu_count, Manager, Queue
import numpy as np
import contractions
import traceback
from collections.abc import Iterable
import functools

NOUN_NUM = 50

class ParseObj:
	def __init__(self) -> None:
		pass
	
	def find(self, pos: str, dep: str) -> bool:
		raise NotImplementedError('Subclass did not instantiate find method')

class Conjunction(ParseObj):
	def __init__(self, *tags) -> None:
		self.tags = tags
	
	def find(self, pos: str, dep: str) -> bool:
		return all(tag.find(pos, dep) for tag in self.tags)
	
	def __str__(self):
		return "_".join(str(tag) for tag in self.tags)

class Disjunction(ParseObj):
	def __init__(self, *tags) -> None:
		self.tags = tags
	
	def find(self, pos: str, dep: str) -> bool:
		return any(tag.find(pos, dep) for tag in self.tags)
	
	def __str__(self):
		return "_".join(str(tag) for tag in self.tags)

class Not(ParseObj):
	def __init__(self, tag: ParseObj) -> None:
		self.tag = tag
	
	def find(self, pos: str, dep: str) -> bool:
		return not self.tag.find(pos, dep)
	
	def __str__(self):
		return str(self.tag)


class OutputArr(ParseObj):
	def __init__(self, *tags) -> None:
		self.tags = tags
	
	def find(self, pos: str, dep: str) -> bool:
		return any(tag.find(pos, dep) for tag in self.tags)

	def findIndex(self, pos: str, dep: str) -> int:
		return list(filter(lambda item: item[1].find(pos, dep), enumerate(self.tags)))[0][0]

	def size(self) -> int:
		return len(self.tags)
	
	def __getitem__(self, index: int):
		return self.tags[index]
	
	def __len__(self):
		return len(self.tags)
	
	def __str__(self):
		return '_'.join(str(tag) for tag in self.tags)

class Chain(ParseObj):
	def __init__(self, *tags) -> None:
		self.tags = tags
	
	def find(self, pos: str, dep: str) -> bool:
		return self.tags[0].find(pos, dep)

	def size(self) -> int:
		return len(self.tags)
	
	def __getitem__(self, key):
		if isinstance(key, slice):
			indices = range(*key.indices(len(self.tags)))
			return Chain(*[self.tags[i] for i in indices])
		return self.tags[key]
	
	def __len__(self):
		return len(self.tags)
	
	def __str__(self):
		return '_'.join(str(tag) for tag in self.tags)

class POS(ParseObj):
	def __init__(self, tag: str) -> None:
		self.tag = tag
	
	def find(self, pos: str, dep: str) -> bool:
		return self.tag == pos
	
	def __str__(self):
		return str(self.tag)
	
class DEP(ParseObj):
	def __init__(self, tag: str) -> None:
		self.tag = tag
	
	def find(self, pos: str, dep: str) -> bool:
		return self.tag == dep

	def __str__(self):
		return str(self.tag)

def worker(index, data, queue: Queue, target_pos: str, outputs: ParseObj, target=["children"]):
	t = time.time()

	nlp = spacy.load("en_core_web_sm", disable=['ner'])
	output = dict()

	targetChildren = target == "children" or target == "child"
	targetBoth = target == "both"

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

		for token in doc:
			if token.pos_ == target_pos:
				subTokens = {}
				printTokens = []
				
				if isinstance(outputs, Chain):
					out = outputs
					
					if isinstance(target, list):
						targetChildren = target[0] == "children" or target[0] == "child"
						targetBoth = target[0] == "both"

					children = [(t, token.dep_) for t in token.children]
					parents = [(t, token.dep_) for t in doc if token in t.children]

					targets = (children if targetChildren else parents if not targetBoth else parents + children)

					for i in range(len(outputs)):
						for subToken, dep in targets:
							if out.find(subToken.pos_, subToken.dep_ if subToken in children else dep):
								subTokens[i] = subToken.text.lower()
								printTokens.append(subToken)
								break
						
						children = [(t, subToken[0].dep_) for subToken in targets for t in subToken[0].children]
						parents = [(t, subToken[0].dep_) for subToken in targets for t in doc if subToken[0] in t.children]

						if i < len(outputs) - 1:
							if isinstance(target, list):
								targetChildren = target[i + 1] == "children" or target[i + 1] == "child"
								targetBoth = target[i + 1] == "both"

							targets = (children if targetChildren else parents if not targetBoth else parents + children)

							out = out[1:]
				else:
					children = [t for t in token.children]
					parents = [t for t in doc if token in t.children]

					targets = (children if targetChildren else parents if not targetBoth else parents + children)

					for subToken in targets:
						if outputs.find(subToken.pos_, subToken.dep_ if subToken in children else token.dep_):
							if isinstance(outputs, OutputArr):
								idx = outputs.findIndex(subToken.pos_, subToken.dep_ if subToken in children else token.dep_)
							else:
								idx = 0

							subTokens[idx] = subToken.text.lower()
							printTokens.append(subToken)
							counter += 1

				if isinstance(outputs, OutputArr | Chain) and len(subTokens.items()) != len(outputs):
					continue

				print(sentence, " -> ", token.text.lower(), [(t.text, t.pos_, t.dep_) for t in printTokens])

				for idx, subToken in subTokens.items():
					tok = token.text.lower()

					if tok not in output:
						output[tok] = [{} for _ in range(len(outputs) if isinstance(outputs, OutputArr | Chain) else 1)]
					
					output[tok][idx][subToken] = output[tok][idx].get(subToken, 0) + 1
		
		if i % 1000 == 0:
			print(f'Thread {index}, {i}/{len(data)}', "sentences parsed,", counter, "matches,", int(time.time() - t), "seconds elapsed")
	
	print(f'Thread {index} completed')

	queue.put(output, timeout=600)

	return True

def parse(data: str, target_pos: str, outputs: ParseObj, target="children"):
	file_in = open(data, 'r')

	text = re.split(r"\.|\?|\!|\;", file_in.read())

	tok_dict: dict[str, set[str]] = {}
	output = dict()

	queue = Queue()

	processes = [Process(target=worker, args=pair + (queue, target_pos, outputs, target)) for pair in enumerate(np.array_split(text, cpu_count()))]

	for p in processes:
		p.start()
	
	for i in range(cpu_count()):
		obj = queue.get()
		print(f"Received item {i} from queue")

		for tok, arr in obj.items():
			if tok not in output:
				output[tok] = arr
			else:
				for i in range(len(arr)):
					for tokenText in arr[i]:
						output[tok][i][tokenText] = output[tok][i].get(tokenText, 0) + arr[i][tokenText]
		
	
	for p in processes:
		p.join()
	
	queue.close()
	queue.join_thread()

	print("Output len", len(output))

	for tok in output:
		for i in range(len(output[tok])):
			sorted_out = sorted(output[tok][i].items(), key=lambda item: item[1], reverse=True)

			if len(output[tok]) == 1:
				tok_dict[tok] = set()
				
				try:
					tok_dict[tok].update(set([k for k, _ in sorted_out]))
				except:
					print(f"{traceback.format_exc()}")
			else:
				if tok not in tok_dict:
					tok_dict[tok] = [list() for _ in range(len(output[tok]))]
				
				try:
					tok_dict[tok][i] = list(set([k for k, _ in sorted_out]))
				except:
					print(f"{traceback.format_exc()}")

	json_ready_dict = { tok: list(tok_dict[tok]) for tok in output if len(tok_dict[tok]) >= 10 or (all(isinstance(item, list | set) for item in tok_dict[tok]) and functools.reduce(lambda a, b: a * b, [len(item) for item in tok_dict[tok]]) >= 10) }
	#for tok in json_ready_dict:
	#	print(tok, json_ready_dict[tok])

	with open(f"data/top_{target_pos.lower()}_{outputs}.json", 'w') as file_out:
		json.dump(json_ready_dict, file_out)

if __name__ == "__main__":
	#parse("data_raw/IMDB_Textblock.txt", "AUX", Conjunction(POS("VERB"), DEP("aux")), "parent")
	#parse("VERB", Conjunction(POS("AUX"), DEP("aux")), "child")
	#parse("AUX", Conjunction(POS("AUX"), DEP("aux")), "parent")
	# parse("VERB", OutputArr(Disjunction(DEP("nsubj"), DEP("nsubjpass")), DEP("dobj")), "child")
	# parse("data_raw/wikitext_textblock.txt", "PRON", Conjunction(POS("NOUN"), DEP("poss")), "parent")
	# parse("data_raw/wikitext_textblock.txt", "VERB", OutputArr(Conjunction(DEP("nsubj"), Disjunction(POS("PROPN"), POS("PRON"), POS("NOUN"))), Conjunction(DEP("dative"), Disjunction(POS("PROPN"), POS("PRON"), POS("NOUN"))), Conjunction(DEP("dobj"), Disjunction(POS("PROPN"), POS("PRON"), POS("NOUN")))), "child")
	# parse("data_raw/IMDB_Textblock.txt", "ADP", OutputArr(Conjunction(DEP("prep"), Disjunction(POS("PROPN"), POS("PRON"), POS("NOUN"))), Conjunction(DEP("pobj"), Disjunction(POS("PROPN"), POS("PRON"), POS("NOUN")))), "both")
	# parse("data_raw/wikitext_textblock.txt", "ADP", OutputArr(Conjunction(Disjunction(DEP("prep"), DEP("dative")), POS("VERB")), Conjunction(DEP("pobj"), Disjunction(POS("PROPN"), POS("PRON"), POS("NOUN")))), "both")
	# parse("data_raw/wikitext_textblock.txt", "ADP", OutputArr(Conjunction(Disjunction(DEP("prep")), POS("AUX")), Conjunction(DEP("pobj"), Disjunction(POS("PROPN"), POS("PRON"), POS("NOUN")))), "both")
	# parse("data_raw/wikitext_textblock.txt", "INTJ", DEP("intj"), "parent")
	# parse("data_raw/wikitext_textblock.txt", "CCONJ", Chain(Conjunction(DEP("cc"), Disjunction(POS("PROPN"), POS("PRON"), POS("NOUN"))), Conjunction(DEP("conj"), Disjunction(POS("PROPN"), POS("PRON"), POS("NOUN")))), ["parent", "child"])
	# parse("data_raw/wikitext_textblock.txt", "CCONJ", Chain(Conjunction(DEP("cc"), POS("ADJ")), Conjunction(DEP("conj"), POS("ADJ"))), ["parent", "child"])
	# parse("data_raw/wikitext_textblock.txt", "CCONJ", Chain(Conjunction(DEP("cc"), POS("VERB")), Conjunction(DEP("conj"), POS("VERB"))), ["parent", "child"])
	# parse("data_raw/wikitext_textblock.txt", "DET", Conjunction(POS("NOUN"), DEP("det")), "parent")
	parse("data_raw/wikitext_textblock.txt", "SCONJ", Chain(Conjunction(DEP("mark"), POS("VERB")), Conjunction(DEP("advcl"), POS("VERB"))), ["parent", "parent"])