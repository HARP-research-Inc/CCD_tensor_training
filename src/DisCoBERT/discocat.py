from .pos import *
from .categories import *
from ..temporal_spacy.temporal_parsing import SUBORDINATING_CONJUNCTIONS

import torch
import torch.nn.functional as F
import time

from src.regression import TwoWordTensorRegression
import re
from src.regression import CPTensorRegression, TwoWordTensorRegression


###############################
###### PARSING FUNCTIONS ######
###############################

MODEL_PATH = "/mnt/ssd/user-workspaces/aidan-svc/CCD_tensor_training/"
CONJUNCTION_LIST = SUBORDINATING_CONJUNCTIONS["temporal"] | SUBORDINATING_CONJUNCTIONS["causal"] | \
	SUBORDINATING_CONJUNCTIONS["conditional"] | SUBORDINATING_CONJUNCTIONS["concessive"] | \
	SUBORDINATING_CONJUNCTIONS["purpose"] | SUBORDINATING_CONJUNCTIONS["result/consequence"] | \
	SUBORDINATING_CONJUNCTIONS["comparison"] | SUBORDINATING_CONJUNCTIONS["manner"] | \
	SUBORDINATING_CONJUNCTIONS["relative (nominal)"] | SUBORDINATING_CONJUNCTIONS["exception"]# |\
	#{"and", "but", "or", "nor", "for", "so", "yet", "either", "neither", "and/or"}

PUNCTUATION_DELIMS = {",", ".", "!", "?", ";", ":"}

def parse_driver(circuit: Circuit, parent: Box, leaves: list, token: spacy.tokens.Token, factory: Box_Factory, doc, levels: dict, level: int):
	"""
	Parameter field names indicate the parent/child relationship in reference
	to the tree structure, NOT the circuit structure. 
	"""
	if level == 0:
		circuit.set_root(parent)
	
	print("parse driver", token.text)

	pos = token.pos_
	
	child_box = factory.create_box(token, pos)

	#print(pos, type(child_box))

	#traversal is in the opposite direction of the tree.
	circuit.add_wire(child_box, parent) # order swapped from tree traversal order

	if(token.n_lefts == 0 and token.n_rights == 0):
		#base case
		leaves.append(child_box)
	
	if level in levels:
		levels[level].append(child_box)
	else:
		levels[level] = [child_box]
	
	for child in token.children:
		#print(token.text, child.text)
		parse_driver(circuit, child_box, leaves, child, factory, doc, levels, level + 1)

def tree_parse(circuit: Circuit, string, spacy_model: spacy.load, factory: Box_Factory, levels: dict, source: Box = None):
	"""
	Parsing traversal order should be in the opposite direction of the circuit.
	Parameter field names indicate the parent/child relationship in reference
	to the tree structure, NOT the circuit structure. 

	args:
		source: source box for the whole tree (in the prototype, it is a composer spider)
	"""
	print("tree parse", string)
	doc = spacy_model(string)
	
	"""for token in doc:
		found = False
		for childA in [child for child in token.children if child.dep_ == "advcl"]:
			for childB in [child for child in childA.children if child.dep_ == "mark"]:
				prevTokenHead = token.head

				token.head = childB
				childA.head = childB
				childB.head = prevTokenHead if prevTokenHead != token else childB.head

				found = True
				break
			
			if found:
				break"""

	for token in doc:
		found = False
		for childA in [child for child in token.children if child.dep_ == "conj"]:
			for childB in [child for child in token.children if child.dep_ == "cc"]:
				prevTokenHead = token.head

				token.head = childB
				childA.head = childB
				childB.head = prevTokenHead if prevTokenHead != token else childB.head

				found = True
				break
			
			if found:
				break

	# Flip prep relationships directly in the spacy structure
	for token in doc:
		if token.pos_ == "VERB":
			prep_children = [child for child in token.children if child.dep_ == "prep"]
			if len(prep_children) > 1:
				prep_children = sorted(prep_children, key=lambda t: t.i)
				for i in range(1, len(prep_children)):
					prep_children[i].head = prep_children[i - 1]
	
	# Flip prep relationships directly in the spacy structure
	for token in doc:
		if token.dep_ == "prep":
			prep_token = token
			original_head = token.head

			print(prep_token.text, original_head.text)

			prep_token.head = prep_token
			original_head.head = prep_token
	
	root = [token for token in doc if token.head == token][0]
	print("root", root)

	leaves = list()

	parse_driver(circuit, source, leaves, root, factory, doc, levels, 0)

	return leaves

def tree_parse_old(circuit: Circuit, string, spacy_model: spacy.load, factory: Box_Factory, levels: dict, source: Box = None):
	"""
	Parsing traversal order should be in the opposite direction of the circuit.
	Parameter field names indicate the parent/child relationship in reference
	to the tree structure, NOT the circuit structure. 

	args:
		source: source box for the whole tree (in the prototype, it is a composer spider)
	"""
	print("tree parse", string)
	doc = spacy_model(string)
	root = [token for token in doc if token.head == token][0]
	print("root", root)

	leaves = list()

	parse_driver(circuit, source, leaves, root, factory, levels, 0)

	return leaves


def split_clauses_with_markers(sentence, nlp: spacy.load):
	# Build regex for conjunctions (prioritize multi-word)
	sorted_conjs = sorted(CONJUNCTION_LIST, key=lambda x: -len(x))
	escaped_conjs = [r'\b' + re.escape(conj) + r'\b' for conj in sorted_conjs]
	conj_pattern = '|'.join(escaped_conjs)
	
	# Build regex for punctuation
	punct_pattern = '|'.join(re.escape(p) for p in PUNCTUATION_DELIMS)

	# Combined pattern: capture all splitters
	pattern = r'\s*(%s|%s)\s*' % (conj_pattern, punct_pattern)

	# Split and keep delimiters
	parts = re.split(pattern, sentence)

	# Group into clauses and splitters
	clauses = parts[::2]
	markers = parts[1::2]

	# Clean up
	clauses = [c.strip() for c in clauses if c.strip()]
	markers = [m.strip() for m in markers if m.strip()]

	return clauses, markers

def driver(discourse: str, nlp: spacy.load):
	"""
	returns: circuit object containing circuit reprsenentation of the discourse.

	"""
	clauses, conjunctions = split_clauses_with_markers(discourse.lower(), nlp)

	factory = Box_Factory(nlp, MODEL_PATH)

	circuit = Circuit("*****DISCOURSE*****")

	# Create a root box for the circuit
	root_box = factory.create_box(None, "bureaucrat")

	# Composer box to combine clauses
	composer = factory.create_box(None, "spider")

	for i, clause in enumerate(clauses):
		#print("CLAUSE", i+1, ":", clause)
		new_circuit = Circuit(f"Clause {i+1}")

		levels = {}

		sources = tree_parse(new_circuit, clause, nlp, factory, levels, composer)

		new_circuit.set_sources(sources)

		new_circuit.set_levels(list(levels.values()))

		#print("Sources:", [source.get_label() for source in sources])

		#print(new_circuit.root)

		circuit.concactenate(new_circuit)
	
	circuit.add_wire(composer, root_box)

	return root_box, circuit

if __name__ == "__main__":

	#version 0.1.0 - bag of clauses approach

	path_to_models = "/mnt/ssd/user-workspaces/aidan-svc/CCD_tensor_training"
	spacy_model = "en_core_web_lg"

	one_clause = "the big fat deformed french man eats a small helpless newborn baby"
	one_clause2 = "small dog eats big man"
	annoying = "she should have been being watched carefully"

	nlp = spacy.load(spacy_model)

	dummy = Category("blank")

	dummy.set_nlp(nlp)
	

	from nltk import Tree

	doc = nlp("accuracy was increased by repeating the test. the dogs among the men")

	def to_nltk_tree(node):
		if node.n_lefts + node.n_rights > 0:
			return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
		else:
			return node.orth_


	[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]


	ref, discourse = driver("The dogs among the men.", nlp)
	ref2, discourse2 = driver("the french freak quickly ate the baby", nlp)
	ref3, discourse3 = driver("accuracy was increased by repeating the test", nlp)

	ref4, discourse4 = driver("the accuracy was increased by repeating the test", nlp)
	embedding4 = discourse4.forward()

	ref5, discourse5 = driver("if you go through the door, you will find a wonderful treasure", nlp)
	embedding5 = discourse5.forward()

	example_sentences = [
		"i had eaten",
		#"his face was extremely ugly",
		"the book is on the table",
		#"she walked through the park in the morning",
		#"he sat beside his friend during the movie",
		"they arrived after the meeting had started",
		# "the keys are under the couch",
		#"we met at the coffee shop near the station",
		"he jumped over the fence quickly",
		"the cat hid behind the curtain",
		"she poured milk into the glass",
		"the painting hangs above the fireplace",
		"i ate some rice and beans"
	]

	for ex in example_sentences:
		r, d = driver(ex, nlp)
		emb = d.forward()

		print("Similarity: ", F.cosine_similarity(emb[1], Box.model_cache.retrieve_BERT(ex), dim=1))

	#print(discourse)
	#print(discourse2)
	#print(discourse3)


	"""embedding2 = discourse2.forward()

	print(embedding2)

	print(F.cosine_similarity(embedding2[1], Box.model_cache.retrieve_BERT("the french freak quickly ate the baby"), dim=1))

	embedding = discourse.forward()

	print(F.cosine_similarity(embedding[1], Box.model_cache.retrieve_BERT("the dogs among the men"), dim=1))

	print(F.cosine_similarity(embedding[1], embedding2[1], dim=1))

	print(F.cosine_similarity(Box.model_cache.retrieve_BERT("the french freak quickly ate the baby"), Box.model_cache.retrieve_BERT("the dogs among the men"), dim=1))

	print(type(embedding))
	embedding3 = discourse3.forward()

	print(F.cosine_similarity(embedding3[1], embedding2[1], dim=1))
	print(F.cosine_similarity(embedding3[1], embedding[1], dim=1))
	print(F.cosine_similarity(embedding3[1], embedding4[1], dim=1))
	print(F.cosine_similarity(embedding3[1], Box.model_cache.retrieve_BERT("accuracy was increased by repeating the test"), dim=1))
"""
	# start_time = time.time()
	# for i in range(1000):
	#	 print("DisCoBERT iteration:", i)
	#	 _, _ = driver("I eat food", nlp)
	# end_time = time.time()

	# DCBERT_time = end_time - start_time

	# start_time = time.time()
	# for i in range(1000):
	#	 print("SBERT iteration:", i)
	#	 Box.model_cache.retrieve_BERT("I eat food")
	# end_time = time.time()

	# SBERT_time = end_time - start_time

	# print("DisCoBERT time:", DCBERT_time)
	# print("SBERT time:", SBERT_time)
	

