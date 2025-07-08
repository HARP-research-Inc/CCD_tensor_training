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
CONJUNCTION_LIST = set()#SUBORDINATING_CONJUNCTIONS["temporal"] | SUBORDINATING_CONJUNCTIONS["causal"] | \
	#SUBORDINATING_CONJUNCTIONS["conditional"] | SUBORDINATING_CONJUNCTIONS["concessive"] | \
	#SUBORDINATING_CONJUNCTIONS["purpose"] | SUBORDINATING_CONJUNCTIONS["result/consequence"] | \
	#SUBORDINATING_CONJUNCTIONS["comparison"] | SUBORDINATING_CONJUNCTIONS["manner"] | \
	#SUBORDINATING_CONJUNCTIONS["exception"]# | SUBORDINATING_CONJUNCTIONS["relative (nominal)"] |
	#{"and", "but", "or", "nor", "for", "so", "yet", "either", "neither", "and/or"}

PUNCTUATION_DELIMS = {".", "!", "?", ";", ":"}

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

	print(pos, type(child_box))

	#traversal is in the opposite direction of the tree.
	circuit.add_wire(child_box, parent) # order swapped from tree traversal order

	if(token.n_lefts == 0 and token.n_rights == 0):
		#base case
		leaves.append(child_box)
	
	if level in levels:
		levels[level].append(child_box)
	else:
		levels[level] = [child_box]
	
	for child in get_children(token):
		#print(token.text, child.text)
		parse_driver(circuit, child_box, leaves, child, factory, doc, levels, level + 1)

def get_children(token):
	return [t for t in token.doc if t.head == token if t != token]

def flip(doc, relation, where: lambda token: True):
	alreadyParsed = set()

	while True:
		flipped = False
		root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
		print("root", root)
		for token in doc:
			print("-", token.text)
			if token.dep_ == relation and token not in alreadyParsed and where(token):
				prev_token = token
				prev_token_dep = token.dep_
				original_head = token.head
				original_head_dep = original_head.dep_
				original_head_head = original_head.head

				original_root = original_head_head == original_head

				"""print(f"\nBEFORE FLIP:")
				print(f"  {prev_token.text} -> {prev_token.head.text} (dep: {prev_token.dep_})")
				print(f"  {original_head.text} -> {original_head.head.text} (dep: {original_head.dep_})")
				print(f"  {original_head.text} children: {[c.text for c in get_children(original_head)]}")


				print(f"\nALL TOKENS BEFORE FLIP:")
				for t in doc:
					children_texts = [c.text for c in get_children(t)]
					print(f"  {t.text}: children={children_texts} subtree={list(tok.text for tok in t.subtree)}")
					
					# Verify children actually point back to this token
					for child in get_children(t):
						if child.head != t:
							print(f"    WARNING: {child.text} is child of {t.text} but points to {child.head.text}")"""

				prev_token.dep_ = original_head_dep
				original_head.dep_ = prev_token_dep
				
				prev_token.head = prev_token if original_root else original_head_head
				original_head.head = prev_token

				"""print(f"\nAFTER FLIP:")
				print(f"  {prev_token.text} -> {prev_token.head.text} (dep: {prev_token.dep_})")
				print(f"  {original_head.text} -> {original_head.head.text} (dep: {original_head.dep_})")
				print(f"  {original_head.text} children: {[c.text for c in get_children(original_head)]}")
				
				# Check if any token lost children unexpectedly
				print(f"\nALL TOKENS AFTER FLIP:")
				for t in doc:
					children_texts = [c.text for c in get_children(t)]
					print(f"  {t.text}: children={children_texts} subtree={list(tok.text for tok in t.subtree)}")

					for child in doc:
						if child.head == t:
							print(child.text, "is child of", t.text, t.is_ancestor(child), child in get_children(t))
					
					# Verify children actually point back to this token
					for child in get_children(t):
						if child.head != t:
							print(f"    WARNING: {child.text} is child of {t.text} but points to {child.head.text}")"""
								
				root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
				print("root", root)

				to_nltk_tree(root).pretty_print()
				
				alreadyParsed.add(original_head)
				alreadyParsed.add(prev_token)
				flipped = True
				break

		if not flipped:
			break
	
def exchange(doc, childCase: lambda token: True, parentCase: lambda token: True):
	alreadyParsed = set()

	for token in doc:
		if token not in alreadyParsed and childCase(token) and parentCase(token.head):
			prev_token = token
			prev_token_dep = token.dep_
			original_head = token.head
			original_head_dep = original_head.dep_

			print("exchanged", prev_token.text, original_head.text)
			print(prev_token_dep, original_head_dep)

			prev_token.head = prev_token if original_head.head == original_head else original_head.head
			prev_token.dep_ = original_head_dep
			original_head.head = prev_token
			original_head.dep_ = prev_token_dep
			alreadyParsed.add(original_head)

			for t in get_children(prev_token):
				if t != original_head:
					print(t.text, "now has parent", original_head.text)
					t.head = original_head

def rewire(doc, relation, childCase: lambda token: True, parentCase: lambda token: True):
	alreadyParsed = set()

	for token in doc:
		if token.dep_ == relation and token not in alreadyParsed and childCase(token.head) and parentCase(token.head.head):
			prev_token = token
			prev_token_dep = token.dep_
			original_head = token.head.head
			original_head_dep = original_head.dep_

			print("flipped", prev_token.text, original_head.text)
			print(prev_token_dep, original_head_dep)

			prev_token.head = prev_token if original_head.head.head == original_head.head else original_head.head.head
			prev_token.dep_ = original_head_dep
			original_head.head = prev_token
			original_head.dep_ = prev_token_dep
			alreadyParsed.add(original_head)	

def rearrange(doc, relation, relationRoot, rootPOS=None, multiLevel=False, replacePOS=None, sourcePOS=None):
	alreadyParsed = set()

	if isinstance(relation, str):
		relation = [relation]
	if isinstance(relationRoot, str):
		relationRoot = [relationRoot]

	for token in doc:
		for childA in [child for child in get_children(token) if any(child.dep_ in rel for rel in relation)]:
			for childB in [child for child in (get_children(childA) if multiLevel else get_children(token)) if any(child.dep_ in rel for rel in relationRoot) and (rootPOS is None or child.pos_ == rootPOS)]:
				print(token.text, childA.text, childB.text, token in alreadyParsed)	
		if token in alreadyParsed:
			continue

		if sourcePOS and token.pos_ != sourcePOS:
			continue
	
		found = False

		for childA in [child for child in get_children(token) if any(child.dep_ in rel for rel in relation)]:
			for childB in [child for child in (get_children(childA) if multiLevel else get_children(token)) if any(child.dep_ in rel for rel in relationRoot) and (rootPOS is None or child.pos_ == rootPOS)]:
				prevTokenHead = token.head
				prevTokenDep = token.dep_

				token.head = childB
				childA.head = childB
				childB.head = prevTokenHead if prevTokenHead != token else childB
				token.dep_ = childB.dep_
				childB.dep_ = prevTokenDep

				if replacePOS:
					childB.pos_ = replacePOS

				alreadyParsed.add(childB)

				print("Rearranged", token.text, childA.text, childB.text)

				#found = True
				#break
			
			#if found:
			#	break

def rearrangeRoot(doc, relation, relationRoot, rootPOS=None, multiLevel=False, replacePOS=None, sourcePOS=None):
	alreadyParsed = set()

	if isinstance(relation, str):
		relation = [relation]
	if isinstance(relationRoot, str):
		relationRoot = [relationRoot]

	for token in doc:
		for childA in [child for child in get_children(token) if any(child.dep_ in rel for rel in relation)]:
			for childB in [child for child in (get_children(childA) if multiLevel else get_children(token)) if any(child.dep_ in rel for rel in relationRoot) and (rootPOS is None or child.pos_ == rootPOS)]:
				print(token.text, childA.text, childB.text, token in alreadyParsed)	
		if token in alreadyParsed:
			continue

		if sourcePOS and token.pos_ != sourcePOS:
			continue
	
		found = False
		
		root = token
		while root.head != root:
			root = root.head

		for childA in [child for child in get_children(token) if any(child.dep_ in rel for rel in relation)]:
			for childB in [child for child in (get_children(childA) if multiLevel else get_children(token)) if any(child.dep_ in rel for rel in relationRoot) and (rootPOS is None or child.pos_ == rootPOS)]:
				prevTokenHead = root.head
				prevTokenDep = root.dep_

				root.head = childB
				childA.head = childB
				childB.head = prevTokenHead if prevTokenHead != root else childB
				token.dep_ = childB.dep_
				childB.dep_ = prevTokenDep

				if replacePOS:
					childB.pos_ = replacePOS

				alreadyParsed.add(childB)

				print("Rearranged", root.text, childA.text, childB.text)



from nltk import Tree

def to_nltk_tree(node):
	if len(get_children(node)) > 0:
		return Tree(node.orth_, [to_nltk_tree(child) for child in get_children(node)])
	else:
		return node.orth_
	
def tree_parse(circuit: Circuit, string, spacy_model: spacy.load, factory: Box_Factory, levels: dict, source: Box = None):
	"""
	Parsing traversal order should be in the opposite direction of the circuit.
	Parameter field names indicate the parent/child relationship in reference
	to the tree structure, NOT the circuit structure. 

	args:
		source: source box for the whole tree (in the prototype, it is a composer spider)
	"""
	string = string.replace("-", "")
	print("tree parse", string)
	doc = spacy_model(string)

	remove = set()
	for token in doc:
		if re.match(r'^[^A-Za-z]$', token.text):
			remove.add(token)
			token.head = token
			token.dep_ = "removed"


	# doc = [token for token in doc if token not in remove]
	

	for token in doc:
		print(token.pos_, token.text)
		for child in get_children(token):
			print(">>>", child.dep_, child.text)
	print()

	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	to_nltk_tree(root).pretty_print()

	remove = set()
	"""for token in doc:
		for child in get_children(token):
			print(token.text, child.dep_, child.text)
			
		if token.dep_ == "relcl" and token.pos_ != "SCONJ":
			print(token.text, "is a relcl")
			if len(list(token.lefts)) > 0:
				child = list(token.lefts)[0]
				dep = child.dep_

				if child.pos_ != "SCONJ":
					remove.add(child)
					child.head = child
				
					if token.pos_ == "VERB" and token.head.head != token.head:
						print(token.head.text, "becomes child of", token.text, "and", token.text, "becomes child of", token.head.head.text)
						prevHead = token.head.head
						token.head.head = token
						if dep != "aux":
							token.head.dep_ = dep
						token.head = prevHead"""
		
		#if token.dep_ == "mark" and token.pos_ == "SCONJ":
		#	remove.add(token)
		#	child.head = child
	
	"""print(remove)
	for token in doc:
		for child in get_children(token):
			print(token.text, child.dep_, child.text)
	doc = [token for token in doc if token not in remove]"""
	
	"""for token in doc:
		found = False
		for childA in [child for child in get_children(token) if child.dep_ == "advcl"]:
			for childB in [child for child in get_children(childA) if child.dep_ == "mark"]:
				prevTokenHead = token.head

				token.head = childB
				childA.head = childB
				childB.head = prevTokenHead if prevTokenHead != token else childB.head

				found = True
				break
			
			if found:
				break"""


	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	to_nltk_tree(root).pretty_print()


	print("\n\n<<< CONJUNCTION FIXING >>>\n\n")

	for token in doc:
		if token.dep_ == "conj" or token.dep_ == "punct":
			has_cc_sibling = any(sib.dep_ == "cc" for sib in get_children(token.head) if sib is not token)

			if not has_cc_sibling:
				for candidate in doc:
					if candidate.dep_ == "cc":
						has_conj_sibling = any(sib.dep_ == "conj" for sib in get_children(candidate.head) if sib is not candidate)

						if not has_conj_sibling:
							if candidate.head != token and token.head != candidate:
								token.head = candidate.head
								token.dep_ = "conj"
								break

	
	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	to_nltk_tree(root).pretty_print()

	print("\n\n<<< CLAUSE POS MODIFICATION >>>\n\n")

	rearrange(doc, ["relcl"], ["nsubjpass", "nsubj"], "PRON", True, "SCONJ")

	for token in doc:
		for child in get_children(token):
			print(token.text, child.dep_, child.text)

	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	to_nltk_tree(root).pretty_print()

	#rearrange(doc, ["prep"], ["pobj"], "SCONJ", True)

	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	to_nltk_tree(root).pretty_print()

	rewire(doc, "prep", lambda child: child.dep_ == "advmod" and child.pos_ == "ADV", lambda parent: parent.pos_ == "VERB")

	for token in doc:
		print(token.pos_, token.text)
		for child in get_children(token):
			print(">>>", child.dep_, child.text)
		print()

	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	to_nltk_tree(root).pretty_print()

	print("\n\n<<< CLAUSE REORDERING >>>\n\n")

	rearrange(doc, ["advcl", "ccomp", "relcl"], ["mark", "advmod", "nsubjpass", "nsubj"], "SCONJ", True)
	
	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	to_nltk_tree(root).pretty_print()

	rearrange(doc, "nsubjpass", "auxpass")

	rearrange(doc, "xcomp", "aux", "PART", True, "ADP", "VERB")

	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	to_nltk_tree(root).pretty_print()

	print("\n\n<<< CONJUNCTION MODIFICATION >>>\n\n")

	explored = set()
	for token in doc:
		if token in explored:
			continue
		
		elems = [token]
		nextParse = [token]
		coordinator = None

		while True:
			expanded = False

			targets = nextParse
			nextParse = []

			for targetElem in targets:
				hasConj = False
				for child in get_children(targetElem):
					if child.dep_ == "conj":
						hasConj = True
				
				for child in get_children(targetElem):
					if child.dep_ == "conj" or (hasConj and child.dep_ == "dobj"):
						elems.append(child)
						nextParse.append(child)
						expanded = True
					if child.dep_ == "cc":
						if coordinator:
							break

						coordinator = child

			if not expanded or coordinator:
				break
		
		if coordinator:
			print(coordinator.text, "has", [elem.text for elem in elems], "children")
			prevTokenHead = elems[0].head
			prevTokenDep = elems[0].dep_

			for elem in elems:
				elem.head = coordinator
				elem.dep_ = "conj"
			
			coordinator.head = prevTokenHead if prevTokenHead != elems[0] else coordinator
			coordinator.dep_ = prevTokenDep
			explored.add(coordinator)

				
	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	to_nltk_tree(root).pretty_print()
	
	rearrange(doc, "conj", "cc")

	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	to_nltk_tree(root).pretty_print()

	#flip(doc, "advmod", lambda token: token.pos_ == "SCONJ" and token.dep_ == "advmod")
	
	"""for token in doc:
		found = False
		alreadyParsed = set()
		
		for childA in [child for child in get_children(token) if child.dep_ == "advcl" or child.dep_ == "ccomp"]:
			for childB in [child for child in get_children(childA) if child.dep_ == "mark" and child.pos_ == "SCONJ"]:
				prevTokenHead = token.head
				prevTokenDep = token.dep_

				token.head = childB
				childA.head = childB
				childB.head = prevTokenHead if prevTokenHead != token else childB
				token.dep_ = childB.dep_
				childB.dep_ = prevTokenDep

				print("Found", token.text, childA.text, childB.text)

				found = True
				break
			
			if found:
				break"""
	
	"""for token in doc:
		if token.dep_ == "prep":
			prep_token = token
			original_head = token.head

			print(prep_token.text, original_head.text)

			prep_token.head = prep_token if original_head.head == original_head else original_head.head
			original_head.head = prep_token"""

	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	to_nltk_tree(root).pretty_print()

	#flip(doc, "relcl")
	
	print("\n\n<<< PREP FLIP >>>\n\n")

	flip(doc, "prep", lambda token: len(list(get_children(token))) > 0 and token.pos_ == "ADP" and not token.head.pos_ == "ADP")
	flip(doc, "agent", lambda token: len(list(get_children(token))) > 0 and token.pos_ == "ADP" and not token.head.pos_ == "ADP")

	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	to_nltk_tree(root).pretty_print()

	for token in doc:
		if token.pos_ == "ADP":
			aux_children = [child for child in get_children(token) if child.pos_ == "AUX"]
			for child in aux_children:
				child.pos_ = "VERB"

	for token in doc:
		for child in get_children(token):
			print(token.text, child.dep_, child.text)
	
	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	to_nltk_tree(root).pretty_print()

	#rearrange(doc, ["advmod"], ["advmod"], "SCONJ")
	flip(doc, "advmod", lambda token: token.pos_ == "SCONJ")

	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	to_nltk_tree(root).pretty_print()
	print("\n\n<<< POS FIXING >>>\n\n")

	for token in doc:
		if token.pos_ == "NOUN":
			prep_children = [child for child in get_children(token) if child.dep_ == "prep"]
			if len(prep_children) > 1:
				prep_children = sorted(prep_children, key=lambda t: t.i)
				for i in range(1, len(prep_children)):
					prep_children[i].head = prep_children[i - 1]
		elif token.pos_ == "VERB":
			prep_children = [child for child in get_children(token) if child.dep_ == "ccomp"]
			if len(prep_children) > 1:
				prep_children = sorted(prep_children, key=lambda t: t.i)
				for i in range(1, len(prep_children)):
					prep_children[i].head = prep_children[i - 1]
		elif token.pos_ == "CCONJ" or token.pos_ == "SCONJ":
			print(token.text, "has children", [child.pos_ for child in get_children(token)])
			adv_children = [child for child in get_children(token) if child.pos_ == "ADV"]
			for child in adv_children:
				child.pos_ = "NOUN"

			if any(child.pos_ == "ADJ" for child in get_children(token)):
				for child in get_children(token):
					if child.pos_ == "NOUN":
						child.pos_ = "ADJ"

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
	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
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
	#pattern = r'\s*(%s|%s)\s*' % (conj_pattern, punct_pattern)
	pattern = r'\s*(%s)\s*' % (punct_pattern)

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
	clauses, conjunctions = split_clauses_with_markers(discourse, nlp)

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

	doc = nlp("his face was repulsive to look at as a result of his neglectful upbringing")

	def to_nltk_tree(node):
		if node.n_lefts + node.n_rights > 0:
			return Tree(node.orth_, [to_nltk_tree(child) for child in get_children(node)])
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
		"The player may play as any nation in the world in the 1936 or 1939 start dates in single-player or multiplayer.",
		#"I co-authored Quantum in Pictures, with Stefano Gogioso, which does the same, but now accessible to people with no maths background.",
		# "Each state has a certain amount of shared and state building slots, both of which affect the whole state, while provinces have province building slots that only impact the individual province.",
		"These divisions require equipment and manpower to fight properly",
		"The tanks, airplanes, and boats could also be manually customised by the player",
		#"I co-authored Picturing Quantum Processes, with Aleks Kissinger, a book providing a fully diagrammatic treatment of quantum theory and its applications",
		#"Sea regions and provinces each have a type of terrain and weather assigned to them that determines how well different types of units will perform in combat there.",
		"Coecke is also a composer and musician, who has been called a pioneer of industrial music, and is also one of the pioneers of employing quantum computers in music",
		"Similarly, major seas and oceans (for warships) and the sky (for warplanes) are divided into different zones known as strategic regions",
		"How well divisions perform in combat depends on various factors, such as the quality of their equipment, the weather, the type of terrain, the skill and traits of the general commanding the divisions, aerial combat in the region, supply lines, and supporting units",
		"I am still supervising, at Oxford and elsewhere, and also still teach at Oxford's Mathematical Institute",
		"For the ground forces, the player may train, customize, and command divisions consisting of various types of infantry, tanks, and other units",
		#"He is a founder of the Quantum Physics and Logic community and conference series, and of the journal Compositionality",
		"previously i was professor of quantum foundations logics and structures at the department of computer science at oxford university where i was 20 years and co-founded and led a multi-disciplinary quantum group that grew to 50 members and i supervised close to 70 phd students",
		"he is also distinguished visiting research chair at the perimeter institute for theoretical physics",
		"i was the first person to have quantum foundations as part of his academic title",
		"he was professor of quantum foundations logics and structures at Oxford University until 2020",
		"bob coecke is a belgian theoretical physicist and logician who is chief scientist at quantum computing company Quantinuum",
		"in addition to mobilization there are other policies including the nation's stance on conscription and commerce",
		#"similarly major seas for oceans (for warships) and the sky (for warplanes) are divided into different zones known as strategic regions",
		"if he had studied the material more thoroughly he might have performed better on the exam which ultimately determined whether he would qualify for the advanced program that begins in the fall",
		"i did not think he was ugly before he showed me his face",
		"he who is without stones commits the first sin",
		"i am sinking",
		"he is ugly",
		"ugly is he who wears the crown",
		"the movie that we watched was amazing",
		"the man walks without feet",
		"the man who walks without feet is strange",
		"strange is the man who walks without feet",
		"i had eaten",
		"his face was extremely ugly",
		"looking at his face was repulsive",
		"to look at his face was repulsive",
		"i saw him leave",
		# "he is so fast",
		"he was left to die",
		"i suggest that he go home early",
		"kids grow up so fast",
		#"to be or not to be",
		"i wish it were friday already",
		#"kids grow up so fast these days",
		"his face was repulsive to look at",
		"his face was repulsive to look at as a result of his neglectful upbringing",
		"what she said that he thought she meant was, in fact, not what she meant at all",
		"the book is on the table",
		"she walked through the park in the morning",
		"he sat beside his friend during the movie",
		"they arrived long after the meeting had started",
		"the keys are under the couch",
		"we met at the coffee shop near the station",
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
	

