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

MODEL_PATH = "/mnt/ssd/user-workspaces/aidan-svc/CCD_tensor_training/models/discobert"
CONJUNCTION_LIST = set()#SUBORDINATING_CONJUNCTIONS["temporal"] | SUBORDINATING_CONJUNCTIONS["causal"] | \
	#SUBORDINATING_CONJUNCTIONS["conditional"] | SUBORDINATING_CONJUNCTIONS["concessive"] | \
	#SUBORDINATING_CONJUNCTIONS["purpose"] | SUBORDINATING_CONJUNCTIONS["result/consequence"] | \
	#SUBORDINATING_CONJUNCTIONS["comparison"] | SUBORDINATING_CONJUNCTIONS["manner"] | \
	#SUBORDINATING_CONJUNCTIONS["exception"]# | SUBORDINATING_CONJUNCTIONS["relative (nominal)"] |
	#{"and", "but", "or", "nor", "for", "so", "yet", "either", "neither", "and/or"}

PUNCTUATION_DELIMS = {".", "!", "?"}#, ";", ":"}

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

	if isinstance(child_box, tuple):
		circuit.add_wire(child_box[0], parent)
		if level in levels:
			levels[level].append(child_box[0])
		else:
			levels[level] = [child_box[0]]
		
		parent = child_box[0]
		child_box = child_box[1]
		level += 1

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
	return [t for t in token.doc if t.head == token if t != token and t.dep_ != "removed"]

def flip(doc, relations, where: lambda token: True):
	alreadyParsed = set()

	if isinstance(relations, str):
		relations = [relations]

	while True:
		flipped = False
		root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
		queue = [root]
		elems = [root]
		
		while len(queue) > 0:
			queue = [child for token in queue for child in get_children(token)]

			elems += queue
		elems.reverse()

		print("root", root)
		for token in elems:
			print("-", token.text)
			if (not relations or token.dep_ in relations) and token not in alreadyParsed and where(token):
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

				print("flip", prev_token.text, prev_token_dep, original_head.text, original_head_dep)

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

			prev_token.head = prev_token if original_head.head == original_head else original_head.head
			prev_token.dep_ = original_head_dep
			original_head.head = prev_token
			original_head.dep_ = prev_token_dep
			alreadyParsed.add(original_head)	

			print(prev_token.text, "now has head", prev_token.head.text)
			print(original_head.text, 'now has head', prev_token.text)

def rearrange(doc, relation, relationRoot, rootPOS=None, multiLevel=False, replacePOS=None, sourcePOS=None, reverse=True):
	alreadyParsed = set()

	if isinstance(relation, str):
		relation = [relation]
	if isinstance(relationRoot, str):
		relationRoot = [relationRoot]
	if isinstance(sourcePOS, str):
		sourcePOS = [sourcePOS]

	while True:
		rearranged = False
		root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
		queue = [root]
		elems = [root]
		
		while len(queue) > 0:
			queue = [child for token in queue for child in get_children(token)]

			elems += queue
		
		if reverse:
			elems.reverse()

		print("root", root)
		for token in elems:
			for childA in [child for child in get_children(token) if any(child.dep_ == rel for rel in relation)]:
				for childB in [child for child in (get_children(childA) if multiLevel else get_children(token)) if any(child.dep_ == rel for rel in relationRoot) and (rootPOS is None or child.pos_ == rootPOS)]:
					print(token.text, childA.text, childB.text, token in alreadyParsed, sourcePOS and token.pos_ not in sourcePOS, token.pos_, sourcePOS)	
			
			if token in alreadyParsed:
				continue

			if sourcePOS and token.pos_ not in sourcePOS:
				continue

			for childA in [child for child in get_children(token) if any(child.dep_ == rel for rel in relation)]:
				for childB in [child for child in (get_children(childA) if multiLevel else get_children(token)) if any(child.dep_ == rel for rel in relationRoot) and (rootPOS is None or child.pos_ == rootPOS)]:
					if childB in alreadyParsed:
						continue
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
					print(childB.text, "head is", childB.head.text, "and", childA.text, "head is", childA.head.text, "and", token.text, "head is", token.head.text)

					root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
					print("root", root)

					to_nltk_tree(root).pretty_print()

					rearranged = True
					break
				
				if rearranged:
					break
		
		if not rearranged:
			break

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

	for token in doc:
		print(token.pos_, token.text)
		for child in get_children(token):
			print(">>>", child.dep_, child.text)
	print()

	roots = [token for token in doc if token.head == token]

	while len(roots) > 1:
		roots[0].head = roots[1]
		roots = roots[1:]

	remove = set()
	for token in doc:
		if re.match(r'^[^A-Za-z0-9;:]$', token.text):
			remove.add(token)
			token.head = token
			token.dep_ = "removed"

	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	to_nltk_tree(root).pretty_print()

	for i, token in enumerate(doc):
		if token.pos_ == "PUNCT" and token.text in [":", ";"]:
			head = None
			child = None

			for j, t in enumerate(doc):
				for k, kt in enumerate(doc):
					if (j < i) != (k < i) and t.dep_ in ["appos", "ccomp"] and t.head != t and t.head == kt:
						child = t
						head = kt

			if child:
				if head.head == head:
					token.head = token
					
				head.head = token
				child.head = token
				head.dep_ = "none"
				child.dep_ = "none"

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

	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	to_nltk_tree(root).pretty_print()


	print("\n\n<<< CONJUNCTION FIXING >>>\n\n")

	for token in doc:
		if token.dep_ == "conj" or token.dep_ == "punct":
			has_cc_sibling = False
			queue = [token.head, token]

			while queue and not has_cc_sibling:
				nextQueue = []

				for item in queue:
					for child in get_children(item):
						if child.dep_ == "conj":
							nextQueue.append(child)
						elif child.dep_ == "cc":
							has_cc_sibling = True
							break
					
					if has_cc_sibling:
						break
				
				queue = nextQueue
			
			if not has_cc_sibling:
				for candidate in doc:
					if candidate.dep_ == "cc":
						has_conj_sibling = any(sib.dep_ == "conj" for sib in get_children(candidate.head) if sib is not candidate)

						if not has_conj_sibling:
							if candidate.head != token and token.head != candidate and token != candidate.head.head:
								print(token.text, "rewired to sibling of", candidate.head)
								token.head = candidate.head
								token.dep_ = "conj"
								break


	for token in doc:
		found = False
		for childA in [child for child in get_children(token) if child.dep_ in ["amod"]]:
			for childB in [child for child in get_children(childA) if child.dep_ in ["cc"] and child.pos_ == "CCONJ"]:
				if any(child.dep_ == "conj" for child in get_children(childA)):
					break

				prev_head = token.head
				childA.head = prev_head if prev_head != token else childA
				token.head = childA
				childA.dep_ = token.dep_
				token.dep_ = "conj"
				print(childA.text, "changed to conjunction with", token.text)
				found = True
				break

			if found:
				break
	
	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	to_nltk_tree(root).pretty_print()

	print("\n\n<<< CLAUSE POS MODIFICATION >>>\n\n")

	#rearrange(doc, ["relcl"], ["nsubjpass", "nsubj"], "PRON", True, "SCONJ")

	#rearrange(doc, ["pcomp"], ["mark"], "SCONJ", True)

	for token in doc:
		found = False
		for childA in [child for child in get_children(token) if child.dep_ in ["pcomp"]]:
			for childB in [child for child in get_children(childA) if child.dep_ in ["mark"] and child.pos_ == "SCONJ"]:
				childB.head = token
				print(childB.text, "head set to", token.text)
				found = True
				break

			if found:
				break

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
				hasCoordinator = False
				for child in get_children(targetElem):
					if child.dep_ == "conj":
						hasConj = True
					if child.dep_ == "cc":
						hasCoordinator = True
				
				for child in get_children(targetElem):
					if child.dep_ == "conj":# or ((hasConj or hasCoordinator) and child.dep_ in ["dobj", "appos", "nummod"]):
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

	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	to_nltk_tree(root).pretty_print()

	print("\n\n<<< CLAUSE REORDERING >>>\n\n")

	for token in doc:
		print(token.pos_, token.text)
		for child in get_children(token):
			print(">>>", child.dep_, child.text)
		print()

	for token in doc:
		for childA in [child for child in get_children(token) if child.dep_ in ["ccomp", "relcl", "xcomp"]]:
			if any([child.dep_ in ["advmod", "nsubjpass"] for child in get_children(childA)]):
				continue

			for childB in [child for child in get_children(childA) if child.dep_ in ["nsubj", "expl", "mark"] and child.pos_ in ["PRON", "SCONJ"]]:
				childB.tag_ = "OPP"
				childB.dep_ = "nsubj"

				#childA.head = prev_head if prev_head != token.head else childA
				#token.head.head = childA

				#found = True
				#break

			#if found:
			#	break
		
		#if token.dep_ == "dep" and token.pos_ == "PRON":
		#	token.tag_ = "OPP"
		#	token.dep_ = "nsubj"
	
	#for token in doc:
	#	if token.dep_ == "expl" and token.pos_ == "PRON":
	#		token.pos_ = "SCONJ"
	
	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	to_nltk_tree(root).pretty_print()

	rearrange(doc, ["advcl", "ccomp", "relcl", "acl", "xcomp", "expl"], ["mark", "advmod", "nsubjpass", "nsubj", "dobj", "expl"], "SCONJ", False)

	print("START SECTION")

	rearrange(doc, ["advcl", "ccomp", "relcl", "acl", "xcomp"], ["mark", "advmod", "nsubjpass", "nsubj", "dobj", "expl"], "SCONJ", True)

	print("END SECTION")
	
	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	to_nltk_tree(root).pretty_print()

	rearrange(doc, "nsubjpass", "auxpass")

	for token in doc:
		print(token.pos_, token.text)
		for child in get_children(token):
			print(">>>", child.dep_, child.text)
	print()

	rearrange(doc, ["xcomp", "advcl", "acl", "aux"], ["aux", "auxpass"], "PART", True, "ADP", None, False)
	
	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	to_nltk_tree(root).pretty_print()

	for token in doc:
		found = False
		if token.pos_ == "SCONJ":
			continue

		for childA in [child for child in get_children(token) if child.dep_ in ["advmod"]]:
			for childB in [child for child in get_children(token) if child.dep_ in ["advcl"]]:
				childB.head = childA
				print(childB.text, "head set to", childA.text)
				found = True
				break

			if found:
				break
	
	for token in doc:
		found = False
		for childA in [child for child in get_children(token) if child.dep_ in ["attr"] and child.pos_ == "PROPN"]:
			for childB in [child for child in get_children(token) if child.dep_ == "ccomp"]:
				for childC in [child for child in get_children(childB) if child.dep_ in ["nsubj"] and child.pos_ == "PRON"]:
					prevHead = childB.head
					prevDep = childB.dep_
					childB.head = childC
					childB.dep_ = childC.dep_
					childA.head = childC
					childC.pos_ == "SCONJ"
					childC.dep_ = prevDep
					childC.head = prevHead
					print(childA.text, childB.text, childC.text, "all rearranged")
					found = True
					break

				if found:
					break
			if found:
				break

	for token in doc:
		print(token.pos_, token.text)
		for child in get_children(token):
			print(">>>", child.dep_, child.text)
	print()

	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	to_nltk_tree(root).pretty_print()
	
	print("\n\n<<< PREP FLIP >>>\n\n")

	for token in doc:
		if token.dep_ == "acomp":
			for child in get_children(token):
				if child.dep_ == "prep":
					child.head = token.head
					print("set", child.text, "head to", token.head.text)

	flip(doc, ["prep", "agent", "dative"], lambda token: len(list(get_children(token))) > 0 and token.pos_ in ["ADP", "SCONJ", "VERB"] and not token.head.pos_ in ["ADP"])# and not token.head.dep_ == "prep")
	
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
	

	rewire(doc, "mark", lambda child: child.pos_ == "VERB", lambda parent: parent.pos_ == "ADP")

	for token in doc:
		if token.pos_ == "VERB" and token.dep_ == "xcomp" and token.head != token and token.head.pos_ == "VERB":
			original_head = token.head
			original_head_dep = original_head.dep_
			token.head = original_head.head if original_head.head != original_head else token
			original_head.dep_ = "nsubj"
			token.dep_ = original_head_dep
			original_head.head = token
	
	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	to_nltk_tree(root).pretty_print()

	flip(doc, "advmod", lambda token: token.head.pos_ in ["ADV", "AUX", "VERB"] and token.pos_ == "SCONJ")

	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	to_nltk_tree(root).pretty_print()
	print("\n\n<<< POS FIXING >>>\n\n")

	for token in doc:
		if token.pos_ in ["NOUN", "PRON"] and token.dep_ == "dep" and token.head.pos_ == "AUX":
			token.pos_ = "ADV"
		elif token.pos_ in ["NOUN", "PRON"] or token.pos_ == "ADJ":
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
			#adv_children = [child for child in get_children(token) if child.pos_ == "ADV"]
			#for child in adv_children:
			#	child.pos_ = "NOUN"

			if any(child.pos_ == "ADJ" for child in get_children(token)):
				for child in get_children(token):
					if child.pos_ == "NOUN":
						child.pos_ = "ADJ"
	
	for token in doc:
		if token.pos_ == "PUNCT" and token.text in [":", ";"] and len(get_children(token)) == 0:
			token.head = token
			token.dep_ = "removed"

	for token in doc:
		print(token.pos_, token.text)
		for child in get_children(token):
			print(">>>", child.dep_, child.text)
	print()
	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
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
	root = [token for token in doc if token.head == token and token.dep_ != "removed"][0]
	print("root", root)

	leaves = list()

	parse_driver(circuit, source, leaves, root, factory, levels, 0)

	return leaves


def split_clauses_with_markers(sentence: str, nlp: spacy.load):
	# Build regex for conjunctions (prioritize multi-word)
	sentence = re.sub(r' [‘’\']', ' ', re.sub(r'[‘’\'] ', ' ', re.sub(r'[“”"]', '', sentence)))
	sorted_conjs = sorted(CONJUNCTION_LIST, key=lambda x: -len(x))
	escaped_conjs = [r'\b' + re.escape(conj) + r'\b' for conj in sorted_conjs]
	conj_pattern = '|'.join(escaped_conjs)
	
	# Build regex for punctuation
	punct_pattern = ''.join(re.escape(p + ' ') for p in PUNCTUATION_DELIMS)

	print(sentence)
	# Combined pattern: capture all splitters
	#pattern = r'\s*(%s|%s)\s*' % (conj_pattern, punct_pattern)
	pattern = r'(?<!\b[A-Z])[%s]\s+' % (punct_pattern)

	# Split and keep delimiters
	parts = re.split(pattern, sentence)

	# Group into clauses and splitters
	clauses = parts #parts[::2]
	markers = parts[1::2]

	print(clauses)

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
		print("CLAUSE", i+1, ":", clause)
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

	path_to_models = "/mnt/ssd/user-workspaces/aidan-svc/CCD_tensor_training/models/discobert"
	spacy_model = "en_core_web_trf"

	one_clause = "the big fat deformed french man eats a small helpless newborn baby"
	one_clause2 = "small dog eats big man"
	annoying = "she should have been being watched carefully"

	nlp = spacy.load(spacy_model)

	dummy = Category("blank")

	dummy.set_nlp(nlp)
	

	from nltk import Tree

	doc = nlp("his face was repulsive to look at as a result of his neglectful upbringing")

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
		"Each state has a certain amount of shared and state building slots, both of which affect the whole state, while provinces have province building slots that only impact the individual province.",
		"These divisions require equipment and manpower to fight properly",
		"The tanks, airplanes, and boats could also be manually customised by the player",
		#"I co-authored Picturing Quantum Processes, with Aleks Kissinger, a book providing a fully diagrammatic treatment of quantum theory and its applications",
		"Sea regions and provinces each have a type of terrain and weather assigned to them that determines how well different types of units will perform in combat there.",
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
		"similarly major seas for oceans (for warships) and the sky (for warplanes) are divided into different zones known as strategic regions",
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
		"he is so fast",
		"he was left to die",
		"i suggest that he go home early",
		"kids grow up so fast",
		"to be or not to be",
		"i wish it were friday already",
		"kids grow up so fast these days",
		"his face was repulsive to look at",
		"his face was repulsive to look at as a result of his neglectful upbringing",
		#"what she said that he thought she meant was, in fact, not what she meant at all",
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
		"i ate some rice and beans",
		"British rock musicians in the 1960s, especially the Rolling Stones, Eric Clapton, and John Mayall, were strongly influenced by the blues, as were such American rock musicians as Mike Bloomfield, Paul Butterfield, and the Allman Brothers Band",
		"A simple example of a recursive rule is the successor function in mathematics, which takes a number as input and yields that number plus 1 as output",
		"I am still supervising, at Oxford and elsewhere, and also still teach at Oxford's Mathematical Institute",
		"In one way or another, socialists now seem more interested in bringing the free market under control than in eliminating it completely",
		"Of particular consequence was his adoption of the behaviouristic theory of semantics according to which meaning is simply the relationship between a stimulus and a verbal response",
		"Grammar increasingly parted company with its older fellow disciplines within philosophy as they moved over to the domain known as natural science, and technical academic grammatical study increasingly became involved with issues represented by empiricism versus rationalism and their successor manifestations on the academic scene",
		"it is difficult to be sure",
		"Dionysius defined a sentence as a unit of sense or thought, but it is difficult to be sure of his precise meaning",
		"Whenever a solid is exposed to a liquid or a gas, a reaction occurs initially on the surface of the solid, and its properties can change dramatically as a result.",
		"But this line of reasoning also led to the uncomfortable notion that elementary gases had polyatomic molecules (O2, H2, and so on), and therefore many chemists rejected Avogadro’s hypotheses.",
		"In September he graduated from the military academy, ranking 42nd in a class of 58.",
		"Most believe that an improved social adjustment of individuals would decrease frustration, insecurity, and fear and would reduce the likelihood of war.",
		"In general, alchemists sought to manipulate the properties of matter in order to prepare more valuable substances.",
		"it reveals a deeper structure, allowing you to solve an entire class of similar problems efficiently",
		"Neither Tolstoy's religion nor his pacifism was shared by the earlier flamboyant Russian anarchist Mikhail Bakunin, who held that religion, capitalism, and the state are forms of oppression that must be smashed if people are ever to be free.",
		"Today, chemists can maneuver atoms one by one with a scanning tunneling microscope, and other techniques of what has become known as nanotechnology are in rapid development.",
		"It maintained divorce but granted only limited legal rights to women",
		#"The areas of specialization that emerged early in the history of chemistry, such as organic, inorganic, physical, analytical, and industrial chemistry, along with biochemistry, remain of greatest general interest.",
		"In the first place, he wanted to be consecrated by the pope himself, so that his coronation should be even more impressive than that of the kings of France.",
		"As these major approaches to peace envisaged in its Charter have not proved very fruitful, the United Nations has developed two new procedures aiming at the limitation of wars.",
		"It was preferable, as far as possible, to avoid basing the grammatical analysis of a language on semantic considerations",
		"It was Berzelius who in 1813 had proposed the alphabetic system for denoting elements, atoms, and molecular formulas, and the use of formulas as an aid for studying chemical composition and reactions began to blossom about 1830.",
	]

	if True:
		example_sentences = [
			"He ate meat, which made a mess.",
			"An animal breathes air, emitting phlogiston in an analogy to a slow fire, fueled by the phlogiston-rich food it consumes.",
			"All of them postulate that there exists an international society of states that accepts the binding force of some norms of international behaviour.",
			"Although Eugene V. Debs won nearly one million votes in the U.S. presidential election of 1920, his showing represented less than 4 percent of the votes cast and remains the electoral high point for American socialists.",
			"Much of the description of the indigenous languages of America has been carried out since the days of Boas and his most notable pupil Sapir by scholars who were equally proficient both in anthropology and in descriptive linguistics; such scholars have frequently added to their grammatical analyses of languages some discussion of the meaning of the grammatical categories and of the correlations between the structure of the vocabularies and the cultures in which the languages operated.",
			#"In fact, Marx and his longtime friend and collaborator Friedrich Engels were largely responsible for attaching the label \"utopian,\" which they intended to be derogatory, to Saint-Simon, Fourier, and Owen, whose \"fantastic pictures of future society\" they contrasted to their own \"scientific\" approach to socialism.",
			"The path to socialism proceeds not through the establishment of model communities that set examples of harmonious cooperation to the world, according to Marx and Engels, but through the clash of social classes.",
			"Since his youth he had spent his life building a party that would win such a victory, and now at the age of 47 he and his party had triumphed.",
			"When four carbon atoms are joined together, two different structures are possible: a linear structure designated n-butane and a branched structure called iso-butane.",
			"The term structuralism was used as a slogan and rallying cry by a number of different schools of linguistics, and it is necessary to realize that it has somewhat different implications according to the context in which it is employed.",
			#"Two important points arise here: first, that the structural approach is not in principle restricted to synchronic linguistics; second, that the study of meaning, as well as the study of phonology and grammar, can be structural in orientation.",
			"For the first two-thirds of the 20th century, chemistry was seen by many as the science of the future.",
			"A few months after this discovery, Marie Curie died as a result of aplastic anemia caused by the action of radiation.",
			"It is this great potential for structural diversity that makes carbon compounds essential to living organisms.",
			"According to the Treaty of Amiens, the British, who had taken the island on the collapse of the French occupation, should have restored it to the Hospitallers; but the British, on the pretext that the French had not yet evacuated certain Neapolitan ports, refused to leave the island.",
			"Personally, he was indifferent to religion: in Egypt he had said that he wanted to become a Muslim.",
			"The mission was simple: kill everyone.",
			"The sudden death of Pierre Curie (April 19, 1906) was a bitter blow to Marie Curie, but it was also a decisive turning point in her career: henceforth she was to devote all her energy to completing alone the scientific work that they had undertaken.",
			"Decisive as ever, he returned to France like a thunderbolt.",
			"Indeed, by this time a fissure had clearly developed between communists on the one hand and socialists, or social democrats, on the other.",
			"Yet, by reducing the number of states, by pushing the frontiers about, by amalgamating populations, and by propagating institutions like those that the Revolution and nationalism had created in France, he prepared the ground for German and Italian unification.",
			"Spain was induced to declare war on Great Britain in December 1804, and it was decided that French and Spanish squadrons massed in the Antilles should lure a British squadron into these waters and defeat it, thus making the balance roughly equal between the Franco-Spanish navy and the British.",
			"In insisting upon the necessity of treating each language as a more or less coherent and integrated system, both European and American linguists of this period tended to emphasize, if not to exaggerate, the structural uniqueness of individual languages.",
			#"There was especially good reason to take this point of view given the conditions in which American linguistics developed from the end of the 19th century.",
			"Under these circumstances, such linguists as Franz Boas (died 1942) were less concerned with the construction of a general theory of the structure of human language than they were with prescribing sound methodological principles for the analysis of unfamiliar languages.",
			"Increasingly, however, and especially in the public mind, the negative aspects of chemistry have come to the fore.",
			"On the evening of November 6, he wrote a letter to the members of the Central Committee exhorting them to proceed that very evening to arrest the members of the Provisional Government.",
			"Cooking, fermentation, glass making, and metallurgy are all chemical processes that date from the beginnings of civilization.",
			"The number was less than 500,000 as recently as 1965.",
			"When a hot body cools down, the thermal energy it loses passes to the surrounding air, which is at a lower temperature.",
			"Because science was still a long way from being able to give a comprehensive account of most stimuli, no significant or interesting results could be expected from the study of meaning for some considerable time, and it was preferable, as far as possible, to avoid basing the grammatical analysis of a language on semantic considerations.",
			"The studies in these and other works made use of paired examples to show how very similar events can be reported in very different ways, depending upon whether and how state and corporate interests may be affected.",
			"When metallic iron becomes red rust, it loses its phlogiston, just as a burning log does.",
			"Hugo Chávez’s call for a 'Bolivarian Revolution.' Apart from the appeal to Simón Bolívar’s reputation as a liberator, however, Chávez did not establish a connection between socialism and Bolívar’s thoughts and deeds.",
			"Such a society would operate on the principle of mutualism, according to which individuals and groups would exchange products with one another on the basis of mutually satisfactory contracts.",
			"He forbade all trade with the British Isles, ordered the confiscation of all goods coming from English factories or from the British colonies, and condemned as fair prize not only every British ship but also every ship that had touched the coasts of England or its colonies.",
			"But there was further development in Prague of the functional approach to syntax (see below).",
			"In the latter part of the 20th century, in the aftermath of two World Wars and in the shadow of nuclear, biological, and chemical holocaust, more was written on the subject than ever before.",
			"To the contrary, Bakunin argued, the dictatorship of the proletariat threatened to become even more oppressive than the bourgeois state, which at least had a militant and organized working class to check its growth.",
			"Now that peace had come, Lenin believed that their opposition was more dangerous than ever, since the peasantry and even a large section of the working class had become disaffected with the Soviet regime.",
			"The two groups fought each other ceaselessly within the same RSDWP and professed the same program until 1912, when Lenin made the split final at the Prague Conference of the Bolshevik Party.",
			"They elected him president",
			"And if this is the case, then the appearance of language could have been brought about by a single genetic mutation in a single individual, so long as that mutation were transmissible to progeny.",
			#"Presley became the teen idol of his decade, greeted everywhere by screaming hordes of young women, and, when it was announced in early 1958 that he had been drafted and would enter the U.S. Army, there was that rarest of all pop culture events, a moment of true grief.",
			"Another important innovation was combinatorial chemistry, in which scores of compounds are simultaneously prepared—all permutations on a basic type—and then screened for physiological activity.",
		]
		#example_sentences = [
		#	"It is this great potential for structural diversity that makes carbon compounds essential to living organisms.",]
		#example_sentences = ["He ate meat which made a mess."]
		#example_sentences = ["The path to socialism proceeds not through the establishment of model communities that set examples of harmonious cooperation to the world, according to Marx and Engels, but through the clash of social classes."]

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
	

