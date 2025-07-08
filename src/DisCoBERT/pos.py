from .categories import *
from src.regression import CPTensorRegression, TwoWordTensorRegression
#from .ann import ann

import torch
from sentence_transformers import SentenceTransformer
import spacy

"""
Models trained: 
- adv *
- aux 
- cconj adj 
- cconj noun
- cconj verb
- determiners *
- interjection
- prep aux
- prep verb
- pronoun
- sconj
- transitive verb *

"""

class NestedNetwork(torch.nn.Module):
	def __init__(self, parent: torch.nn.Module, child: torch.nn.Module):
		super().__init__()

		self.parent = parent
		self.child = child

	def forward(self, *inputs):
		return self.parent(self.child(*inputs))

class FunctionConjunction(torch.nn.Module):
	def __init__(self, coordinator: torch.nn.Module, children: list[torch.nn.Module]):
		super().__init__()

		self.coordinator = coordinator
		self.children = children

	def forward(self, *inputs):
		arr = self.children
		state = self.coordinator(arr[0](*inputs) if isinstance(arr[0], torch.nn.Module) else arr[0], arr[1](*inputs) if isinstance(arr[1], torch.nn.Module) else arr[1])
		arr = arr[2:]

		while arr:
			state = self.coordinator(state, arr[0](*inputs) if isinstance(arr[0], torch.nn.Module) else arr[0])
			arr = arr[1:]

		return state

class Spider(Box):
	"""
	DisCoCat Spider. Defined as a Box utilzing a composition
	function equivalent to the adposition "and", and equivalent in the 
	graphical calculus to a logical "and".
	"""
	def __init__(self, label, model_path):
		super().__init__(label, model_path)
		self.model_path = "cconj_adj_model"
		self.type = "spider"
		self.model = Box.model_cache.load_model(self.model_path, "and", n=2)
		self.embedding_state = None
	
	def forward_helper(self):

		for packet in self.packets:
			if type(packet[1]) is not torch.Tensor:
				raise ValueError(f"Expected a torch.Tensor for packet, got {type(packet[1])}")
			if self.embedding_state is None:
				self.embedding_state = packet[1]
			else:
				self.embedding_state = self.model(self.embedding_state, packet[1])
		
		if self.embedding_state is None:
			raise ValueError("No packets were processed, embedding state is None.") 
		
		return self.embedding_state

class Determiner(Box):
	def __init__(self, label: str, model_path: str):
		super().__init__(label, model_path)
		self.grammar = ['SELF', 'NOUN']
		self.type = "DET"
		self.model = Box.model_cache.load_ann((label, "det_model"), n=1)

	def forward_helper(self):
		"""
		returns the model
		"""
		print(f"Parsing determiner {self.label}...")
		return self.model

class Adverb(Box):
	def __init__(self, label: str, model_path: str):
		super().__init__(label, model_path)
		self.type = "ADV"
		self.grammar = ['SELF', 'VERB', '|', "SELF", "ADJ"]
		self.model = Box.model_cache.load_ann((label, "general_adv_model"), n=1)
	
	def forward_helper(self):
		"""
		returns the model
		"""
		return self.model

class Auxilliary(Box):
	def __init__(self, label: str, model_path: str):
		super().__init__(label, model_path)
		self.type = "AUX"
		self.grammar = ['SELF', 'VERB']
		self.model = Box.model_cache.load_ann((label, "aux_model"), n=1)
	
	def forward_helper(self):
		"""
		returns the model
		"""
		state_packets = [packet[1] for packet in self.packets if isinstance(packet[1], torch.Tensor)]

		if len(state_packets) == 1:
			output = self.model(state_packets[0])

			for packet in self.packets:
				if isinstance(packet[1], torch.nn.Module):
					output = packet[1](output)
			
			return output
		else:
			return self.model

class Interjection(Box):
	def __init__(self, label: str, model_path: str):
		super().__init__(label, model_path)
		self.type = "INTJ"
		self.grammar = ['SELF', 'SENTENCE']
		self.model = Box.model_cache.load_ann((label, "intj_model"), n=1)
	
	def forward_helper(self):
		"""
		returns the model
		"""
		return self.model

class Possessive(Box):
	"""

	"""
	def __init__(self, label: str, model_path: str):
		super().__init__(label, model_path)
		self.type = "PRON"
		self.grammar = ['SELF','NOUN']

		self.embedding_state = Box.model_cache.load_ann((label, "pron_model"), n=1)

	def forward_helper(self):
		return self.model

class Noun(Box):
	"""

	"""
	def __init__(self, label: str, model_path: str):
		super().__init__(label, model_path)
		self.type = "NOUN"
		self.grammar = ['ADJ','SELF']

		self.embedding_state = Box.model_cache.retrieve_BERT(label)

		self.inward_requirements: dict = {("ADJ", "0:inf"), ("PRON", "1:1")}

	def forward_helper(self):
		print(f"Parsing noun {self.label}...")

		for packet in self.packets:
			if isinstance(packet[1], torch.nn.Module):
				self.embedding_state = packet[1](self.embedding_state)
		
		return self.embedding_state

class VerbState(Box):
	"""

	"""
	def __init__(self, label: str, model_path: str):
		super().__init__(label, model_path)
		self.type = "STATE"
		self.grammar = ['PREP_MOD','SELF']

		self.embedding_state = Box.model_cache.retrieve_BERT(label)

		self.inward_requirements: dict = {("PREP_MOD", "0:inf")}

	def forward_helper(self):
		print(f"Parsing noun {self.label}...")

		for packet in self.packets:
			if isinstance(packet[1], torch.nn.Module):
				self.embedding_state = packet[1](self.embedding_state)
		
		return self.embedding_state

class Adjective(Box):
	"""

	"""
	def __init__(self, label: str, model_path: str):
		super().__init__(label, model_path)
		self.grammar = ['SELF', 'NOUN']
		self.type = "ADJ"
		self.model = Box.model_cache.load_ann((label, "adj_model"), n=1)
		self.inward_requirements: dict = {("ADV", "0:inf")}

	def forward_helper(self):
		"""
		returns the model
		"""

		#adv handling will be implemented when adv class is implemented
		return self.model

class Intransitive_Verb(Box):
	def __init__(self, label: str, model_path: str):
		super().__init__(label, model_path)
		self.grammar = ['NOUN', 'SELF', 'NOUN']
		self.type = "VERB"

		self.inward_requirements: dict = {("ADV", "0:inf"),
										  ("INTJ", "0:inf"), 
										 ("NOUN", "1:1")}
		self.model = Box.model_cache.load_ann((label, "intransitive_model"), n=1)
	
	def forward_helper(self):
		state_packets = [packet[1] for packet in self.packets if isinstance(packet[1], torch.Tensor)]

		print("intransitive packet length", len(self.packets))
		if len(state_packets) != 1:
			raise ValueError(f"Transitive verb {self.label} requires exactly one state packet, got {len(state_packets)}.")  
		
		output = self.model(state_packets[0])

		for packet in self.packets:
			if isinstance(packet[1], torch.nn.Module):
				output = packet[1](output)

		return output
		
	

class Transitive_Verb(Box):
	"""

	"""
	def __init__(self, label: str, model_path: str):
		super().__init__(label, model_path)
		self.grammar = ['NOUN', 'SELF', 'NOUN']
		self.type = "VERB"

		self.inward_requirements: dict = {("ADV", "0:inf"),
										  ("INTJ", "0:inf"), 
										 ("NOUN", "2:2")} 
		
		self.model = Box.model_cache.load_ann((label, "transitive_model"), n=2)

	def forward_helper(self):
		"""
		returns an embedding state after processing the NOUN packets.
		"""
		noun_packets = [packet[1] for packet in self.packets if packet[0] and not isinstance(packet[1], torch.nn.Module)]

		print("transitive packet length", len(self.packets))
		if len(noun_packets) != 2:
			raise ValueError(f"Transitive verb {self.label} requires exactly two NOUN packets, got {len(noun_packets)}.")  
		
		#noun packets at index 1 should be pytorch tensors
		output = self.model(noun_packets[0], noun_packets[1])

		####adverb stuff####
		for packet in self.packets:
			if isinstance(packet[1], torch.nn.Module):
				print("test")
				model:torch.nn.Module = packet[1]
				output = model(output)

		return output

class Linking_Verb(Box):
	"""

	"""
	def __init__(self, label: str, model_path: str):
		super().__init__(label, model_path)
		self.grammar = ['NOUN', 'SELF', 'ADP', '|', 'NOUN', 'SELF', 'ADJ']
		self.type = "VERB"
		
		self.model = Box.model_cache.load_ann((label, "aux_linking_model"), n=2)

	def forward_helper(self):
		"""
		returns an embedding state after processing the NOUN packets.
		"""
		word_packets = [packet[1] for packet in self.packets if packet[0] != "ADV"] #if packet[0] == "NOUN" or packet[0] == "ADJ" or packet[0] == "ADP"]
		adv_packets = [packet[1] for packet in self.packets if packet[0] == "ADV"]
		print([packet[0] for packet in self.packets])

		print("linking packet length", len(self.packets))
		if len(word_packets) != 2:
			raise ValueError(f"Linking verb {self.label} requires exactly two packets, got {len(word_packets)}.")  

		for i in range(len(word_packets)):
			if isinstance(word_packets[i], torch.nn.Module):
				word_packets[i] = word_packets[i](word_packets[i - 1])
		
		output = self.model(word_packets[0], word_packets[1])

		for packet in adv_packets:
			model:torch.nn.Module = packet
			output = model(output)

		return output

class Ditransitive_Verb(Box):
	"""

	"""
	def __init__(self, label: str, model_path: str):
		super().__init__(label, model_path)
		self.grammar = ['NOUN', 'SELF', 'NOUN']
		self.type = "VERB"

		self.inward_requirements: dict = {("ADV", "0:inf"),
										  ("INTJ", "0:inf"), 
										 ("NOUN", "3:3")} 
		
		self.model = Box.model_cache.load_ann((label, "ditransitive_model"), n=3)

	def forward_helper(self):
		"""
		returns an embedding state after processing the NOUN packets.
		"""
		noun_packets = [packet[1] for packet in self.packets if packet[0] == "NOUN"]

		print("ditransitive packet length", len(self.packets))
		if len(noun_packets) != 3:
			raise ValueError(f"Transitive verb {self.label} requires exactly three NOUN packets, got {len(noun_packets)}.")  
		
		#noun packets at index 1 should be pytorch tensors
		output = self.model(noun_packets[0], noun_packets[1], noun_packets[2])

		####adverb stuff####
		for packet in self.packets:
			if packet[0] == "ADV" or packet[0] == "INTJ":
				print("test")
				model:torch.nn.Module = packet[1]
				output = model(output)

		return output

class PrepositionHelper(Category):
	def __init__(self, label):
		super().__init__(label)

class Preposition(Box):
	def __init__(self, label: str, model_path: str):
		super().__init__(label, model_path)
		self.grammar = ['VERB', 'SELF', 'VERB', '|', 'AUX', 'SELF', 'NOUN', '|', 'NOUN', 'SELF', 'NOUN']
		self.type = "PREP"
		
		self.models = [Box.model_cache.load_ann((label, "prep_model"), n=2), Box.model_cache.load_ann((label, "prep_aux_model"), n=2), Box.model_cache.load_ann((label, "prep_verb_model"), n=2)]
		self.general_model = Box.model_cache.load_ann((label, "prep_noun_model"), n=1)

	def forward_helper(self):
		print(f"Parsing preposition {self.label}...")
		word_packets = [packet for packet in self.packets if isinstance(packet[1], torch.Tensor)]#if packet[0] == "NOUN" or packet[0] == "VERB" or packet[0] == "AUX"]
		print([packet[0] for packet in self.packets])

		print("packet length", len(word_packets), "out of", len(self.packets))

		#if len(word_packets) == 1:
		#	return self.general_model(word_packets[0][1])

		if len(word_packets) != 2:
			raise ValueError(f"Preposition {self.label} requires exactly two packets, got {len(word_packets)} from {len(self.packets)}.")  
		
		output = self.models[0](word_packets[0][1], word_packets[1][1]) if word_packets[0][0] == "NOUN" and word_packets[1][0] == "NOUN" \
			else self.models[1](word_packets[0][1], word_packets[1][1]) if word_packets[0][0] == "AUX" and word_packets[1][0] == "NOUN" \
			else self.models[2](word_packets[0][1], word_packets[1][1])

		return output

class PrepositionalModifier(Box):
	def __init__(self, label: str, model_path: str):
		super().__init__(label, model_path)
		self.grammar = ['SELF']
		self.type = "PREP_MOD"
		
		self.model = Box.model_cache.load_ann((label, "prep_prt_model"), n=1)

	def forward_helper(self):
		return self.model


class SubordinatingConjunction(Box):
	def __init__(self, label: str, model_path: str):
		super().__init__(label, model_path)
		self.grammar = ['SENTENCE', 'SELF', 'SENTENCE']
		self.type = "SCONJ"
		
		self.models = [Box.model_cache.load_ann((label, "general_sconj_model"), n=1), Box.model_cache.load_ann((label, "sconj_model"), n=2)]

	def forward_helper(self):
		word_packets = [packet[1] for packet in self.packets]

		print("packet length", len(self.packets))
		if len(word_packets) not in (1, 2):
			raise ValueError(f"Subordinating conjunction {self.label} requires exactly two packets, got {len(word_packets)} from {len(self.packets)}.")  
		
		if len(word_packets) == 1 and isinstance(word_packets[0], torch.nn.Module):
			return NestedNetwork(self.models[len(word_packets) - 1], word_packets[0])

		output = self.models[len(word_packets) - 1](*word_packets)
	
		return output
	
"""class Preposition(Box):
	def __init__(self, label: str, model_path: str):
		super().__init__(label, model_path)
		self.grammar = ['SELF', 'NOUN']
		self.type = "PREP"
		self.model = Box.model_cache.load_ann((label, "prep_noun_model"), n=1)

	def forward_helper(self):
		noun_packets = [packet[1] for packet in self.packets if packet[0] == "NOUN" or packet[0] == "VERB"]

		if len(noun_packets) != 1:
			raise ValueError(f"Preposition {self.label} requires exactly one NOUN packets, got {len(noun_packets)}.")  
		
		#noun packets at index 1 should be pytorch tensors
		output = self.model(noun_packets[0])

		return output"""

class Intransitive_Verb(Box):
	def __init__(self, label: str, model_path: str):
		super().__init__(label, model_path)
		self.grammar = ['NOUN', 'SELF', 'NOUN']
		self.type = "VERB"

		self.inward_requirements: dict = {("ADV", "0:inf"),
										  ("INTJ", "0:inf"), 
										 ("NOUN", "1:1")}
		self.model = Box.model_cache.load_ann((label, "intransitive_model"), n=1)
	
	def forward_helper(self):
		noun_packets = [packet[1] for packet in self.packets if isinstance(packet[1], torch.Tensor)]

		print("packet length", len(self.packets))
		if len(noun_packets) != 1:
			raise ValueError(f"Intransitive verb {self.label} requires exactly one NOUN packet, got {len(noun_packets)}.")  
		
		output = self.model(noun_packets[0])

		####adverb stuff####
		for packet in self.packets:
			if isinstance(packet[1], torch.nn.Module):
				model:torch.nn.Module = packet[1]
				output = model(output)

		return output

class Conjunction(Box):
	"""

	"""
	def __init__(self, label: str, model_path: str):
		super().__init__(label, model_path)
		self.grammar = ['VERB', 'SELF', 'VERB', '|', 'AUX', 'SELF', 'VERB', '|', 'NOUN', 'SELF', 'NOUN', '|', 'ADJ', 'SELF', 'ADJ']
		self.type = "CCONJ"
		
		self.models = [Box.model_cache.load_ann((label, "cconj_verb_model"), n=2), Box.model_cache.load_ann((label, "cconj_verb_model"), n=2), Box.model_cache.load_ann((label, "cconj_noun_model"), n=2), Box.model_cache.load_ann((label, "cconj_adj_model"), n=2)]

	def forward_helper(self):
		"""
		returns an embedding state after processing the NOUN packets.
		"""

		print(f"Parsing conjunction {self.label}...")
		word_packets = [packet[1] for packet in self.packets if isinstance(packet[1], torch.Tensor)]
		print([packet[0] for packet in self.packets])

		if any(isinstance(packet[1], torch.nn.Module) for packet in self.packets):
			return FunctionConjunction(self.models[3], [packet[1] for packet in self.packets])

		print("packet length", len(word_packets), "out of", len(self.packets))
		while len(word_packets) > 2:
			word_packets[1] = self.models[3](word_packets[0], word_packets[1])
			word_packets = word_packets[1:]
		
		output = self.models[0](word_packets[0], word_packets[1]) if word_packets[0] == "VERB" and word_packets[1] == "VERB" \
			else self.models[1](word_packets[0], word_packets[1]) if word_packets[0] == "AUX" and word_packets[1] == "VERB" \
			else self.models[2](word_packets[0], word_packets[1]) if word_packets[0] == "NOUN" and word_packets[1] == "NOUN" \
			else self.models[3](word_packets[0], word_packets[1])

		return output

	"""
	"PART OF SPEECH" DEFINITION PROBLEM...

	NEW PARTS OF SPEEC WILL BE DEFINED HERE



	VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
	"""

class Box_Factory(object):
	"""
	Factory for creating boxes.
	"""
	def __init__(self, NLP: spacy.load, model_path, lenient = True):
		self.NLP = NLP
		self.model_path = model_path
		self.lenient = lenient

	def returns_state(self, token, feature):
		return self.is_linking_verb(token, feature) if feature == "AUX" else feature not in ["ADV", "AUX", "DET"]

	def is_linking_verb(self, token, feature: str): # set(['nsubj', 'prep']), set(['nsubj', 'attr']), set(['nsubj', 'acomp']), set(['acomp', 'ccomp']), set(['acomp', 'relcl']) 
		return feature in ["AUX", "VERB"] and (set(child.dep_ for child in token.children) in [set([relA, relB]) for relA in ['nsubj', 'acomp'] for relB in ['prep', 'attr', 'acomp', 'ccomp', 'relcl', 'xcomp']] or (feature == "AUX" and any("subj" in child.dep_ for child in token.children) and any(dep in child.dep_ for dep in ["aux", "attr", "relcl", "acomp"] for child in token.children)))

	def create_box(self, token: spacy.tokens.Token, feature: str):
		if token is not None:
			label = str(token.text).replace(',', '').lower()

		if feature == "spider":
			return Spider("SPIDER", self.model_path)
		elif feature == "bureaucrat":
			return Bureaucrat("REFERENCE")
		elif self.is_linking_verb(token, feature):
			print(token.text, "is linking verb because", [child.dep_ for child in token.children])
			return Linking_Verb(label, self.model_path)
		elif feature == "PRON" and any(child.dep_ == "poss" for child in token.children):
			return Possessive(label, self.model_path)
		elif feature == "NOUN" or feature == "PROPN" or feature == "PRON" or feature == "NUM":
			return Noun(label, self.model_path)
		elif feature == "ADJ":
			return Adjective(label, self.model_path)
		elif feature == "VERB":
			nsubj, dobj, dative = (None, None, None)
			for child in token.children:
				if "subj" in child.dep_ or child.dep_ in ["nsubj", "nsubjpass", "attr", "relcl"]:
					nsubj = child.text
				if child.dep_ in ["dobj", "expl"] or (child.dep_ == "ccomp" and self.returns_state(child, child.pos_)):
					dobj = child.text
				if child.dep_ == "dative":
					dative = child.text
			if not nsubj:
				if dobj:
					return Intransitive_Verb(label, self.model_path)
				else:
					#raise ValueError(f"Sanity check: verb {label} somehow has no subject.")
					return VerbState(label, self.model_path)
			else:
				if dobj and dative:
					return Ditransitive_Verb(label, self.model_path)
				elif dobj:
					return Transitive_Verb(label, self.model_path)
				else:
					return Intransitive_Verb(label, self.model_path)
		elif feature == "DET":
			return Determiner(label, self.model_path)
		elif feature == "ADV":
			return Adverb(label, self.model_path)
		elif feature == "INTJ":
			return Interjection(label, self.model_path)
		elif feature == "ADP" and len(list(token.children)) == 0:
			return PrepositionalModifier(label, self.model_path)
		elif feature == "ADP":
			return Preposition(label, self.model_path)
		elif feature == "SCONJ":
			return SubordinatingConjunction(label, self.model_path)
		elif feature == "AUX":
			return Auxilliary(label, self.model_path)
		elif feature == "CCONJ":
			return Conjunction(label, self.model_path)
		else:
			if self.lenient:
				return Box(label, self.model_path)
			else:
				raise ValueError(f"Unknown feature: {feature}")
	
	def set_lenient(self, value: bool):
		"""
		Sets the lenient mode of the factory.
		If lenient is True, unknown features will return a generic Box.
		If False, an error will be raised for unknown features.
		"""
		self.lenient = value
		return self
 
if __name__ == "__main__":
	factory = Box_Factory(spacy.load("en_core_web_trf"), "/mnt/ssd/user-workspaces/aidan-svc/CCD_tensor_training/")

	# tiny_discourse = Circuit("Tiny Discourse")

	for i in range(1000):
		test = factory.create_box(("Abbasid", "adj_model"), "ADJ")

		#print(i, type(test.model))
	

	# tiny_discourse.forward()

	# print(tiny_discourse)


