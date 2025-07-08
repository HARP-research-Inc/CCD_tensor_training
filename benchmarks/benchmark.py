from sentence_transformers import SentenceTransformer
import mteb
from src.DisCoBERT.DisCoBERT import DisCoBERT
from src.DisCoBERT.categories import Box
from torch import nn

def sentence_to_circuit_test():
	model = DisCoBERT("en_core_web_lg")
	with open("benchmarks/test.txt") as f:
		sentences = f.readlines()
	total = 0
	breaks = 0

	aggr_cosine_similarity = 0.00

	working_sentences = list()
	breaking_sentences = list()

	#main loop
	for sentence in sentences:
		print(sentence)
		sentence = sentence.strip()#.lower().replace(",", "")
		if len(sentence) <= 2 or sentence[0] == "=":
			continue
		total += 1
		try:
			embedding = model.encode(sentence)
			if embedding is None or embedding.shape[0] == 0:
				raise ValueError("Embedding is empty or None")
		except:
			breaks += 1
			breaking_sentences.append(sentence)
			import traceback
			print(traceback.format_exc())
		else:
			ground_truth = Box.model_cache.retrieve_BERT("the dogs among the men")

			aggr_cosine_similarity += nn.CosineSimilarity(dim=1)(embedding, ground_truth).item()
			working_sentences.append(sentence)

	return total, breaks, working_sentences, breaking_sentences, aggr_cosine_similarity / total if total > 0 else 0.0

def example():
	model_name = "sentence-transformers/all-MiniLM-L6-v2"

	# or using SentenceTransformers
	model = SentenceTransformer(model_name)

	# select the desired tasks and evaluate
	tasks = mteb.get_tasks(tasks=["AmazonReviewsClassification"], languages=["eng"])
	evaluation = mteb.MTEB(tasks=tasks)
	results = evaluation.run(model)

	for result in results:
		print(result)


if __name__ == "__main__":
	

	total, breaks, working_sentences, breaking_sentences, cosine_similarity = sentence_to_circuit_test()

	print(f"DisCoBERT breaks for {breaks} of {total} sentences tested. Framemwork breaks for {breaks/total * 100:.2f}% of sentences.")
	print(f"Average cosine similarity: {cosine_similarity:.4f}")
	print("Working sentences:")
	[print("-", sentence) for sentence in working_sentences]
	print("Breaking sentences:")
	[print("-", sentence) for sentence in breaking_sentences]