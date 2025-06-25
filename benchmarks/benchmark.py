from sentence_transformers import SentenceTransformer
import mteb
from benchmarks.DisCoBERT_wrapper import DisCoBERTWrapper

if __name__ == "__main__":
	model_name = "sentence-transformers/all-MiniLM-L6-v2"

	# or using SentenceTransformers
	model = SentenceTransformer(model_name)

	# select the desired tasks and evaluate
	tasks = mteb.get_tasks(tasks=["AmazonReviewsClassification"], languages=["eng"])
	evaluation = mteb.MTEB(tasks=tasks)
	results = evaluation.run(model)

	for result in results:
		print(result)