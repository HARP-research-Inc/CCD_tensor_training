import numpy as np
from mteb.encoder_interface import PromptType
from src.DisCoBERT.DisCoBERT import DisCoBERT
import mteb

class DisCoBERTWrapper:
	def __init__(self):
		self.model = DisCoBERT("en_core_web_lg")
	def encode(
		self,
		sentences: list[str],
		task_name: str,
		prompt_type: PromptType | None = None,
		**kwargs,
	) -> np.ndarray:
		"""Encodes the given sentences using the encoder.

		Args:
			sentences: The sentences to encode.
			task_name: The name of the task.
			prompt_type: The prompt type to use.
			**kwargs: Additional arguments to pass to the encoder.

		Returns:
			The encoded sentences.
		"""

		matrix = np.zeros((len(sentences), 384))



		for i, sentence in enumerate(sentences):
			matrix[i] = self.model.encode(sentence[i])
		
		return matrix

if __name__ == "__main__":
	# or using SentenceTransformers
	model = DisCoBERTWrapper()

	# select the desired tasks and evaluate
	tasks = mteb.get_tasks(tasks=["AmazonReviewsClassification"], languages=["eng"])
	evaluation = mteb.MTEB(tasks=tasks)
	results = evaluation.run(model)

	for result in results:
		print(result)