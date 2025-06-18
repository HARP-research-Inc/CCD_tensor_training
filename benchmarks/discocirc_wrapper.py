import numpy as np
from mteb.encoder_interface import PromptType

class DisCoCircWrapper:
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
		pass