import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gc
import random
import time

import sys
from pathlib import Path

DEVICE = 3

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

class CPTensorRegression(nn.Module):
    def __init__(self, input_dims: list[int], output_dim: int, rank: int):
        """
        CP-decomposed tensor regression for heterogeneous input dimensions.

        Args:
            input_dims: List of dimensions for each input vector
            output_dim: Output vector dimension
            rank: CP decomposition rank
        """
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.rank = rank
        self.arity = len(input_dims)

        # One factor matrix per input, each of shape (rank, input_dim_i)
        self.input_factors = nn.ParameterList([
            nn.Parameter(torch.randn(rank, dim)) for dim in input_dims
        ])

        # Output factor (rank, output_dim)
        self.output_factor = nn.Parameter(torch.randn(rank, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for param in self.input_factors:
            nn.init.xavier_uniform_(param)
        nn.init.xavier_uniform_(self.output_factor)

    def forward(self, *inputs):
        """
        Forward pass.

        Args:
            inputs: list of tensors with shapes [(batch_size, input_dim_i), ...]
                    Must match `input_dims`.

        Returns:
            Output: (batch_size, output_dim)
        """
        if len(inputs) != self.arity:
            raise ValueError(f"Expected {self.arity} inputs, got {len(inputs)}")
        for i, (inp, dim) in enumerate(zip(inputs, self.input_dims)):
            if inp.shape[1] != dim:
                raise ValueError(f"Input {i} expected dim {dim}, got {inp.shape[1]}")

        # Project each input to (batch_size, rank)
        projections = [
            torch.matmul(inp, factor.T) for inp, factor in zip(inputs, self.input_factors)
        ]

        # Elementwise multiply over rank dimension
        joint = projections[0]
        for proj in projections[1:]:
            joint = joint * proj  # (batch_size, rank)

        # Output projection
        output = joint @ self.output_factor + self.bias  # (batch_size, output_dim)
        return output

class CPTensorTransform(nn.Module):
    def __init__(self, tensor_shape, rank):
        """
        CP-decomposed transformation of a tensor into another tensor of same shape.
        Args:
            tensor_shape: Tuple[int] – shape of the input/output tensor (e.g., (d, d))
            rank: int – CP decomposition rank
        """
        super().__init__()
        self.tensor_shape = tensor_shape
        self.rank = rank
        self.order = len(tensor_shape)

        # One factor matrix per mode
        self.factors = nn.ParameterList([
            nn.Parameter(torch.randn(rank, dim)) for dim in tensor_shape
        ])

        # Output factor: how the rank components are recombined
        self.output_weights = nn.Parameter(torch.randn(rank))

    def forward(self, x):
        """
        Apply CP-transformation to input tensor `x`.
        x: Tensor of shape `tensor_shape`, or batch of them: (B, *tensor_shape)
        Returns:
            Tensor of shape `tensor_shape` (or batched)
        """
        B = x.shape[0] if x.dim() > len(self.tensor_shape) else None
        # Project along each mode
        projections = []

        for mode, factor in enumerate(self.factors):
            # Factor: (rank, dim)
            if B is None:
                proj = torch.tensordot(x, factor.T, dims=([mode], [1]))  # (..., rank)
            else:
                proj = torch.tensordot(x, factor.T, dims=([mode+1], [1]))  # (B, ..., rank)
            projections.append(proj)

        # Elementwise multiply all projections
        joint = projections[0]
        for proj in projections[1:]:
            joint = joint * proj

        # Weighted sum over rank dimension
        output = torch.sum(joint * self.output_weights, dim=-1)  # final dims match tensor_shape

        return output

class TwoWordTensorRegression(nn.Module):
	def __init__(self, noun_dim, sent_dim):
		"""
		Regression initialization
		"""
		super().__init__()
		self.sent_dim = sent_dim
		self.noun_dim = noun_dim

		self.V = nn.Parameter(torch.randn(sent_dim, noun_dim, noun_dim))
		self.bias = nn.Parameter(torch.zeros(sent_dim))

		# Optional: initialize slice-by-slice
		for i in range(sent_dim):
			torch.nn.init.xavier_uniform_(self.V[i])

	def forward(self, s, o):
		"""
		Optionally normalize inputs
		
		Args:
			s = F.normalize(s, dim=1)
			o = F.normalize(o, dim=1)
		
		Returns:

		"""

		Vs_o = torch.einsum('ljk,bj,bk->bl', self.V, s, o)
		return Vs_o + self.bias

class ThreeWordTensorRegression(nn.Module):
	def __init__(self, input_dim, output_dim):
		"""
		Regression initialization for mapping three input vectors to one output vector.
		"""
		super().__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim

		# Tensor for mapping three input vectors to one output vector
		self.V = nn.Parameter(torch.randn(output_dim, input_dim, input_dim, input_dim))
		self.bias = nn.Parameter(torch.zeros(output_dim))

		# Initialize slice-by-slice
		for i in range(output_dim):
			torch.nn.init.xavier_uniform_(self.V[i])

	def forward(self, x1, x2, x3):
		"""
		Forward pass for three input vectors.

		Args:
			x1, x2, x3: Input tensors of shape (batch_size, input_dim)

		Returns:
			Output tensor of shape (batch_size, output_dim)
		"""
		Vx1x2x3 = torch.einsum('lijk,bi,bj,bk->bl', self.V, x1, x2, x3)
		return Vx1x2x3 + self.bias

class OneWordTensorRegression(nn.Module):
	def __init__(self, input_dim, output_dim):
		"""
		Regression initialization for mapping one input vector to one output vector.
		"""
		super().__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim

		# Tensor for mapping one input vector to one output vector
		self.V = nn.Parameter(torch.randn(output_dim, input_dim))
		self.bias = nn.Parameter(torch.zeros(output_dim))

		torch.nn.init.xavier_uniform_(self.V)

	def forward(self, x):
		"""
		Forward pass for one input vector.

		Args:
			x: Input tensor of shape (batch_size, input_dim)

		Returns:
			Output tensor of shape (batch_size, output_dim)
		"""
		Vx = torch.einsum('ik,bk->bi', self.V, x)
		return Vx + self.bias


def parallel_shuffle(data1, data2):
	"""
	Randomly shuffle two datasets in parallel.

	Args:
		data1: First dataset to shuffle
		data2: Second dataset to shuffle
	Modifies:
		data1, data2
	"""
	if len(data1) != len(data2):
		raise ValueError("Datasets must have the same length to shuffle in parallel.")

	indices = list(range(len(data1)))
	random.shuffle(indices)

	# Shuffle both datasets using the same indices
	if isinstance(data1, torch.Tensor):
		data1[:] = torch.stack([data1[i] for i in indices])
	else:
		data1[:] = [data1[i] for i in indices]

	if isinstance(data2, torch.Tensor):
		data2[:] = torch.stack([data2[i] for i in indices])
	else:
		data2[:] = [data2[i] for i in indices]

#############################################
############# k-word regression #############
#############################################

def k_word_regression(model_destination, embedding_set, ground_truth, tuple_len, module: nn.Module,
					num_epochs = 50, word_dim = 100, sentence_dim = 300, lr = 0.1, shuffle = False, device = 3):
	"""
	Regression function meant to handle differnt word len regressions. Produces
	linear map between k embeddings and one ground_truth tensor. Thus far only
	supports 1, 2, and 3 word regressions.

	Args: 
		model_destination: file path of final regression model (.pt file preferred)
		embedding_set: list of tuples of raw dependent data embeddings
		ground_truth: tensor containing empirical contextual sentence embedding 
		num_epochs: number of epochs to train
		word_dim: dimension word embeddings 
		sentence_dim: dimension of sentence embeddings

	Throws:
		Exception if ground truth data is of different len to word embedding data len
	"""

	tm = time.time()

	t = ground_truth # t stores ground truth data
	s_o = embedding_set #s_o stores word data
	device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
	if device == f"cuda:{device}":
		torch.cuda.empty_cache() 
	module.to(device)

	if len(t) != len(s_o):
		raise Exception("Mismatched data dimensions")
	
	#print(">shuffling data...")
	if shuffle:
		parallel_shuffle(s_o, t)
	#print(">done!\n\n\n")

	num_nouns = len(t)
	test_size = num_nouns // 5  # 20% of the data
	train_size = num_nouns - test_size  # Remaining 80%

	# Allocating space for testing and training tensors
	#print(">allocating space for testing and training tensors...")
	s_o_tensor = torch.zeros((train_size, tuple_len, word_dim)).to(device)
	test_s_o_tensor = torch.zeros((test_size, tuple_len, word_dim)).to(device)
	
	ground_truth = torch.zeros((train_size, sentence_dim)).to(device)
	ground_truth_test = torch.zeros((test_size, sentence_dim)).to(device)
	#print(">done!\n\n\n")

	# Partitioning between testing and training sets
	#print(">Partitioning between testing and training sets...")
	noun_pairs_test = s_o[:test_size]
	sentence_test = t[:test_size]

	verb1_noun_pairs = s_o[test_size:]
	verb1_sentences = t[test_size:]
	#print(">done!\n\n\n")

	# Assembling training tensors
	#print(">Assembling training tensors...")
	for i, noun_tup in enumerate(verb1_noun_pairs):
		for tup_i in range(0, tuple_len):
			s_o_tensor[i][tup_i] = noun_tup[tup_i]
		ground_truth[i] = torch.Tensor(verb1_sentences[i])
	#print(">done!\n\n\n")
	
	# Assembling test tensors
	#print(">Assembling testing tensors...")
	for i, noun_tup in enumerate(noun_pairs_test):
		for tup_i in range(0, tuple_len):
			test_s_o_tensor[i][tup_i] = noun_tup[tup_i]
		ground_truth_test[i] = torch.Tensor(sentence_test[i])
	#print(">done!\n\n\n")
	
	#utilizing Adadelta regularization
	optimizer = optim.Adadelta(module.parameters(), lr=lr)

	#print(">Running regression...")

	nouns_train = list()
	for i in range(tuple_len):
		#print(i, tuple_len, s_o_tensor.shape)
		nouns_train.append(s_o_tensor[:, i, :])

	for epoch in range(num_epochs):
		optimizer.zero_grad()

		# Forward pass
		#to-do: make more dynamic w/ list
		if tuple_len == 3:
			predicted = module(nouns_train[0],nouns_train[1], nouns_train[2])
		elif tuple_len == 2:
			predicted = module(nouns_train[0],nouns_train[1])
		elif tuple_len == 1:
			predicted = module(nouns_train[0])
		else:
			raise Exception("Unsupported tuple length")

		# Compute loss (Mean Squared Error)
		loss = torch.mean((predicted - ground_truth)**2)

		# Backward and optimize
		loss.backward()
		optimizer.step()

		# Print loss for each epoch
		#print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.20f}')

		# Debugging: Check if weights are being updated
		# if epoch == 0 or epoch == num_epochs - 1:
		#	 print(f"Sample weights at epoch {epoch + 1}: {list(model.parameters())[0][0][:5]}")

	#print(f'>done! Final In-Sample Loss: {loss.item():.20f}\n\n\n')

	# Save model weights
	torch.save(module.state_dict(), model_destination)
	print(f"Model weights saved to: {model_destination}")


	"""************************testing************************"""


	print(">************************testing************************")

	nouns_test = list()
	for i in range(tuple_len):
		nouns_test.append(test_s_o_tensor[:, i, :])

	if tuple_len == 3:
		predicted_test = module(nouns_test[0],nouns_test[1], nouns_test[2])
	elif tuple_len == 2:
		predicted_test = module(nouns_test[0],nouns_test[1])
	elif tuple_len == 1:
		predicted_test = module(nouns_test[0])
	else:
		raise Exception("Unsupported tuple length")

	

	loss = torch.mean((predicted_test - ground_truth_test)**2)

	print(f'>done! Test Sample Loss: {loss.item():.4f}\n\n\n')
	print(f"Took {time.time() - tm} seconds")

def multi_word_regression(model_destination, word_embeddings, sentence_embeddings, tuple_len, module: nn.Module,
					num_epochs = 50, word_dim = 100, sentence_dim = 300, lr = 0.1, shuffle = False, device = 3):
	"""
	Regression function meant to handle differnt word len regressions. Produces
	linear map between k embeddings and one ground_truth tensor.

	Args: 
		model_destination: file path of final regression model (.pt file preferred)
		embedding_set: list of tuples of raw dependent data embeddings
		ground_truth: tensor containing empirical contextual sentence embedding 
		num_epochs: number of epochs to train
		word_dim: dimension word embeddings 
		sentence_dim: dimension of sentence embeddings

	Throws:
		Exception if ground truth data is of different len to word embedding data len
	"""

	tm = time.time()

	device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
	if device == f"cuda:{device}":
		torch.cuda.empty_cache() 
	module.to(device)

	if len(word_embeddings) != len(sentence_embeddings):
		raise Exception("Mismatched data dimensions")
	
	# print(">shuffling data...")
	if shuffle:
		parallel_shuffle(sentence_embeddings, word_embeddings)
	#print(">done!\n\n\n")

	test_size = len(sentence_embeddings) // 10  # 20% of the data
	train_size = len(sentence_embeddings) - test_size  # Remaining 80%

	# Allocating space for testing and training tensors
	# print(">allocating space for testing and training tensors...")
	word_tensors_train = torch.zeros((train_size, tuple_len, word_dim)).to(device)
	word_tensors_test = torch.zeros((test_size, tuple_len, word_dim)).to(device)
	
	output_tensors_train = torch.zeros((train_size, sentence_dim)).to(device)
	output_tensors_test = torch.zeros((test_size, sentence_dim)).to(device)
	#print(">done!\n\n\n")

	# Partitioning between testing and training sets
	# print(">Partitioning between testing and training sets...")
	words_train = word_embeddings[:train_size]
	#print("words train", len(words_train))
	sentence_train = sentence_embeddings[:train_size]

	words_test = word_embeddings[train_size:]
	# print("words test", len(words_test))
	sentence_test = sentence_embeddings[train_size:]
	#print(">done!\n\n\n")

	try:
		# Assembling training tensors
		print(">Assembling training tensors...")
		for i, words in enumerate(words_train):
			for j in range(0, tuple_len):
				word_tensors_train[i][j] = words[j]
			output_tensors_train[i] = torch.Tensor(sentence_train[i])
		#print(">done!\n\n\n")
		
		# Assembling test tensors
		print(">Assembling testing tensors...")
		for i, words in enumerate(words_test):
			for j in range(0, tuple_len):
				word_tensors_test[i][j] = words[j]
			output_tensors_test[i] = torch.Tensor(sentence_test[i])
		#print(">done!\n\n\n")
	except:
		import traceback
		print(traceback.format_exc())
		return
	
	#utilizing Adadelta regularization
	optimizer = optim.RMSprop(module.parameters(), lr=lr)

	# print(">Running regression...")

	train_words = list()
	for i in range(tuple_len):
		#print(i, tuple_len, s_o_tensor.shape)
		train_words.append(word_tensors_train[:, i, :])

	for epoch in range(num_epochs):
		optimizer.zero_grad()

		# Forward pass
		#to-do: make more dynamic w/ list
		if tuple_len == len(train_words):
			predicted = module(*train_words)
		else:
			raise Exception("Tuple length mismatch")

		# Compute loss (Mean Squared Error)
		loss = torch.mean((predicted - output_tensors_train)**2)

		# Backward and optimize
		loss.backward()
		optimizer.step()

		# Print loss for each epoch
		# print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.20f}')

		# Debugging: Check if weights are being updated
		# if epoch == 0 or epoch == num_epochs - 1:
		#	 print(f"Sample weights at epoch {epoch + 1}: {list(model.parameters())[0][0][:5]}")

	#print(f'>done! Final In-Sample Loss: {loss.item():.20f}\n\n\n')

	# Save model weights
	torch.save(module.state_dict(), model_destination)
	print(f"Model weights saved to: {model_destination}")


	"""************************testing************************"""


	print(">************************testing************************")

	test_words = list()
	for i in range(tuple_len):
		test_words.append(word_tensors_test[:, i, :])

	if tuple_len == len(test_words):
		predicted_test = module(*test_words)
	else:
		raise Exception("Tuple length mismatch")

	

	loss = torch.mean((predicted_test - output_tensors_test)**2)

	print(f'>done! Test Sample Loss: {loss.item():.4f}\n\n\n')
	print(f"Took {time.time() - tm} seconds")

def batch_word_regression(model_destination, word_embeddings, sentence_embeddings, tuple_len, module: nn.Module,
					num_epochs = 50, word_dim = 100, sentence_dim = 300, lr = 0.1, shuffle = False, device = 3, batch_size = 256):
	"""
	Memory-efficient regression function using batched training with indexing.
	
	Args: 
		model_destination: file path of final regression model (.pt file preferred)
		word_embeddings: list of tuples of raw dependent data embeddings
		sentence_embeddings: tensor containing empirical contextual sentence embedding 
		tuple_len: number of words per sample
		module: neural network module
		num_epochs: number of epochs to train
		word_dim: dimension word embeddings 
		sentence_dim: dimension of sentence embeddings
		lr: learning rate
		shuffle: whether to shuffle data
		device: CUDA device number
		batch_size: batch size for training
	"""
	import time
	import torch
	import torch.optim as optim
	
	tm = time.time()

	device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
	if device == f"cuda:{device}":
		torch.cuda.empty_cache() 
	module.to(device)

	if len(word_embeddings) != len(sentence_embeddings):
		raise Exception("Mismatched data dimensions")
	
	# Create indices for data instead of copying
	indices = list(range(len(sentence_embeddings)))
	
	if shuffle:
		import random
		random.shuffle(indices)
	
	# Split indices for train/test
	test_size = len(sentence_embeddings) // 10  # 10% for testing
	train_size = len(sentence_embeddings) - test_size
	
	train_indices = indices[:train_size]
	test_indices = indices[train_size:]
	
	print(f"Training samples: {train_size}, Test samples: {test_size}")
	
	# Optimizer
	optimizer = optim.RMSprop(module.parameters(), lr=lr)
	
	print(">Starting batched training...")
	
	# Training loop
	for epoch in range(num_epochs):
		module.train()
		epoch_loss = 0.0
		num_batches = 0
		
		# Process training data in batches
		for batch_start in range(0, len(train_indices), batch_size):
			batch_end = min(batch_start + batch_size, len(train_indices))
			batch_indices = train_indices[batch_start:batch_end]
			current_batch_size = len(batch_indices)
			
			optimizer.zero_grad()
			
			# Create batch tensors on-the-fly
			batch_words = []
			for word_idx in range(tuple_len):
				word_batch = torch.zeros((current_batch_size, word_dim), device=device)
				for i, idx in enumerate(batch_indices):
					word_batch[i] = torch.tensor(word_embeddings[idx][word_idx], device=device)
				batch_words.append(word_batch)
			
			# Target batch
			target_batch = torch.zeros((current_batch_size, sentence_dim), device=device)
			for i, idx in enumerate(batch_indices):
				target_batch[i] = torch.tensor(sentence_embeddings[idx], device=device)
			
			# Forward pass
			predicted = module(*batch_words)
			
			# Compute loss
			loss = torch.mean((predicted - target_batch)**2)
			
			# Backward pass
			loss.backward()
			optimizer.step()
			
			epoch_loss += loss.item()
			num_batches += 1
			
			torch.cuda.empty_cache() if device.type == 'cuda' else None
		
		avg_loss = epoch_loss / num_batches
		if epoch % 10 == 0 or epoch == num_epochs - 1:
			print(f'Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.6f}')
	
	print(f'>Training complete! Final Loss: {avg_loss:.6f}\n')
	
	# Save model weights
	torch.save(module.state_dict(), model_destination)
	print(f"Model weights saved to: {model_destination}")
	
	"""************************testing************************"""
	
	print(">************************testing************************")
	
	module.eval()
	test_loss = 0.0
	num_test_batches = 0
	
	with torch.no_grad():
		# Process test data in batches
		for batch_start in range(0, len(test_indices), batch_size):
			batch_end = min(batch_start + batch_size, len(test_indices))
			batch_indices = test_indices[batch_start:batch_end]
			current_batch_size = len(batch_indices)
			
			# Create test batch tensors
			batch_words = []
			for word_idx in range(tuple_len):
				word_batch = torch.zeros((current_batch_size, word_dim), device=device)
				for i, idx in enumerate(batch_indices):
					word_batch[i] = torch.tensor(word_embeddings[idx][word_idx], device=device)
				batch_words.append(word_batch)
			
			# Target batch
			target_batch = torch.zeros((current_batch_size, sentence_dim), device=device)
			for i, idx in enumerate(batch_indices):
				target_batch[i] = torch.tensor(sentence_embeddings[idx], device=device)
			
			# Forward pass
			predicted = module(*batch_words)
			
			# Compute loss
			loss = torch.mean((predicted - target_batch)**2)
			test_loss += loss.item()
			num_test_batches += 1
			
			# Clean up
			del batch_words, target_batch, predicted
			torch.cuda.empty_cache() if device.type == 'cuda' else None
	
	avg_test_loss = test_loss / num_test_batches
	print(f'>Test Loss: {avg_test_loss:.4f}\n')
	print(f"Total time: {time.time() - tm:.2f} seconds")