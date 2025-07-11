Yes – one promising strategy is to “bootstrap” a higher‐order, contextual verb tensor from sentence‐level embeddings produced by a model like BERT. The idea is to leverage empirical co-occurrence data and modern contextual embeddings to instantiate the DisCoCat notion that a transitive verb should be a multilinear map (a 3‑tensor) that takes a subject and an object and outputs a sentence meaning. Here’s one way to implement this:

---

### 1. Data Collection

1. **Select Prototypical Subjects and Objects:**  
   For a given verb (say “eat”), choose the top \(N\) subjects and top \(N\) objects that most frequently co-occur with it. For example, you might set  
   \[
   N = 1000,
   \]  
   so that you have 1000 candidate subject nouns (e.g. “child”, “dog”, “person”, …) and 1000 candidate object nouns (e.g. “apple”, “sandwich”, …).

2. **Programmatic Sentence Generation:**  
   Construct minimal sentences using your verb with every combination of subject and object. For instance, generate sentences of the form:  
   \[
   \texttt{"[SUBJECT] eat [OBJECT]"}
   \]  
   This yields \(1000 \times 1000 = 1\,000\,000\) sentences for the verb “eat.”

---

### 2. Sentence Embedding via a Contextual Model

1. **Embed Each Sentence:**  
   Feed each of these sentences into a contextual embedding model (e.g. BERT) to obtain a sentence embedding vector  
   \[
   s_{ij} \in \mathbb{R}^{s},
   \]  
   where \(i\) indexes the subject and \(j\) indexes the object. The dimension \(s\) is the chosen sentence embedding dimension (for example, 768 if you use BERT-base).

2. **Assemble a 3‑Tensor:**  
   Arrange the resulting embeddings into a tensor  
   \[
   T \in \mathbb{R}^{N \times s \times N}
   \]  
   so that  
   \[
   T(i, \cdot, j) = s_{ij}\,.
   \]  
   In other words, the first “mode” corresponds to subjects, the second to the sentence embedding space, and the third to objects.

---

### 3. Recovering a Verb Tensor Representation

The goal is to obtain a continuous 3‑tensor \(V \in \mathbb{R}^{d \times s \times d}\) that plays the role of the verb in the compositional model. (Here, \(d\) is the dimension of noun embeddings in your chosen noun space.) There are several ways to proceed:

#### **A. Tensor Regression / Multilinear Map Estimation**

- **Assumption:** You posit that the sentence meaning for a transitive sentence is generated via a bilinear map  
  \[
  s \approx a^\top V\, b
  \]  
  where \(a \in \mathbb{R}^d\) is the subject embedding and \(b \in \mathbb{R}^d\) is the object embedding, and the contraction \(a^\top V\, b\) (with the appropriate index contractions) yields a vector in \(\mathbb{R}^{s}\).

- **Procedure:**  
  For each of the \(i,j\) pairs you already have:  
  - \(a_i \in \mathbb{R}^d\): the embedding for the \(i\)th subject (from your noun embedding model)  
  - \(b_j \in \mathbb{R}^d\): the embedding for the \(j\)th object  
  - \(s_{ij} \in \mathbb{R}^{s}\): the sentence embedding from BERT

  Then, solve a regression (e.g. via least-squares) to find the tensor \(V\) that minimizes  
  \[
  \sum_{i,j} \left\| s_{ij} - \left(a_i^\top\, V\, b_j\right) \right\|^2\,.
  \]
  
  In practice, one might impose low-rank constraints (using CP or Tucker decompositions) on \(V\) to ensure generalization.

#### **B. Direct Tensor Construction via Outer-Products**

- **Simplest Baseline:**  
  If you start with a traditional verb embedding \(v \in \mathbb{R}^{d}\) (e.g. learned from a word-level model) and you want to “lift” it into a 3‑tensor, you could define:
  \[
  V = v \otimes w \otimes v\,,
  \]
  where \(w \in \mathbb{R}^{s}\) is a fixed “verb context” vector (or even the average over the sentence embeddings). Although this is very rigid (yielding a rank‑1 tensor), it provides a baseline.

- **Hybrid Approach:**  
  Use the empirical tensor \(T\) as guidance. For example, project the discrete \(T\) onto a continuous tensor model by aligning the subject and object modes with their continuous embeddings. That is, if you have mapping matrices \(U \in \mathbb{R}^{N \times d}\) and \(W \in \mathbb{R}^{N \times d}\) that map the \(N\) discrete subject and object indices into the noun embedding space, then you can “lift” \(T\) via:
  \[
  V_{ijk} \approx \sum_{p,q} U_{ip}\, T(p, k, q)\, W_{qj}\,,
  \]
  where now \(V\) is in \(\mathbb{R}^{d \times s \times d}\).

#### **C. Neural Network (MLP) Based Approach**

- **Idea:**  
  Train a neural network that takes as input a subject–object pair \((a, b)\) (concatenated or combined bilinearly) and outputs the corresponding sentence embedding \(s\). The architecture of the network can be designed so that its first (or last) layer has a multilinear structure that factors as a tensor \(V\).

- **Outcome:**  
  In this case, once the network is trained on the 1 million (or a subset) generated examples, you can “read off” an effective verb tensor \(V\) from the weights that govern the bilinear interaction.

---

### 4. Summary of How It Works

1. **Data Construction:**  
   - Choose top co-occurrence subjects and objects.  
   - Programmatically generate sentences (e.g. “subject verb object”).  
   - Compute sentence embeddings (using BERT, etc.) to fill a tensor \(T \in \mathbb{R}^{1000 \times s \times 1000}\).

2. **Tensor Lifting / Factorization:**  
   - Use regression or tensor decomposition methods to recover a continuous 3‑tensor \(V \in \mathbb{R}^{d \times s \times d}\) that “explains” the observed sentence embeddings when contracted with subject and object vectors.

3. **Usage in Compositional Semantics:**  
   - The learned \(V\) now acts as the verb’s meaning in the compositional pipeline. Given any subject vector \(a\) and object vector \(b\) (in \(\mathbb{R}^{d}\)), you obtain a sentence meaning by computing  
     \[
     s \approx a^\top V\, b \,,
     \]
     which produces a vector in the sentence space \(\mathbb{R}^{s}\).

This method effectively “grounds” the verb’s meaning in the distribution of its contextual usage (via subject and object combinations) as captured by a modern contextual model. It is a data-driven way to produce a higher-order tensor that aligns with the compositional requirements of the DisCoCat framework.

Below is one example implementation in Python using PyTorch and the SentenceTransformers library. In this toy example, we assume a small number of subjects and objects (for example, 5 each) and “lift” the verb embedding into a 3‑tensor by regressing from BERT-generated sentence embeddings. (In a full‐scale experiment you’d use, say, 1000 subjects and 1000 objects.) The code proceeds as follows:

1. Generate sentences of the form “subject VERB object.”
2. Use a SentenceTransformer (here, “all‑MiniLM‑L6‑v2”) to get a sentence embedding for each generated sentence.
3. Also embed the subject and object words separately.
4. Project these (originally high‑dimensional) embeddings into a lower-dimensional space (here we choose \(d = s = 50\)) via fixed linear projections.
5. Define a learnable verb tensor \(V \in \mathbb{R}^{d \times s \times d}\) so that for each subject–object pair the predicted sentence embedding is computed via  
   \[
   \hat{s}_{ij} = a_i^\top\, V\, b_j \quad\text{(implemented as an Einstein summation)}
   \]
6. Train \(V\) by minimizing the mean-squared error between the predicted sentence embeddings and those from BERT.

You can run the code below (after installing the required packages, e.g. via `pip install torch sentence-transformers`) as a demonstration:

---

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer

# --- Step 1: Define subjects, objects, and the verb ---
subjects = ["child", "dog", "man", "woman", "student"]
objects  = ["apple", "sandwich", "pizza", "salad", "burger"]
verb = "eat"

# Generate sentences of the form "subject verb object"
sentences = [f"{subj} {verb} {obj}" for subj in subjects for obj in objects]

# --- Step 2: Obtain sentence embeddings via a SentenceTransformer ---
# Using a lightweight model; the output dimension here is 384.
model = SentenceTransformer('all-MiniLM-L6-v2')
with torch.no_grad():
    sent_embs = model.encode(sentences, convert_to_tensor=True)  # shape: (25, 384)

# Reshape sentence embeddings to a tensor of shape (num_subjects, num_objects, embedding_dim)
num_subjects = len(subjects)
num_objects = len(objects)
sent_embs = sent_embs.view(num_subjects, num_objects, -1)  # (5, 5, 384)

# --- Step 3: Get separate embeddings for subjects and objects ---
with torch.no_grad():
    subj_embs = model.encode(subjects, convert_to_tensor=True)  # (5, 384)
    obj_embs  = model.encode(objects, convert_to_tensor=True)   # (5, 384)

# --- Step 4: Project embeddings to a lower-dimensional space ---
# We choose d = s_dim = 50 for this toy example.
d = 50         # Dimension for noun (subject/object) embeddings.
s_dim = 50     # Dimension for sentence embeddings in our compositional model.
input_dim = sent_embs.shape[-1]  # 384 from the transformer model

# Create fixed (non-trainable) linear projections for subjects, objects, and sentences.
proj_subj = nn.Linear(input_dim, d, bias=False)
proj_obj  = nn.Linear(input_dim, d, bias=False)
proj_sent = nn.Linear(input_dim, s_dim, bias=False)

# (Optionally, one could freeze these layers. Here we set requires_grad=False.)
for param in proj_subj.parameters():
    param.requires_grad = False
for param in proj_obj.parameters():
    param.requires_grad = False
for param in proj_sent.parameters():
    param.requires_grad = False

# Apply projections.
subj_embs_proj = proj_subj(subj_embs)  # shape: (5, d)
obj_embs_proj  = proj_obj(obj_embs)      # shape: (5, d)
# For sentences, flatten and then reshape back:
sent_embs_proj = proj_sent(sent_embs.view(-1, input_dim)).view(num_subjects, num_objects, s_dim)  # (5,5,s_dim)

# --- Step 5: Define the learnable verb tensor ---
# Our goal is to learn V in R^(d x s_dim x d)
V = nn.Parameter(torch.randn(d, s_dim, d), requires_grad=True)

# --- Step 6: Set up the regression and training loop ---
optimizer = optim.Adam([V], lr=0.01)
criterion = nn.MSELoss()

num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    # For each subject-object pair, compute the predicted sentence embedding.
    # We use torch.einsum with the convention:
    #   subj_embs_proj: shape (num_subjects, d)
    #   V: shape (d, s_dim, d)
    #   obj_embs_proj: shape (num_objects, d)
    # Then, for each pair (i, j):
    #   pred[i, j, :] = einsum('p,ps,q->s', subj_embs_proj[i], V, obj_embs_proj[j])
    pred = torch.einsum('ip,ps,jq->ijs', subj_embs_proj, V, obj_embs_proj)  # shape: (5, 5, s_dim)
    
    loss = criterion(pred, sent_embs_proj)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:4d}, Loss: {loss.item():.4f}")

print("Training complete. Learned verb tensor V shape:", V.shape)
```

---

### Explanation

- **Data Setup:**  
  We start with a small set of subjects and objects. For each pair we form a sentence (e.g. “child eat apple”) and obtain its BERT-based embedding.

- **Projection:**  
  Since the raw transformer embeddings are high-dimensional (384), we project subjects, objects, and sentences down to a lower dimension (here 50) so that our verb tensor \(V\) has a manageable number of parameters. In a production system these projections might be learned jointly, but here we fix them for clarity.

- **Verb Tensor & Contraction:**  
  The learnable parameter \(V\) (of shape \((d, s\_dim, d)\)) is used to “lift” the subject and object embeddings into a prediction for the sentence embedding via the contraction  
  \[
  \hat{s}_{ij} = \text{einsum}('ip,ps,jq->ijs',\,\text{subj\_embs\_proj},\,V,\,\text{obj\_embs\_proj})\,.
  \]
  
- **Training:**  
  We then minimize the mean squared error (MSE) between the predicted sentence embeddings and those computed directly from the sentences. After training, the tensor \(V\) acts as our contextual, 3‑tensor representation for the verb.

This code provides a concrete demonstration of “lifting” a traditional vector‐embedding version of a verb into a contextual 3‑tensor using sentence-level embeddings. Adjust parameters (such as the number of subjects/objects or projection dimensions) as needed for larger experiments.