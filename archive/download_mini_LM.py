from sentence_transformers import SentenceTransformer

# Load and download the model automatically
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Model downloaded successfully!")