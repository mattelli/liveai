import numpy as np
import faiss  # pip install faiss-cpu

# -----------------------------
# 1. Create sample vector data
# -----------------------------
# Let's say we have 1000 vectors of dimension 128
np.random.seed(42)
dimension = 128
num_vectors = 1000

# Random float32 vectors
data_vectors = np.random.random((num_vectors, dimension)).astype('float32')

# -----------------------------
# 2. Build a FAISS index
# -----------------------------
# Using a simple L2 (Euclidean) index
index = faiss.IndexFlatL2(dimension)  # exact search
print("Is trained?", index.is_trained)  # Flat indexes are always trained

# Add vectors to the index
index.add(data_vectors)
print(f"Number of vectors in index: {index.ntotal}")

# -----------------------------
# 3. Perform a similarity search
# -----------------------------
# Create a random query vector
query_vector = np.random.random((1, dimension)).astype('float32')

# Search for the top 5 most similar vectors
k = 5
distances, indices = index.search(query_vector, k)

print("\nTop 5 similar vectors:")
for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
    print(f"{rank}. Index: {idx}, Distance: {dist:.4f}")

# -----------------------------
# 4. Example with cosine similarity
# -----------------------------
# FAISS works with L2 by default, but cosine similarity can be done
# by normalizing vectors to unit length before indexing/searching.
def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

# Normalize data and query
data_vectors_norm = normalize(data_vectors)
query_vector_norm = normalize(query_vector)

# Cosine similarity via inner product
index_cosine = faiss.IndexFlatIP(dimension)
index_cosine.add(data_vectors_norm)
distances_cos, indices_cos = index_cosine.search(query_vector_norm, k)

print("\nTop 5 similar vectors (cosine similarity):")
for rank, (idx, score) in enumerate(zip(indices_cos[0], distances_cos[0]), start=1):
    print(f"{rank}. Index: {idx}, Similarity: {score:.4f}")
