"""
This script demonstrates how to generate sentence embeddings using a pretrained
SentenceTransformer model and compare sentence similarity using cosine similarity.

Step-by-step:

1. Suppress Logging Noise
   - Reduces unnecessary warnings from the transformers library.

2. Load Sentence Embedding Model
   - SentenceTransformer("all-MiniLM-L6-v2"):
       * A lightweight, fast model for generating sentence embeddings.
       * Converts sentences into dense numerical vectors (embeddings).
       * These embeddings capture semantic meaning.

3. Define First Set of Sentences
   - sentences1 is a list of example sentences.

4. Encode Sentences into Embeddings
   - model.encode(...):
       * Converts each sentence into a vector (embedding).
       * convert_to_tensor=True returns PyTorch tensors instead of lists.

5. Define Second Set of Sentences
   - sentences2 is another list of sentences to compare against.

6. Encode Second Set
   - Same encoding process applied to sentences2.

7. Compute Similarity
   - util.cos_sim(embeddings1, embeddings2):
       * Computes cosine similarity between all pairs of embeddings.
       * Output is a matrix:
           rows = sentences1
           columns = sentences2

8. Print Similarity Matrix
   - Shows how similar each sentence in list 1 is to each sentence in list 2.

9. Compare Matching Pairs
   - The loop compares sentence pairs at the same index:
       sentences1[i] vs sentences2[i]
   - cosine_scores[i][i] extracts the similarity score for each pair.

Key Concepts:

- Sentence Embeddings:
  Numerical vector representations of sentences capturing meaning.

- Semantic Similarity:
  Similar sentences will have embeddings that are close in vector space.

- Cosine Similarity:
  Measures similarity between two vectors:
      1.0 → identical meaning
      0.0 → unrelated
     -1.0 → opposite (rare in embeddings)

- Similarity Matrix:
  A grid showing similarity between every pair of sentences.

Example Interpretation:
- "The cat sits outside" vs "The dog plays in the garden"
  → likely moderately similar (both about animals outdoors)

- "A man is playing guitar" vs "A woman watches TV"
  → lower similarity (different activities)

Use Cases:

- Semantic search
- Document clustering
- Recommendation systems
- Duplicate detection

This script shows a core NLP pattern:
→ Convert text → embeddings → compare using vector similarity
"""

from transformers.utils import logging
logging.set_verbosity_error()

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences1 = ['The cat sits outside',
              'A man is playing guitar',
              'The movies are awesome']

embeddings1 = model.encode(sentences1, convert_to_tensor=True)

embeddings1

sentences2 = ['The dog plays in the garden',
              'A woman watches TV',
              'The new movie is so great']

embeddings2 = model.encode(sentences2, 
                           convert_to_tensor=True)

print(embeddings2)

from sentence_transformers import util

cosine_scores = util.cos_sim(embeddings1,embeddings2)

print(cosine_scores)

for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i],
                                                 sentences2[i],
                                                 cosine_scores[i][i]))