# Semantic Match Algorithm

A semantic matching algorithm used to rank user-submitted sentences based on their embedding similarity. The exact same algorithm we came up for [Up2Mates](https://github.com/ha-wq/up2mates) (currently, [Xalqa](https://t.me/xalqauzbot)).

## Play around with the algorithm!

For a step-by-step local build guide, see the [USAGE](USAGE.md)

## Algorithm Steps

1. **Data Representation**
   Each submission is stored with:

   * **`text`**: the user-provided string (e.g., "I am panda and I LOVE bamboos").
   * **`embedding`**: a fixed-length vector representing the semantic content, computed by a pre-trained SentenceTransformer model (e.g., `paraphrase-MiniLM-L3-v2`).

2. **Embedding Computation**
   On receiving a new submission, concatenate the input into the pattern:

   ```python
   user_text = f"I am {A} and I LOVE {B}"
   embedding = model.encode([user_text])[0]  # NumPy array
   ```

   Store both `user_text` and its `embedding`.

3. **Similarity Measurement**
   To rank all entries relative to a chosen query entry (with embedding `q`):

   1. Gather embeddings of all other entries into a NumPy array **E** of shape `(N, D)`.
   2. Compute cosine similarities in one vectorized step:

      ```python
      # q: (D,), E: (N, D)
      q_norm = np.linalg.norm(q)
      E_norms = np.linalg.norm(E, axis=1)
      similarities = (E @ q) / (E_norms * q_norm + 1e-10)  # shape (N,)
      ```
   3. Sort indices by descending similarity:

      ```python
      order = np.argsort(similarities)[::-1]
      ranked_texts = [others[i].text for i in order]
      ```
   4. Prepend the query text if desired, yielding:

      ```python
      semantic_list = [query_text] + ranked_texts
      ```

4. **Efficiency Considerations**

   * **Batch encoding**: compute embeddings for multiple texts in one call to leverage model optimizations.
   * **Vectorization**: use matrix multiplication and array norms to avoid Python loops, ensuring fast similarity computation even for hundreds of entries.

5. **Extensibility**

   * Swap in any transformer-based embedding model without changing the core logic.
   * Adapt the similarity metric (e.g., Euclidean distance) by replacing the cosine formula.

---

*This algorithm forms the core of semantic ranking and can be embedded into any service or application to provide instant, relevance-based sorting of text entries.*
