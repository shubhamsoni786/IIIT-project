import numpy as np
from sentence_transformers import SentenceTransformer


# Cache loaded models
_models = {}


def get_model(model_name):

    if model_name not in _models:
        _models[model_name] = SentenceTransformer(model_name)

    return _models[model_name]


def search(
    index,
    chunks,
    question,
    model_name="all-MiniLM-L6-v2",
    k=3
):

    # Make sure we actually received a question
    if not question:
        return []

    model = get_model(model_name)

    query_embedding = model.encode(
        [question]
    )

    distances, indices = index.search(
        np.array(query_embedding),
        k
    )

    results = []

    for i in indices[0]:

        # Avoid invalid FAISS indexes
        if i >= 0 and i < len(chunks):
            results.append(chunks[i])

    return results