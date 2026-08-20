from sentence_transformers import SentenceTransformer

# Cache loaded models so they aren't downloaded every time
_models = {}

def get_model(model_name):
    if model_name not in _models:
        _models[model_name] = SentenceTransformer(model_name)
    return _models[model_name]


def create_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    model = get_model(model_name)
    embeddings = model.encode(chunks)
    return embeddings