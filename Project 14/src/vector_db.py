import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# PDF Read
doc = fitz.open("data/sample.pdf")

text = ""

for page in doc:
    text += page.get_text()

doc.close()

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

print("Chunks:", len(chunks))

# Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

# FAISS Database
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

print("Vectors Stored:", index.ntotal)