import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

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

print("Total Chunks:", len(chunks))

# Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create Embeddings
embeddings = model.encode(chunks)

print("\nEmbedding Shape:", embeddings.shape)

print("\nFirst Embedding (first 10 values):")
print(embeddings[0][:10])