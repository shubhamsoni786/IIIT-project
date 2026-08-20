import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF open karo
doc = fitz.open("data/sample.pdf")

# Saara text ek string me collect karo
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

print("\nFirst Chunk:\n")
print(chunks[0])