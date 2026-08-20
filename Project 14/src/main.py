from pdf_loader import load_pdf
from chunker import create_chunks
from embedder import create_embeddings
from vector_store import create_index
from retriever import search
from chatbot import ask_gemini

text = load_pdf("../data/sample.pdf")

chunks = create_chunks(text)

embeddings = create_embeddings(chunks)

index = create_index(embeddings)

question = input("Enter your question: ")

results = search(index, chunks, question)

context = "\n\n".join(results)

answer = ask_gemini(question, context)

print(answer)