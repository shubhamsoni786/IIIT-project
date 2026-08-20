import fitz

pdf_path = "data/sample.pdf"

doc = fitz.open(pdf_path)

print("Number of pages:", len(doc))

for page in doc:
    text = page.get_text()
    print(text)

doc.close()