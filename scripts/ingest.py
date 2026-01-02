from app.utils.loader import load_pdf
from app.utils.splitter import split_docs
from app.services.vectorstore import get_vectorstore

PDF_PATH = "sample.pdf"

docs = load_pdf(PDF_PATH)
chunks = split_docs(docs)

vectorstore = get_vectorstore()
vectorstore.add_documents(chunks)

print(f"✅ Indexed {len(chunks)} chunks successfully")
