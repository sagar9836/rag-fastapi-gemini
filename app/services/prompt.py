from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a factual assistant.

Answer the question ONLY using the context below.
If the answer is not present, say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{question}
""")
