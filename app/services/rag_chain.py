from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.services.query_rewriter import QUERY_REWRITE_PROMPT
from app.services.memory import get_memory

from app.services.retriever import get_retriever
from app.services.prompt import RAG_PROMPT


def get_llm():
    """
    Local LLM via Ollama (stable, no quotas).
    """
    return ChatOllama(
        model="mistral:7b-instruct-q4_0",
        temperature=0.2,
    )

def rewrite_question(llm, history, question):
    if not history:
        return question

    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in history
    )

    rewrite_chain = QUERY_REWRITE_PROMPT | llm | StrOutputParser()
    return rewrite_chain.invoke({
        "history": history_text,
        "question": question
    })


def get_rag_chain():
    """
    LCEL-based RAG chain.
    Responsible ONLY for answer generation.
    """
    retriever = get_retriever(k=5)
    llm = get_llm()

    chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain
