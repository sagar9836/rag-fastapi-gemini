from langchain_core.prompts import ChatPromptTemplate

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_template("""
You are a query rewriting assistant.

Conversation history:
{history}

Follow-up question:
{question}

Rewrite the follow-up question into a standalone question.
If it is already standalone, return it unchanged.
""")
