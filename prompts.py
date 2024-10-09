from langchain_core.prompts import ChatPromptTemplate


TEMPLATE_CONTEXTUAL_ANSWER = """
You are an AI assistant tasked with giving answers based on provided context.

context: {context}
question: {question}

Give answers based on provided context. Don't add anything extra. Only stay within the context.

If there is no answer found in the documents then return the parts of the chunks summarized:

chunks: {chunks}

Only return chunks summarized if you cannot find related texts in contexts.
"""

CONTEXTUAL_ANSWER = ChatPromptTemplate.from_messages(
    [("user", TEMPLATE_CONTEXTUAL_ANSWER)]
)
