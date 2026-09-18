from langchain_core.tools import tool
from pinecone import Pinecone
from langchain_mistralai import MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import os

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=os.getenv("MISTRAL_API_KEY")
)

pc = Pinecone(
    api_key=os.getenv("PINECAI_API_KEY") if False else os.getenv("PINECONE_API_KEY")
)

def chunk_reranker(question, docs):
    if not docs:
        return []

    rerank_result = pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=question,
        documents=[doc.page_content for doc in docs],
        top_n=min(3, len(docs)),
        return_documents=True
    )

    scored_docs = [
        (docs[item.index], item.score)
        for item in rerank_result.data
    ]

    return scored_docs

def context_retriever_from_docs(top_docs):
    context = "\n\n".join(
        doc.page_content
        for doc, score in top_docs
    )
    return context

def fresh_retrieval(question, index_name, k, fetch_k):
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k
        }
    )
    docs = retriever.invoke(question)
    top_chunks = chunk_reranker(question=question, docs=docs)
    return top_chunks

def cache_retrieval(question, cache_chunks):
    plain_docs = [doc for doc, score in cache_chunks]
    cache_top_chunks = chunk_reranker(question=question, docs=plain_docs)
    cache_relevant = [
        doc
        for doc, score in cache_top_chunks
        if score > threshold_score
    ]
    return cache_relevant

cache_chunks = []
threshold_score = 0.85

from langchain_core.runnables import RunnableConfig

@tool
def Query_Handler(question: str, config: RunnableConfig) -> str:
    """
    Use this tool to answer any question about a PDF whose content has already been embedded.

    This tool:
    - Searches the Pinecone vector store for relevant chunks
    - Uses Pinecone's hosted reranker to find the most relevant chunks
    - Returns the most relevant context as a string

    Args:
        question: The user's question about the PDF
        index_name: The Pinecone index name returned by Create_embeddings

    Returns:
        A string containing the most relevant context to answer the question.
        Use this context to form your final answer to the user.

    IMPORTANT:
    - Only use this tool if Create_embeddings has already been called for this PDF in the current conversation.
    - Use the index_name returned by Create_embeddings — do not guess or fabricate it.
    - If the returned context does not contain the answer, say "I couldn't find this in the PDF." Do not hallucinate.
    """
    index_name = config["configurable"]["index_name"]
    cache_relevant = cache_retrieval(question=question, cache_chunks=cache_chunks)

    if cache_relevant:
        fresh_chunks_with_score = fresh_retrieval(
            question=question,
            index_name=index_name,
            k=2,
            fetch_k=5
        )
        fresh_docs = [doc for doc, score in fresh_chunks_with_score]
        combined_docs = fresh_docs + cache_relevant
        new_scored = chunk_reranker(question=question, docs=combined_docs)
        cache_chunks.extend(new_scored)
        cache_chunks.sort(key=lambda x: x[1], reverse=True)
        del cache_chunks[10:]
        return context_retriever_from_docs(new_scored)
    else:
        fresh_chunks_with_score = fresh_retrieval(question=question, index_name=index_name, k=5, fetch_k=10)
        fresh_docs = [doc for doc, score in fresh_chunks_with_score]
        new_scored = chunk_reranker(question=question, docs=fresh_docs)
        cache_chunks.extend(new_scored)
        cache_chunks.sort(key=lambda x: x[1], reverse=True)
        del cache_chunks[10:]
        return context_retriever_from_docs(new_scored)