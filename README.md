# PDF RAG Chatbot with Caching and Reranking

A single-session Streamlit app for asking questions about an uploaded PDF using retrieval-augmented generation.

## Architecture

The app chunks an uploaded PDF with `RecursiveCharacterTextSplitter` using 1,000-character chunks and 200-character overlap. Chunks are embedded with Mistral embeddings and stored in a per-session Pinecone index.

The repository contains two retrieval pipelines:

- **Baseline:** plain similarity search with `k=3`.
- **Advanced:** MMR retrieval, Pinecone hosted `bge-reranker-v2-m3`, then the top 3 reranked chunks.

The advanced path also caches query results and uses a similarity threshold to avoid redundant reranker calls. LangGraph provides the tool-calling agent architecture; LangChain handles orchestration, and Groq runs `openai/gpt-oss-120b` as the LLM.

## Tech Stack

Streamlit, LangGraph, LangChain, Groq, Mistral embeddings, Pinecone, and Pinecone hosted reranking.

## Run It

```bash
pip install -r requirements.txt
streamlit run app.py
```

Configure the API keys required by the app before starting it.

## RAGAS Results

A 47-question RAGAS comparison was run against the same PDF and questions for both pipelines.

| Metric | Baseline | Advanced | Improvement |
|---|---|---|---|
| Faithfulness | 0.7815 | 0.8082 | +0.0267 |
| Answer Relevancy | 0.8085 | 0.8447 | +0.0361 |
| Context Precision | 0.8278 | 0.9345 | +0.1067 |
| Context Recall | 0.8444 | 0.8849 | +0.0405 |

Context Precision jumped more because reranking changes which chunks are selected as context, so precision is the metric it should affect most. Faithfulness, relevancy, and recall depend more on the generation step, so their smaller movement is the expected signature of a correctly scoped improvement, not a weak result.

The evaluation script lives in `evaluation/` and was generated with AI assistance.

## Known Limitations

This is intentionally a lightweight single-session demo: it has no authentication and does not keep history across sessions. It is not designed as a production multi-tenant application.

## What I’d Build Next

I’d add authentication, persistent conversation history, and tenant-isolated document and index management while keeping the baseline and advanced pipelines available for ongoing evaluation.
