# Smart Chat

A CLI tool for smart chat.

## LLM

Model is `h2o-danube-1.8b-chat.Q8_0.gguf`

## RAG

Embeddings in Milvus. Model is `mxbai-embed-large-v1-f16.gguf`

## Flow

1. User sends prompt.
2. Similarity search (match must meet a certain threshold).
3. Pick top match.
4. Make final prompt, context + ask to summarize.
