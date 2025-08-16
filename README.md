# Local-RAG
VaultRAG is a local Retrieval-Augmented Generation app that combines mxbai-embed-large embeddings, PyTorch similarity search, and an Ollama/Llama model behind a Streamlit UI to answer questions from your document vault with source-aware replies.

It caches embeddings to disk (embeddings.pkl + flag) and uses PyTorch cosine-similarity + top-k retrieval (k=3) to keep retrieval fast and prompts compact, dramatically reducing repeated embedding compute and model context size.

Running inference against a local OpenAI/Ollama endpoint and preserving both raw and context-enriched conversation history produces low-latency, auditable responses while minimizing API/compute overhead.
