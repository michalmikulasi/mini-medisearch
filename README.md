ScholarStream: Context-Aware RAG Engine

ScholarStream is a lightweight Retrieval-Augmented Generation (RAG) pipeline designed to transform dense, unstructured HTML literature into grounded, cited insights. 

While many RAG systems suffer from "source drowning" or "hallucination," this engine implements custom ranking and token management logic to ensure multi-perspective accuracy.


1. Noise-Resilient HTML Parsing
Standard text extraction often pulls in navigation menus and bibliographies which clutter vector space. ScholarStream uses a multi-stage cleaning process (BeautifulSoup + Regex) to prune non-discursive elements, ensuring only the core thesis of a document is indexed.

2. Diversity-Weighted "Round-Robin" Ranking
To prevent "Winner-Takes-All" bias—where one keyword-dense document dominates the context—I implemented a custom diversity filter. The engine identifies the top-ranked chunks globally but prioritizes selecting the highest-scoring segment from *each* unique source first. This guarantees the LLM receives a collaborative, multi-source dataset.

3. Precision Sentence Extraction
Instead of passing raw, messy chunks to the LLM, the system employs a "Representative Sentence" extractor. It filters for factual density based on length windows and metadata filtering, providing the model with high-signal "facts" rather than fragmented headers.

4. Deterministic Token Budgeting
To handle strict API limits, the system uses a priority queue for token allocation:
Fixed Reserves: System instructions and user queries are protected.
Dynamic Slicing: Remaining tokens are divided equally among sources.
Hard Truncation: Uses `tiktoken` to programmatically ensure the payload never exceeds the 1600-token limit.


Prerequisites
Python 3.9+
OpenAI API Key

Installation
```bash
pip install openai tiktoken beautifulsoup4 scikit-learn cloudscraper
