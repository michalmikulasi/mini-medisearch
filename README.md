# mini-medisearch
This is a smaller version of a functional platform medisearch which uses GPT to retrieve medical data for users

his project implements a retrieval-augmented generation (RAG) system inspired by MediSearch. It transforms raw medical literature into a structured, grounded assistant that provides cited answers while strictly adhering to complex context.

The pipeline has been refined to move beyond simple keyword matching, ensuring more medical accuracy and source diversity:

1. More advanced HTML Scrubbing
The Hurdle: Medical journals contain significant "noise" (bibliographies, headers, and navigation) that distracts TF-IDF vectorizers. So I implemented more aggressive cleaning using BeautifulSoup and regex to prune tags, ensuring the system only indexes the actual scientific discourse.

2. Diversity-Aware "Round-Robin" Ranking
The Hurdle: Standard TF-IDF ranking often suffers from "Winner-Takes-All" bias(this is something i encountered specifically while doing this simple project), where one long or keyword-dense article drowns out the others, leading to single-source answers.

So, I developed a custom ranking logic that computes global TF-IDF/Cosine Similarity scores but selects the top-ranked chunk from each unique source first. This guarantees a multi-perspective, collaborative answer across all provided literature.

3. Precision Context Construction
The Hurdle: Randomly pulling sentences from chunks often retrieves headers or citations rather than factual data.

The Solution: Implemented a "Representative Sentence" extractor that filters for sentences within a specific length window (12–60 words) and ignores digit-heavy metadata, ensuring the LLM receives higher-quality "facts" to build its answer.

Token Budget Management
The system enforces a strict 1600-token hard limit with a deterministic priority queue:

Fixed Budget: System instructions, user query, and the source legend are tokenized and reserved first.

Dynamic Allocation: The remaining token "pie" is divided equally among the top-ranked chunks.

Truncation Logic: If a representative sentence exceeds its allocated "slice," it is programmatically truncated using tiktoken to ensure the final payload wont exceed the limit.

Engineering Hurdles & Solutions
Moved from redundant validation more robust validation. The system verifies environment variables at launch, preventing wasted processing if the API client is uninitialized.

Cross-Platform File Handling: Overcame file-path errors common in different OS environments (Windows vs. Linux) by migrating from string-based paths to pathlib.Path objects. And I do realize that most users would be windows users...

Hallucination Control: By setting temperature=0.0 and using a "Zero-Knowledge" system prompt, the model is strictly forbidden from using internal training data, ensuring every claim is grounded in the provided token context.

Requirements & Execution
Python 3.9+

Libraries: openai, tiktoken, beautifulsoup4, scikit-learn, cloudscraper

Setup:

Bash
export OPENAI_API_KEY="your_api_key_here"
Run:

Bash
python main.py

There are more things to be improved. I would be happy to elaborate on my solution further(personally or online).
