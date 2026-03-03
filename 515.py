# importing libraries
import os
import re
import tiktoken
from typing import List, Dict
from pathlib import Path
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# Configuration Constants
MAX_CHUNK_WORDS = 500
TOP_K_CHUNKS = 3
MAX_CONTEXT_TOKENS = 1600
TOKEN_SAFETY_MARGIN = 150
MODEL_NAME = "gpt-4o-mini"

# Initialize Global Client
# We don't check for the key here to avoid crashing on import; 
# the check is handled in main().
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Sources: locally saved HTML files and urls
SOURCES: Dict[str, Dict[str, str]] = {
    "1": {"file": "source1.html", "url": "https://www.ahajournals.org/doi/10.1161/HYP.0000000000000238"},
    "2": {"file": "source2.html", "url": "https://www.ahajournals.org/doi/10.1161/CIRCRESAHA.121.318083"},
    "3": {"file": "source3.html", "url": "https://www.ahajournals.org/doi/10.1161/CIR.0000000000001341"},
}

def clean_html(raw_html: str) -> str:
    """Cleans the html by removing scripts, styles, tables, tags and others using BeautifulSoup."""
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "table", "footer", "nav", "header", "aside"]):
        tag.decompose()
    
    # Prune reference/bibliography sections to avoid citation noise
    for ref_section in soup.find_all(attrs={"class": re.compile(r'ref|bib|cite', re.I)}):
        ref_section.decompose()

    text = soup.get_text(separator=" ")
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text: str, chunk_size: int = MAX_CHUNK_WORDS) -> List[str]:
    """Text ingestion and chunking (default: 500 words)."""
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def build_chunks() -> List[Dict]:
    """Function to build text chunks with source data."""
    all_chunks = []
    base_path = Path(__file__).parent.absolute()
    
    for sid, info in SOURCES.items():
        file_path = Path(info["file"])
        if not file_path.exists():
            file_path = base_path / info["file"]

        if not file_path.exists():
            print(f"Skipping {info['file']}: File not found.")
            continue
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                clean_txt = clean_html(f.read())
                text_chunks = chunk_text(clean_txt)
                for c in text_chunks:
                    all_chunks.append({"text": c, "source_id": sid})
        except Exception as e:
            print(f"Error processing {info['file']}: {e}")
            
    return all_chunks

def get_ranked_content(chunks: List[Dict], query: str):
    """
    Ensures diversity by pulling the top chunk from EACH unique source.
    This prevents the system from only citing a single source.
    """
    if not chunks:
        return []

    # 1. Rank all chunks globally using TF-IDF
    chunk_texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(chunk_texts)
    query_vec = vectorizer.transform([query])
    
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    ranked_chunks = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

    # 2. Diversity Selection (Round-Robin)
    selected_chunks = []
    seen_sources = set()

    for score, chunk in ranked_chunks:
        sid = chunk["source_id"]
        if sid not in seen_sources:
            selected_chunks.append(chunk)
            seen_sources.add(sid)
        if len(selected_chunks) >= 3:
            break

    # 3. Extract one representative sentence from each selected source
    final_sentences = []
    for chunk in selected_chunks:
        sents = re.split(r'(?<![A-Z])(?<=[.!?])\s+', chunk["text"])
        rep_sent = next((s.strip() for s in sents if 12 < len(s.split()) < 60), sents[0])
        final_sentences.append({"text": rep_sent, "sid": chunk["source_id"]})
    
    return final_sentences

def count_tokens(text: str) -> int:
    """Token usage tracked using tiktoken."""
    return len(tiktoken.encoding_for_model(MODEL_NAME).encode(text))

def safe_api_call(messages: List[Dict]) -> str:
    """Handles OpenAI API interaction with error handling."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0 
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API Error: {str(e)}"

def generate_answer(relevant_sentences: List[Dict], query: str) -> str:
    """Context construction with strict 1600 token limits."""
    source_legend = "\n".join([f"[{sid}]: {info['url']}" for sid, info in SOURCES.items()])
    
    # Calculate token budget
    overhead_tokens = count_tokens(source_legend + query + "System prompt...") 
    available_tokens = MAX_CONTEXT_TOKENS - overhead_tokens - TOKEN_SAFETY_MARGIN

    context_str = ""
    for s in relevant_sentences:
        formatted_sent = f"{s['text']} [{s['sid']}]\n"
        if count_tokens(context_str + formatted_sent) > available_tokens:
            break
        context_str += formatted_sent

    if not context_str.strip():
        return "I'm sorry, I couldn't find enough information in the provided sources to give a relevant answer."

    messages = [
        {"role": "system", "content": (
            "You are a medical assistant. Use ONLY the provided sentences to create a structured report. "
            "Use Markdown headers. Cite every fact using [1], [2], or [3]. "
            "Do not use internal knowledge."
        )},
        {"role": "user", "content": f"CONTEXT SENTENCES:\n{context_str}\n\nSOURCE LEGEND:\n{source_legend}\n\nQUESTION: {query}"}
    ]

    return safe_api_call(messages)

def main():
    print("Initializing Mini MediSearch...")
    
    # Unified API validation check
    if not os.getenv("OPENAI_API_KEY"):
        print("CRITICAL ERROR: OPENAI_API_KEY environment variable not set.")
        return

    chunks = build_chunks()
    if not chunks:
        print("No source files found. Ensure source1.html, source2.html, and source3.html are present.")
        return
        
    query = input("\nEnter your medical query: ")
    relevant_content = get_ranked_content(chunks, query)
    
    print("\nGenerating Answer...\n" + "="*30)
    answer = generate_answer(relevant_content, query)
    print(answer)

if __name__ == "__main__":
    main()