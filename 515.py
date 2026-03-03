
import os
import re
import tiktoken
from typing import List, Dict
from pathlib import Path
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

#Constants
MAX_CHUNK_WORDS = 500
TOP_K_RESULTS = 3
MAX_CONTEXT_TOKENS = 1600
TOKEN_SAFETY_MARGIN = 150
ENGINE_MODEL = "gpt-4o-mini"

#Initialize Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#Data Sources: Update these with your research files
DOC_SOURCES: Dict[str, Dict[str, str]] = {
    "1": {"file": "paper_1.html", "url": "https://archive.org/details/research_sample_1"},
    "2": {"file": "paper_2.html", "url": "https://archive.org/details/research_sample_2"},
    "3": {"file": "paper_3.html", "url": "https://archive.org/details/research_sample_3"},
}

def clean_document_content(raw_html: str) -> str:
    """Removes boilerplate, scripts, and navigation noise from HTML documents."""
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "table", "footer", "nav", "header", "aside"]):
        tag.decompose()
    
    #Prune reference/bibliography sections to focus on core discourse
    for noise in soup.find_all(attrs={"class": re.compile(r'ref|bib|cite|appendix', re.I)}):
        noise.decompose()

    text = soup.get_text(separator=" ")
    return re.sub(r'\s+', ' ', text).strip()

def segment_text(text: str, limit: int = MAX_CHUNK_WORDS) -> List[str]:
    """Segments text into manageable word-count chunks."""
    words = text.split()
    return [' '.join(words[i:i+limit]) for i in range(0, len(words), limit)]

def index_local_repository() -> List[Dict]:
    """Parses local HTML files into a structured index."""
    indexed_chunks = []
    base_dir = Path(__file__).parent.absolute()
    
    for doc_id, info in DOC_SOURCES.items():
        target_path = base_dir / info["file"]

        if not target_path.exists():
            continue
            
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                content = clean_document_content(f.read())
                segments = segment_text(content)
                for seg in segments:
                    indexed_chunks.append({"text": seg, "doc_id": doc_id})
        except Exception as e:
            print(f"Error indexing {info['file']}: {e}")
            
    return indexed_chunks

def get_diverse_relevant_context(chunks: List[Dict], user_query: str):
    """Ranks chunks and ensures multi-source representation."""
    if not chunks:
        return []

    #TF-IDF Global Ranking
    corpus = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vec = vectorizer.transform([user_query])
    
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

    #Diversity Logic (Select top chunk per unique source)
    selected = []
    collected_ids = set()

    for score, chunk in ranked:
        did = chunk["doc_id"]
        if did not in collected_ids:
            selected.append(chunk)
            collected_ids.add(did)
        if len(selected) >= TOP_K_RESULTS:
            break

    # 3. Representative Sentence Extraction
    refined_context = []
    for chunk in selected:
        sentences = re.split(r'(?<![A-Z])(?<=[.!?])\s+', chunk["text"])
        # Target factual sentences (avoiding short fragments or citations)
        best_match = next((s.strip() for s in sentences if 15 < len(s.split()) < 55), sentences[0])
        refined_context.append({"text": best_match, "id": chunk["doc_id"]})
    
    return refined_context

def calculate_tokens(text: str) -> int:
    """Returns token count using the project's encoding model."""
    return len(tiktoken.encoding_for_model(ENGINE_MODEL).encode(text))

def dispatch_inference(messages: List[Dict]) -> str:
    """Communicates with the LLM via the OpenAI interface."""
    try:
        response = client.chat.completions.create(
            model=ENGINE_MODEL,
            messages=messages,
            temperature=0.0 
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Inference Error: {str(e)}"

def construct_and_run_query(context_segments: List[Dict], query: str) -> str:
    """Builds a grounded prompt with strict token constraints."""
    citation_legend = "\n".join([f"[{did}]: {info['url']}" for did, info in DOC_SOURCES.items()])
    
    #Token Budgeting
    base_overhead = calculate_tokens(citation_legend + query + "System Instructions") 
    usable_budget = MAX_CONTEXT_TOKENS - base_overhead - TOKEN_SAFETY_MARGIN

    context_payload = ""
    for seg in context_segments:
        entry = f"{seg['text']} [{seg['id']}]\n"
        if calculate_tokens(context_payload + entry) > usable_budget:
            break
        context_payload += entry

    if not context_payload.strip():
        return "Insufficient data found in local repository to answer query."

    payload = [
        {"role": "system", "content": (
            "You are a Research Intelligence Assistant. Use the provided context to generate a technical report. "
            "Structure your response with Markdown. Every claim must be cited using [1], [2], or [3]. "
            "Strictly avoid using information outside the provided context."
        )},
        {"role": "user", "content": f"CONTEXT DATA:\n{context_payload}\n\nCITATIONS:\n{citation_legend}\n\nRESEARCH QUESTION: {query}"}
    ]

    return dispatch_inference(payload)

def main():
    print("--- ScholarStream RAG Engine Initialized ---")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment.")
        return

    repository = index_local_repository()
    if not repository:
        print("Local repository is empty. Please check your source files.")
        return
        
    user_input = input("\nEnter your research query: ")
    context = get_diverse_relevant_context(repository, user_input)
    
    print("\nSynthesizing Answer...\n" + "-"*40)
    result = construct_and_run_query(context, user_input)
    print(result)

if __name__ == "__main__":
    main()
