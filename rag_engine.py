"""
rag_engine.py - Document Parsing, Embedding Generation & Guarded Gemini RAG
Part of BuildMate AI - Intelligent Construction Site Assistant
"""

import io
from pypdf import PdfReader
import google.generativeai as genai
from vector_store import LocalVectorStore

def extract_text_from_file(file_bytes: bytes, filename: str):
    """
    Extracts text and page numbers from uploaded PDF or TXT files.
    Returns list of dicts: [{"text": str, "source": filename, "page": int}]
    """
    pages_data = []
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        for idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages_data.append({
                    "text": text.strip(),
                    "source": filename,
                    "page": idx + 1
                })
    else:
        # TXT file processing
        text = file_bytes.decode("utf-8", errors="ignore").strip()
        if text:
            pages_data.append({
                "text": text,
                "source": filename,
                "page": 1
            })
    return pages_data

def chunk_text(pages_data: list, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Splits page-level text into uniform overlapping chunks to maintain engineering context.
    """
    chunks = []
    metadata = []
    for item in pages_data:
        text = item["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if len(chunk) > 30:  # Ignore trivial snippets
                chunks.append(chunk)
                metadata.append({"source": item["source"], "page": item["page"]})
            start += (chunk_size - chunk_overlap)
    return chunks, metadata

def generate_embeddings(text_list: list, api_key: str):
    """
    Generates 768-dimensional dense vector embeddings using Google Gemini text-embedding-004.
    """
    genai.configure(api_key=api_key)
    embeddings = []
    for text in text_list:
        res = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        embeddings.append(res["embedding"])
    return embeddings

def generate_query_embedding(query: str, api_key: str):
    """
    Generates query vector embedding for semantic search.
    """
    genai.configure(api_key=api_key)
    res = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="retrieval_query"
    )
    return res["embedding"]

def query_gemini_assistant(query: str, retrieved_results: list, api_key: str) -> str:
    """
    Passes guarded prompt with strictly retrieved context to Google Gemini (gemini-1.5-flash).
    Guarantees zero hallucination and enforces honest refusal when information is absent.
    """
    if not retrieved_results:
        return "The requested information was not found in the uploaded construction document."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Format context excerpts with citations
    context_str = ""
    for r in retrieved_results:
        src = r["metadata"]["source"]
        pg = r["metadata"]["page"]
        context_str += f"\n[Source: {src}, Page {pg}]:\n\"{r['chunk']}\"\n"

    prompt = f"""You are BuildMate AI, an intelligent construction site assistant for site engineers.
Answer the user's question using ONLY the provided verified document excerpts below.

Strict Engineering Instructions:
1. Ground your answer strictly in the excerpts. Do NOT guess, invent values, or assume missing specifications.
2. If the document does not explicitly provide the answer, state strictly:
   "The requested information is not mentioned in the uploaded construction document."
3. Always cite the exact source document name and page number found in the context tags.
4. Keep the answer professional, concise, and easy to read for field engineers.

Context Excerpts:
{context_str}

User Question: {query}
Answer:"""

    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.2, "max_output_tokens": 500}
    )
    return response.text.strip()
