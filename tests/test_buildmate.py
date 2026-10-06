"""
test_buildmate.py - Automated Unit Test Suite for BuildMate AI
Evaluates text extraction, chunking, local vector search, hallucination refusals, and calculator arithmetic.
"""

import pytest
import numpy as np
from calculator import calculate_concrete_volume
from vector_store import LocalVectorStore
from rag_engine import chunk_text

def test_quantity_calculator_volume():
    """Validates deterministic volume arithmetic: 4 columns of 0.3 x 0.3 x 3.0 m = 1.080 m3"""
    vol = calculate_concrete_volume(4, 0.30, 0.30, 3.00)
    assert vol == 1.080

def test_calculator_invalid_inputs():
    """Ensures negative or zero dimensions raise ValueError."""
    with pytest.raises(ValueError):
        calculate_concrete_volume(0, 0.3, 0.3, 3.0)

def test_semantic_chunking_overlap():
    """Validates sliding window text chunking and overlap retention."""
    mock_pages = [{"text": "A" * 600, "source": "test.pdf", "page": 1}]
    chunks, meta = chunk_text(mock_pages, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 2
    assert meta[0]["source"] == "test.pdf"
    assert meta[0]["page"] == 1

def test_vector_cosine_similarity():
    """Validates local NumPy vector retrieval with exact identity match."""
    store = LocalVectorStore()
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    store.add_chunks(["Column M30 concrete", "Waterproofing membrane"], [vec1, vec2], [{"source": "spec.pdf", "page": 2}, {"source": "spec.pdf", "page": 5}])
    
    # Query matching vec1 exactly
    results = store.search([1.0, 0.0, 0.0], top_k=1, threshold=0.5)
    assert len(results) == 1
    assert "M30" in results[0]["chunk"]
    assert results[0]["metadata"]["page"] == 2
    assert results[0]["score"] > 0.99

def test_vector_store_threshold_filtering():
    """Ensures orthogonal or unrelated vectors below threshold are excluded."""
    store = LocalVectorStore()
    vec = [1.0, 0.0, 0.0]
    store.add_chunks(["Irrelevant text"], [vec], [{"source": "spec.pdf", "page": 1}])
    
    # Orthogonal query
    results = store.search([0.0, 1.0, 0.0], top_k=1, threshold=0.5)
    assert len(results) == 0

def test_missing_info_refusal_handler():
    """Verifies that empty retrieval triggers the explicit refusal message."""
    from rag_engine import query_gemini_assistant
    refusal_msg = query_gemini_assistant("What is duct thickness?", [], api_key="dummy_key")
    assert "not found in the uploaded construction document" in refusal_msg
