# BuildMate AI

An intelligent construction site assistant combining document-grounded Retrieval-Augmented Generation (RAG) with Google Gemini and an on-site concrete quantity calculator.

Upload construction documents (PDF or TXT) to query structural specifications, concrete mix grades, curing periods, and safety codes with verifiable page-level citations. The app also includes a deterministic quantity calculator for columns, beams, slabs, and footings.

The deterministic calculations and local vector store do not rely on language models for math. The RAG assistant answers strictly from retrieved context and refuses to hallucinate when clauses are absent.

## Requirements

* Python 3.10 or newer
* A modern web browser
* Google Gemini API Key (from [Google AI Studio](https://aistudio.google.com/))

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/username/BuildMate-AI.git
   cd BuildMate-AI
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Launch the Streamlit app:
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Project Structure

```text
BuildMate-AI/
├── data/
│   └── sample_structural_specs.txt   # Sample construction specification document
├── tests/
│   └── test_buildmate.py             # 6 automated unit tests (pytest)
├── .gitignore                        # Git exclusion rules
├── README.md                         # Project documentation and setup guide
├── app.py                            # Streamlit web application interface
├── calculator.py                     # Deterministic concrete volume calculator
├── rag_engine.py                     # Document parsing, chunking, and Gemini RAG
├── requirements.txt                  # Python dependencies
└── vector_store.py                   # In-memory NumPy vector store with cosine similarity
```

## Running Unit Tests

Run the full automated test suite:
```bash
pytest tests/ -v
```

All 6 test cases validate PDF text parsing, semantic chunking overlap, local vector cosine similarity, hallucination refusal, and volumetric arithmetic.

## Academic Assessment

* **Project**: TAE-2 Mini Project Report
* **Subject**: Large Language Models (LLM)
* **Branch**: CSE (AIML) – Academic Year 2026–27
