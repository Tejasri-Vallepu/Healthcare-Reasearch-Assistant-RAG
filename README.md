# 🩺 Healthcare Research Assistant using RAG

A healthcare research assistant that uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant evidence from trusted diabetes research literature and generate grounded answers with source and page citations.

The system is designed for **research and educational purposes** and is not intended for medical diagnosis or treatment decisions.

---

## 📌 Project Overview

Healthcare and medical literature contains a large amount of detailed information distributed across research documents and reports. Finding relevant evidence manually can be time-consuming.

This project builds a RAG-based research assistant that allows users to ask natural-language questions about diabetes research.

The system:

1. Extracts text from medical PDF documents.
2. Cleans the extracted text.
3. Splits the documents into smaller chunks.
4. Converts chunks into vector embeddings.
5. Stores embeddings in ChromaDB.
6. Retrieves relevant evidence for a user query.
7. Reranks retrieved evidence using a Cross-Encoder.
8. Generates a grounded answer using Gemini.
9. Displays the answer along with source and page information.
10. Provides the retrieved evidence through an interactive Streamlit interface.

---

## 🎯 Project Objectives

* Build a domain-specific healthcare RAG system.
* Retrieve evidence from trusted medical literature.
* Improve retrieval quality using reranking.
* Generate answers grounded only in retrieved evidence.
* Provide source and page-level citations.
* Reduce unsupported or hallucinated responses.
* Build an interactive research assistant using Streamlit.

---

## 🏗️ System Architecture

```text
Medical PDF Documents
        │
        ▼
   PDF Extraction
        │
        ▼
    Text Cleaning
        │
        ▼
      Chunking
        │
        ▼
    Embeddings
        │
        ▼
     ChromaDB
        │
        │
     User Query
        │
        ▼
   Query Embedding
        │
        ▼
   Vector Retrieval
        │
        ▼
   Top-K Candidates
        │
        ▼
 Cross-Encoder Reranker
        │
        ▼
   Top Evidence
        │
        ▼
      Gemini
        │
        ▼
 Grounded Answer
        │
        ▼
 Source + Page Citations
        │
        ▼
    Streamlit UI
```

---

## 📚 Knowledge Source

The initial knowledge base uses selected chapters from:

**NIDDK — Diabetes in America**

The project currently contains five PDF chapters:

* `DIA_Ch01.pdf`
* `DIA_Ch03.pdf`
* `DIA_Ch10.pdf`
* `DIA_Ch13.pdf`
* `DIA_Ch38.pdf`

These documents are processed and converted into searchable evidence chunks.

---

## 🛠️ Technologies Used

| Technology               | Purpose                         |
| ------------------------ | ------------------------------- |
| Python                   | Core programming language       |
| PyMuPDF                  | PDF text extraction             |
| Sentence Transformers    | Text embeddings                 |
| `all-MiniLM-L6-v2`       | Embedding model                 |
| ChromaDB                 | Vector database                 |
| Cross-Encoder            | Evidence reranking              |
| `ms-marco-MiniLM-L-6-v2` | Reranking model                 |
| Gemini                   | Answer generation               |
| Streamlit                | Web application                 |
| python-dotenv            | Environment variable management |

---

## 📂 Project Structure

```text
Healthcare_RAG/
│
├── .venv/
│
├── data/
│   ├── documents/
│   │   ├── DIA_Ch01.pdf
│   │   ├── DIA_Ch03.pdf
│   │   ├── DIA_Ch10.pdf
│   │   ├── DIA_Ch13.pdf
│   │   └── DIA_Ch38.pdf
│   │
│   └── processed/
│       ├── extracted_text/
│       ├── cleaned_text/
│       └── chunks/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── pdf_extractor.py
│   ├── text_cleaner.py
│   ├── chunker.py
│   ├── embedding_generator.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── retrieval_evaluation.py
│   ├── generator.py
│   └── rag_pipeline.py
│
├── notebooks/
│
├── chroma_db/
│
├── app/
│   └── app.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔄 Data Processing Pipeline

### 1. PDF Extraction

Medical PDFs are processed using PyMuPDF.

The extracted text preserves page boundaries so that source-level page citations can later be generated.

Example:

```text
--- PAGE 1 ---

Extracted text...

--- PAGE 2 ---

Extracted text...
```

---

### 2. Text Cleaning

Raw PDF extraction can contain:

* unnecessary line breaks
* broken words
* formatting artifacts
* repeated whitespace

The cleaning stage normalizes the text while preserving important medical information and page markers.

---

### 3. Chunking

Large documents are divided into smaller chunks.

Current configuration:

```text
Chunk size: 400
Chunk overlap: 80
```

Chunking allows the retrieval system to identify smaller sections of documents that are relevant to a user's question.

Each chunk also stores metadata such as:

* source document
* page number
* chunk ID

---

## 🧠 Embedding Generation

Each text chunk is converted into a numerical vector using:

```text
all-MiniLM-L6-v2
```

The model generates **384-dimensional embeddings**.

These embeddings represent the semantic meaning of the text and allow similar questions and evidence chunks to be matched.

---

## 🗄️ Vector Database

The project uses **ChromaDB** for persistent vector storage.

Collection:

```text
diabetes_research
```

Current knowledge base:

```text
Total chunks: 412
```

Distribution:

| Document  |  Chunks |
| --------- | ------: |
| DIA_Ch01  |     101 |
| DIA_Ch03  |      71 |
| DIA_Ch10  |      87 |
| DIA_Ch13  |      96 |
| DIA_Ch38  |      57 |
| **Total** | **412** |

---

## 🔎 Retrieval

When a user submits a question:

1. The question is converted into an embedding.
2. ChromaDB compares the query embedding with stored chunk embeddings.
3. The most semantically similar chunks are retrieved.
4. These chunks are passed to the reranking stage.

The retrieval candidate pool is currently configured separately from the final evidence selection so that the reranker has multiple candidates to compare.

---

## 🎯 Reranking

Initial vector retrieval provides semantically similar candidates, but semantic similarity alone does not always guarantee that the most useful evidence appears first.

A Cross-Encoder is therefore used:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The Cross-Encoder receives:

```text
[User Question, Retrieved Chunk]
```

and produces a relevance score.

The candidates are then sorted by this score and the highest-ranked evidence is passed to the generation stage.

---

## 🤖 Answer Generation

Gemini is used as the language model.

The generator receives:

```text
User Question
+
Reranked Evidence
```

The prompt instructs the model to:

* use only the provided evidence
* avoid unsupported claims
* avoid diagnosis
* avoid treatment recommendations
* clearly state when evidence is insufficient
* provide source and page citations
* list the sources actually used

Example citation format:

```text
[Source: DIA_Ch13, Page: 2]
```

---

## 🔗 End-to-End RAG Pipeline

The main pipeline is implemented in:

```text
src/rag_pipeline.py
```

The workflow is:

```text
Question
   ↓
Retrieve Documents
   ↓
Retrieve Candidate Pool
   ↓
Cross-Encoder Reranking
   ↓
Select Top Evidence
   ↓
Gemini Generation
   ↓
Grounded Answer
```

---

## 🖥️ Streamlit Application

The application is implemented in:

```text
app/app.py
```

The interface provides:

* research question input
* chat-style interaction
* generated answers
* retrieved evidence
* source document
* page number
* healthcare research disclaimer
* RAG pipeline overview

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd Healthcare_RAG
```

---

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_api_key_here
```

Do not commit the `.env` file to GitHub.

---

### 5. Build the processed data

Run the project modules in the required order:

```bash
python src/pdf_extractor.py
python src/text_cleaner.py
python src/chunker.py
python src/embedding_generator.py
python src/vector_store.py
```

---

### 6. Run the RAG pipeline

```bash
python src/rag_pipeline.py
```

---

### 7. Run the Streamlit application

```bash
python -m streamlit run app/app.py
```

The application will open in the browser.

---

## 🧪 Example Questions

The system can be tested with questions such as:

```text
What are the diagnostic criteria for diabetes?

What is A1c used for in diabetes diagnosis?

What is the difference between type 1 and type 2 diabetes?

What are the risk factors for type 2 diabetes?

What is prediabetes?
```

---

## 📊 Retrieval Evaluation

A retrieval evaluation module is included:

```text
src/retrieval_evaluation.py
```

It evaluates multiple test questions and examines the relevance of the retrieved and reranked evidence.

The evaluation process helps identify retrieval weaknesses before making improvements to the RAG pipeline.

---

## 🔐 Environment Variables

The project uses environment variables for API credentials.

Required variable:

```text
GEMINI_API_KEY
```

The `.env` file should never be committed to GitHub.

Recommended `.gitignore` entries:

```text
.env
.venv/
__pycache__/
*.pyc
chroma_db/
```

---

## ⚠️ Healthcare Safety

This application is designed for:

* research
* educational purposes
* medical literature exploration
* evidence retrieval

It is **not designed for**:

* medical diagnosis
* personalized treatment decisions
* emergency medical guidance
* replacing healthcare professionals

The system is explicitly instructed to avoid unsupported medical claims and to state when the retrieved evidence is insufficient.

---

## 🚀 Future Improvements

Potential improvements include:

* Hybrid retrieval using keyword + semantic search
* Better metadata filtering
* Improved chunking strategies
* Retrieval evaluation with labeled ground truth
* More comprehensive medical literature
* Query expansion
* Better citation validation
* Conversation-aware retrieval
* Source confidence indicators
* Retrieval caching
* Production deployment
* Authentication and access control
* Monitoring and evaluation dashboards

---

## 💡 Key Learning Outcomes

This project demonstrates practical experience with:

* RAG architecture
* document processing
* semantic search
* vector databases
* embeddings
* Cross-Encoder reranking
* LLM-based generation
* prompt engineering
* source-grounded generation
* citation handling
* Streamlit application development
* modular Python project architecture
* healthcare-domain AI considerations

---

## 📌 Project Highlights

**Domain:** Healthcare / Medical Research

**Architecture:** Retrieval-Augmented Generation (RAG)

**Vector Database:** ChromaDB

**Embedding Model:** `all-MiniLM-L6-v2`

**Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

**LLM:** Gemini

**Interface:** Streamlit

**Knowledge Base:** NIDDK Diabetes in America

**Current Knowledge Base:** 412 processed chunks
