from pathlib import Path


# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------
# Path(__file__) gives the location of this Python file.
# .resolve() converts it into an absolute path.
#
# config.py is inside:
# Healthcare_RAG/src/
#
# Therefore, .parent.parent takes us to:
# Healthcare_RAG/
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------
# DATA DIRECTORIES
# ---------------------------------------------------------

# Original PDF documents
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"

# Processed data
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Extracted text files
EXTRACTED_TEXT_DIR = PROCESSED_DIR / "extracted_text"


# ---------------------------------------------------------
# VECTOR DATABASE DIRECTORY
# ---------------------------------------------------------

# ChromaDB will eventually store our vector database here.
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"


# ---------------------------------------------------------
# CREATE REQUIRED DIRECTORIES
# ---------------------------------------------------------

# mkdir(..., exist_ok=True) creates the directory if it
# doesn't already exist.
#
# exist_ok=True prevents an error if the directory exists.
EXTRACTED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)