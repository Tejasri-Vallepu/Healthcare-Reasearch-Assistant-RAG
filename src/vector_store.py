import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from config import PROCESSED_DIR, CHROMA_DB_DIR


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"

COLLECTION_NAME = "diabetes_research"


# ---------------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded successfully.\n")


# ---------------------------------------------------------
# CREATE CHROMADB CLIENT
# ---------------------------------------------------------

client = chromadb.PersistentClient(
    path=str(CHROMA_DB_DIR)
)


# ---------------------------------------------------------
# CREATE OR GET COLLECTION
# ---------------------------------------------------------

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# ---------------------------------------------------------
# LOAD CHUNKS
# ---------------------------------------------------------

def load_chunks(chunks_path: Path) -> list[dict]:
    """
    Load chunks from a JSON file.
    """

    with open(
        chunks_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ---------------------------------------------------------
# ADD DOCUMENT TO CHROMADB
# ---------------------------------------------------------

def add_document(chunks_path: Path):
    """
    Generate embeddings for all chunks in one document
    and store them in ChromaDB.
    """

    print(
        f"Processing: {chunks_path.name}"
    )

    chunks = load_chunks(chunks_path)

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        f"Generating embeddings for "
        f"{len(texts)} chunks..."
    )

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "page": chunk["page"]
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print(
        f"Stored {len(chunks)} chunks "
        f"in ChromaDB.\n"
    )


# ---------------------------------------------------------
# ADD ALL DOCUMENTS
# ---------------------------------------------------------

def build_vector_database():
    """
    Process every chunk JSON file and store
    everything in ChromaDB.
    """

    chunks_dir = (
        PROCESSED_DIR / "chunks"
    )

    chunk_files = sorted(
        chunks_dir.glob("*.json")
    )

    if not chunk_files:

        print("No chunk files found.")

        return

    print(
        f"Found {len(chunk_files)} "
        f"chunk files.\n"
    )

    for chunk_file in chunk_files:

        add_document(chunk_file)

    print("=" * 60)

    print(
        "Vector database creation completed."
    )

    print(
        f"Total stored chunks: "
        f"{collection.count()}"
    )

    print("=" * 60)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    build_vector_database()