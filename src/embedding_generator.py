import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from config import PROCESSED_DIR


# ---------------------------------------------------------
# EMBEDDING MODEL
# ---------------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------------

model = SentenceTransformer(MODEL_NAME)


# ---------------------------------------------------------
# LOAD CHUNKS
# ---------------------------------------------------------

def load_chunks(chunks_path: Path) -> list[dict]:
    """
    Load chunk data from a JSON file.
    """

    with open(
        chunks_path,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    return chunks


# ---------------------------------------------------------
# GENERATE EMBEDDINGS
# ---------------------------------------------------------

def generate_embeddings(chunks: list[dict]):
    """
    Generate an embedding vector for every chunk.
    """

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    return embeddings


# ---------------------------------------------------------
# TEST EMBEDDINGS
# ---------------------------------------------------------

def test_embeddings():

    chunks_dir = (
        PROCESSED_DIR / "chunks"
    )

    chunk_files = sorted(
        chunks_dir.glob("*.json")
    )

    if not chunk_files:

        print("No chunk files found.")

        return

    # Test using the first document.
    first_file = chunk_files[0]

    print(
        f"Loading chunks from: "
        f"{first_file.name}"
    )

    chunks = load_chunks(
        first_file
    )

    print(
        f"Number of chunks: "
        f"{len(chunks)}"
    )

    embeddings = generate_embeddings(
        chunks
    )

    print("\nEmbedding generation successful!")

    print(
        f"Number of embeddings: "
        f"{len(embeddings)}"
    )

    print(
        f"Embedding dimension: "
        f"{len(embeddings[0])}"
    )

    print(
        "\nFirst embedding:"
    )

    print(
        embeddings[0][:10]
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    test_embeddings()