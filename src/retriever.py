import chromadb
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_DB_DIR


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"

COLLECTION_NAME = "diabetes_research"

TOP_K = 10


# ---------------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded successfully.\n")


# ---------------------------------------------------------
# CONNECT TO CHROMADB
# ---------------------------------------------------------

client = chromadb.PersistentClient(
    path=str(CHROMA_DB_DIR)
)


# ---------------------------------------------------------
# LOAD EXISTING COLLECTION
# ---------------------------------------------------------

collection = client.get_collection(
    name=COLLECTION_NAME
)


# ---------------------------------------------------------
# RETRIEVE RELEVANT CHUNKS
# ---------------------------------------------------------

def retrieve_documents(
    query: str,
    top_k: int = TOP_K
) -> list[dict]:
    """
    Retrieve the most relevant chunks from ChromaDB.

    Parameters
    ----------
    query : str
        User's research question.

    top_k : int
        Number of relevant chunks to retrieve.

    Returns
    -------
    list[dict]
        Retrieved chunks with metadata and distance.
    """

    # -----------------------------------------------------
    # Convert the user's question into an embedding.
    # -----------------------------------------------------

    query_embedding = model.encode(
        query
    ).tolist()


    # -----------------------------------------------------
    # Search ChromaDB for similar vectors.
    # -----------------------------------------------------

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )


    # -----------------------------------------------------
    # Format results into an easier structure.
    # -----------------------------------------------------

    retrieved_documents = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]


    for document, metadata, distance, chunk_id in zip(
        documents,
        metadatas,
        distances,
        ids
    ):

        retrieved_documents.append(
            {
                "chunk_id": chunk_id,
                "text": document,
                "source": metadata["source"],
                "page": metadata["page"],
                "distance": distance
            }
        )


    return retrieved_documents


# ---------------------------------------------------------
# TEST RETRIEVAL
# ---------------------------------------------------------

def test_retrieval():

    query = (
        "What are the diagnostic criteria "
        "for diabetes?"
    )

    print("=" * 60)

    print(
        f"Query:\n{query}"
    )

    print("=" * 60)


    results = retrieve_documents(
        query=query,
        top_k=TOP_K
    )


    for index, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRESULT {index}"
        )

        print(
            "-" * 60
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        print(
            f"Source: "
            f"{result['source']}"
        )

        print(
            f"Page: "
            f"{result['page']}"
        )

        print(
            f"Distance: "
            f"{result['distance']:.4f}"
        )

        print(
            "\nText:"
        )

        print(
            result["text"][:1000]
        )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    test_retrieval()