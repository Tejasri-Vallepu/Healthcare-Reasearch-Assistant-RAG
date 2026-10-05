from sentence_transformers import CrossEncoder

from src.retriever import retrieve_documents

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_RETRIEVAL = 10

TOP_K_RERANKED = 5


# ---------------------------------------------------------
# LOAD RERANKER
# ---------------------------------------------------------

print("Loading reranker model...")

reranker = CrossEncoder(
    RERANKER_MODEL
)

print(
    "Reranker model loaded successfully.\n"
)


# ---------------------------------------------------------
# RERANK DOCUMENTS
# ---------------------------------------------------------

def rerank_documents(
    query: str,
    documents: list[dict],
    top_k: int = TOP_K_RERANKED
) -> list[dict]:
    """
    Rerank retrieved documents using a CrossEncoder.

    Parameters
    ----------
    query : str
        User's research question.

    documents : list[dict]
        Documents retrieved from ChromaDB.

    top_k : int
        Number of final documents to return.

    Returns
    -------
    list[dict]
        Reranked documents.
    """

    # -----------------------------------------------------
    # Create question-document pairs.
    # -----------------------------------------------------

    pairs = [
        [query, document["text"]]
        for document in documents
    ]


    # -----------------------------------------------------
    # Calculate relevance scores.
    # -----------------------------------------------------

    scores = reranker.predict(
        pairs
    )


    # -----------------------------------------------------
    # Attach reranking scores.
    # -----------------------------------------------------

    reranked_documents = []

    for document, score in zip(
        documents,
        scores
    ):

        document_copy = document.copy()

        document_copy["rerank_score"] = float(
            score
        )

        reranked_documents.append(
            document_copy
        )


    # -----------------------------------------------------
    # Sort by highest relevance score.
    # -----------------------------------------------------

    reranked_documents.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )


    # -----------------------------------------------------
    # Return top results.
    # -----------------------------------------------------

    return reranked_documents[:top_k]


# ---------------------------------------------------------
# TEST RERANKING
# ---------------------------------------------------------

def test_reranking():

    query = (
        "What are the diagnostic criteria "
        "for diabetes?"
    )


    print("=" * 60)

    print(
        f"Query:\n{query}"
    )

    print("=" * 60)


    # -----------------------------------------------------
    # STEP 1: Retrieve candidate documents.
    # -----------------------------------------------------

    print(
        "\nRetrieving candidate documents..."
    )

    retrieved_documents = retrieve_documents(
        query=query,
        top_k=TOP_K_RETRIEVAL
    )


    print(
        f"Retrieved "
        f"{len(retrieved_documents)} candidates."
    )


    # -----------------------------------------------------
    # STEP 2: Rerank candidates.
    # -----------------------------------------------------

    print(
        "\nReranking candidates..."
    )

    reranked_documents = rerank_documents(
        query=query,
        documents=retrieved_documents,
        top_k=TOP_K_RERANKED
    )


    # -----------------------------------------------------
    # STEP 3: Display final results.
    # -----------------------------------------------------

    print(
        "\nFINAL RERANKED RESULTS"
    )

    print(
        "=" * 60
    )


    for index, result in enumerate(
        reranked_documents,
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
            f"Original distance: "
            f"{result['distance']:.4f}"
        )

        print(
            f"Rerank score: "
            f"{result['rerank_score']:.4f}"
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
    test_reranking()