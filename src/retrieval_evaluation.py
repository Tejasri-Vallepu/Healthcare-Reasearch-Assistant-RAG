from reranker import (
    rerank_documents,
    TOP_K_RETRIEVAL,
    TOP_K_RERANKED
)

from retriever import retrieve_documents


# ---------------------------------------------------------
# TEST QUESTIONS
# ---------------------------------------------------------

TEST_QUESTIONS = [

    "What are the diagnostic criteria for diabetes?",

    "What is A1c used for in diabetes diagnosis?",

    "What is the difference between type 1 and type 2 diabetes?",

    "What are the risk factors for type 2 diabetes?",

    "What is prediabetes?"

]


# ---------------------------------------------------------
# EVALUATE ONE QUESTION
# ---------------------------------------------------------

def evaluate_question(query: str):

    print("\n")
    print("=" * 70)

    print(
        f"QUESTION:\n{query}"
    )

    print("=" * 70)


    # -----------------------------------------------------
    # STEP 1: Retrieve candidates
    # -----------------------------------------------------

    retrieved_documents = retrieve_documents(
        query=query,
        top_k=TOP_K_RETRIEVAL
    )


    print(
        f"\nRetrieved candidates: "
        f"{len(retrieved_documents)}"
    )


    # -----------------------------------------------------
    # STEP 2: Rerank candidates
    # -----------------------------------------------------

    reranked_documents = rerank_documents(
        query=query,
        documents=retrieved_documents,
        top_k=TOP_K_RERANKED
    )


    # -----------------------------------------------------
    # STEP 3: Display results
    # -----------------------------------------------------

    print(
        "\nTop reranked evidence:"
    )

    for rank, result in enumerate(
        reranked_documents,
        start=1
    ):

        print(
            f"\nRank {rank}"
        )

        print(
            "-" * 50
        )

        print(
            f"Chunk: {result['chunk_id']}"
        )

        print(
            f"Source: {result['source']}"
        )

        print(
            f"Page: {result['page']}"
        )

        print(
            f"Rerank score: "
            f"{result['rerank_score']:.4f}"
        )

        # Show only the first 400 characters
        # so the evaluation output stays readable.

        text_preview = (
            result["text"]
            .replace("\n", " ")
        )

        print(
            f"Evidence: "
            f"{text_preview[:400]}..."
        )


# ---------------------------------------------------------
# RUN ALL TESTS
# ---------------------------------------------------------

def run_evaluation():

    print(
        "\nStarting retrieval evaluation..."
    )

    print(
        f"Number of test questions: "
        f"{len(TEST_QUESTIONS)}"
    )


    for question in TEST_QUESTIONS:

        evaluate_question(
            question
        )


    print("\n")
    print("=" * 70)

    print(
        "Retrieval evaluation completed."
    )

    print("=" * 70)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    run_evaluation()