from src.retriever import retrieve_documents
from src.reranker import rerank_documents
from src.generator import generate_answer


# ---------------------------------------------------------
# RAG SETTINGS
# ---------------------------------------------------------

TOP_K_RETRIEVAL = 10
TOP_K_RERANKED = 5


# ---------------------------------------------------------
# RUN RAG PIPELINE
# ---------------------------------------------------------

def answer_question(question: str):

    retrieved_documents = retrieve_documents(
        query=question,
        top_k=TOP_K_RETRIEVAL
    )

    print(
        f"\nRetrieved "
        f"{len(retrieved_documents)} candidates."
    )

    reranked_documents = rerank_documents(
        query=question,
        documents=retrieved_documents,
        top_k=TOP_K_RERANKED
    )

    print(
        f"Selected top "
        f"{len(reranked_documents)} evidence chunks."
    )

    answer = generate_answer(
        question=question,
        evidence=reranked_documents
    )

    return {
        "answer": answer,
        "evidence": reranked_documents
    }

    # -----------------------------------------------------
    # STEP 1: RETRIEVE CANDIDATE DOCUMENTS
    # -----------------------------------------------------

    retrieved_documents = retrieve_documents(
        query=question,
        top_k=TOP_K_RETRIEVAL
    )

    print(f"\nRetrieved {len(retrieved_documents)} candidates.")


    # -----------------------------------------------------
    # STEP 2: RERANK DOCUMENTS
    # -----------------------------------------------------

    reranked_documents = rerank_documents(
        query=question,
        documents=retrieved_documents,
        top_k=TOP_K_RERANKED
    )

    print(f"Selected top {len(reranked_documents)} evidence chunks.")


    # -----------------------------------------------------
    # STEP 3: GENERATE GROUNDED ANSWER
    # -----------------------------------------------------

    answer = generate_answer(
        question=question,
        evidence=reranked_documents
    )

    return answer


# ---------------------------------------------------------
# TEST THE COMPLETE PIPELINE
# ---------------------------------------------------------

if __name__ == "__main__":

    question = "What are the risk factors for type 2 diabetes?"

    print("\n")
    print("=" * 70)
    print("HEALTHCARE RAG ASSISTANT")
    print("=" * 70)

    print(f"\nQuestion: {question}")

    answer = answer_question(question)

    print("\nGenerated Answer:")
    print("=" * 70)
    print(answer)

    print("\n")
    print("=" * 70)
    print("RAG PIPELINE COMPLETED")
    print("=" * 70)