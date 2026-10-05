import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.rag_pipeline import answer_question


st.set_page_config(
    page_title="Healthcare Research Assistant",
    page_icon="🩺",
    layout="wide"
)


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:

    st.title("🩺 Healthcare RAG")

    st.write(
        "A research assistant that retrieves "
        "evidence from trusted diabetes literature "
        "and generates grounded answers."
    )

    st.divider()

    st.subheader("RAG Pipeline")

    pipeline_steps = [
        "📄 Medical Literature",
        "✂️ Text Chunking",
        "🔢 Embeddings",
        "🗄️ ChromaDB",
        "🔎 Retrieval",
        "🎯 Cross-Encoder Reranking",
        "🤖 Gemini",
        "📚 Grounded Answer"
    ]

    for index, step in enumerate(pipeline_steps):

        st.write(step)

        if index < len(pipeline_steps) - 1:
            st.caption("↓")

    st.divider()

    st.caption(
        "Knowledge Source: NIDDK Diabetes in America"
    )


st.title("🩺 Healthcare Research Assistant")

st.write(
    "Ask questions about diabetes research using "
    "evidence retrieved from trusted medical literature."
)


st.info(
    "⚠️ **Research Use Only**\n\n"
    "This assistant provides information from "
    "retrieved medical literature for research and "
    "educational purposes. It is not a substitute "
    "for professional medical advice, diagnosis, "
    "or treatment."
)


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input(
    "Ask a diabetes research question..."
)


if question:

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching medical literature..."
        ):

            result = answer_question(question)

        st.markdown(result["answer"])

        st.subheader("📚 Retrieved Evidence")

        for index, evidence in enumerate(
            result["evidence"],
            start=1
        ):

            source = evidence["source"]
            page = evidence["page"]

            with st.expander(
                f"Evidence {index} • "
                f"{source} • Page {page}"
            ):

                st.write(evidence["text"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"]
        }
    )