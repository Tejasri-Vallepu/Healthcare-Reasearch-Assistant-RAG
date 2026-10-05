import json
from pathlib import Path
import re

from config import PROCESSED_DIR


# ---------------------------------------------------------
# CHUNKING SETTINGS
# ---------------------------------------------------------

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80


# ---------------------------------------------------------
# SPLIT DOCUMENT INTO PAGES
# ---------------------------------------------------------

def split_into_pages(text: str) -> list[dict]:
    """
    Split a document using PAGE markers.

    Each page keeps:
    - page number
    - page text
    """

    page_pattern = r"--- PAGE (\d+) ---"

    matches = list(
        re.finditer(
            page_pattern,
            text
        )
    )

    pages = []

    for index, match in enumerate(matches):

        page_number = int(
            match.group(1)
        )

        start = match.end()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(text)

        page_text = text[start:end].strip()

        pages.append(
            {
                "page": page_number,
                "text": page_text
            }
        )

    return pages


# ---------------------------------------------------------
# SPLIT TEXT INTO PARAGRAPHS
# ---------------------------------------------------------

def split_into_paragraphs(text: str) -> list[str]:
    """
    Split page text into paragraphs.
    """

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


# ---------------------------------------------------------
# SPLIT LARGE PARAGRAPH
# ---------------------------------------------------------

def split_large_paragraph(
    paragraph: str,
    max_words: int
) -> list[str]:
    """
    Split a paragraph into smaller pieces if necessary.
    """

    words = paragraph.split()

    if len(words) <= max_words:
        return [paragraph]

    pieces = []

    for start in range(
        0,
        len(words),
        max_words
    ):

        pieces.append(
            " ".join(
                words[start:start + max_words]
            )
        )

    return pieces


# ---------------------------------------------------------
# CREATE CHUNKS
# ---------------------------------------------------------

def create_chunks(text: str) -> list[dict]:
    """
    Create chunks while preserving page information.
    """

    pages = split_into_pages(text)

    chunks = []

    chunk_number = 1

    for page_data in pages:

        page_number = page_data["page"]

        paragraphs = split_into_paragraphs(
            page_data["text"]
        )

        processed_paragraphs = []

        for paragraph in paragraphs:

            processed_paragraphs.extend(
                split_large_paragraph(
                    paragraph,
                    CHUNK_SIZE
                )
            )

        current_words = []

        for paragraph in processed_paragraphs:

            words = paragraph.split()

            # If the next paragraph would exceed the
            # maximum chunk size, save the current chunk.
            if (
                len(current_words) + len(words)
                > CHUNK_SIZE
                and current_words
            ):

                chunks.append(
                    {
                        "chunk_id": chunk_number,
                        "text": " ".join(
                            current_words
                        ),
                        "page": page_number
                    }
                )

                chunk_number += 1

                # Keep the final CHUNK_OVERLAP words.
                current_words = current_words[
                    -CHUNK_OVERLAP:
                ]

            current_words.extend(words)

        # Save remaining content from the page.
        if current_words:

            chunks.append(
                {
                    "chunk_id": chunk_number,
                    "text": " ".join(
                        current_words
                    ),
                    "page": page_number
                }
            )

            chunk_number += 1

    return chunks


# ---------------------------------------------------------
# PROCESS ALL DOCUMENTS
# ---------------------------------------------------------

def process_all_documents():
    """
    Create chunks for every cleaned document
    and save them as JSON files.
    """

    cleaned_text_dir = (
        PROCESSED_DIR / "cleaned_text"
    )

    chunks_dir = (
        PROCESSED_DIR / "chunks"
    )

    chunks_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    text_files = sorted(
        cleaned_text_dir.glob("*.txt")
    )

    if not text_files:

        print(
            "No cleaned text files were found."
        )

        return

    print(
        f"Found {len(text_files)} documents.\n"
    )

    total_chunks = 0

    for text_path in text_files:

        print(
            f"Chunking: {text_path.name}"
        )

        text = text_path.read_text(
            encoding="utf-8"
        )

        chunks = create_chunks(text)

        # Document name without .txt
        document_name = text_path.stem

        # Add useful metadata to every chunk.
        for chunk in chunks:

            chunk["chunk_id"] = (
                f"{document_name}_"
                f"chunk_{chunk['chunk_id']:04d}"
            )

            chunk["source"] = document_name

        # JSON output path.
        output_path = (
            chunks_dir
            / f"{document_name}.json"
        )

        output_path.write_text(
            json.dumps(
                chunks,
                indent=4,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        print(
            f"Created {len(chunks)} chunks"
        )

        print(
            f"Saved -> {output_path}\n"
        )

        total_chunks += len(chunks)

    print("=" * 60)

    print(
        f"Total chunks created: {total_chunks}"
    )

    print("=" * 60)


# ---------------------------------------------------------
# PROGRAM ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    process_all_documents()