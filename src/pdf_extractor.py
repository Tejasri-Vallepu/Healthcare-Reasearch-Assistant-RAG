from pathlib import Path

import pymupdf

from config import DOCUMENTS_DIR, EXTRACTED_TEXT_DIR


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from every page of a PDF.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file.

    Returns
    -------
    str
        Complete extracted text from the PDF.
    """

    # Open the PDF document.
    document = pymupdf.open(pdf_path)

    extracted_pages = []

    # Process every page in the PDF.
    for page_number, page in enumerate(document, start=1):

        # Extract text from the current page.
        page_text = page.get_text("text")

        # Keep the page number with the extracted text.
        # This will help us provide source/page citations later.
        extracted_pages.append(
            f"\n--- PAGE {page_number} ---\n"
            f"{page_text}"
        )

    # Close the PDF after extraction.
    document.close()

    # Combine all pages into one text string.
    return "\n".join(extracted_pages)


def save_extracted_text(pdf_path: Path) -> Path:
    """
    Extract text from one PDF and save it as a TXT file.
    """

    # Extract the PDF text.
    extracted_text = extract_text_from_pdf(pdf_path)

    # Keep the original filename but change .pdf to .txt.
    output_filename = pdf_path.stem + ".txt"

    output_path = EXTRACTED_TEXT_DIR / output_filename

    # Save the extracted text using UTF-8 encoding.
    output_path.write_text(
        extracted_text,
        encoding="utf-8"
    )

    return output_path


def extract_all_pdfs():
    """
    Extract text from every PDF in the documents directory.
    """

    # Find all PDF files.
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files were found.")
        return

    print(f"Found {len(pdf_files)} PDF files.\n")

    # Process each PDF one by one.
    for pdf_path in pdf_files:

        print(f"Processing: {pdf_path.name}")

        try:
            output_path = save_extracted_text(pdf_path)

            print(
                f"Successfully extracted text -> "
                f"{output_path.name}\n"
            )

        except Exception as error:

            print(
                f"Failed to process {pdf_path.name}: "
                f"{error}\n"
            )


if __name__ == "__main__":
    extract_all_pdfs()