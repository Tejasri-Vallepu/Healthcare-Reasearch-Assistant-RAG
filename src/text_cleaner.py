from pathlib import Path
import re

from config import EXTRACTED_TEXT_DIR, PROCESSED_DIR


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving
    page markers and important medical information.
    """

    # -----------------------------------------------------
    # 1. Protect page markers
    # -----------------------------------------------------
    # Convert variations such as:
    #
    # --- PAGE 1 ---1-1
    #
    # into:
    #
    # --- PAGE 1 ---
    #
    # 1-1
    #
    text = re.sub(
        r"--- PAGE (\d+) ---",
        r"\n--- PAGE \1 ---\n",
        text
    )

    # -----------------------------------------------------
    # 2. Fix words broken across PDF lines
    # -----------------------------------------------------
    #
    # Example:
    #
    # defi-
    # nition
    #
    # becomes:
    #
    # definition
    #
    text = re.sub(
        r"(?<=\w)-\s*\n\s*(?=\w)",
        "",
        text
    )

    # -----------------------------------------------------
    # 3. Normalize spaces and tabs
    # -----------------------------------------------------

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # -----------------------------------------------------
    # 4. Normalize excessive blank lines
    # -----------------------------------------------------

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    # -----------------------------------------------------
    # 5. Remove spaces from blank lines
    # -----------------------------------------------------

    text = re.sub(
        r" *\n *",
        "\n",
        text
    )

    # -----------------------------------------------------
    # 6. Ensure PAGE markers are separated
    # -----------------------------------------------------

    text = re.sub(
        r"--- PAGE (\d+) ---",
        r"\n--- PAGE \1 ---\n",
        text
    )

    # -----------------------------------------------------
    # 7. Remove excessive blank lines again
    # -----------------------------------------------------

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    # -----------------------------------------------------
    # 8. Remove leading/trailing whitespace
    # -----------------------------------------------------

    text = text.strip()

    return text


# ---------------------------------------------------------
# PROCESS ONE FILE
# ---------------------------------------------------------

def clean_text_file(
    input_path: Path,
    output_path: Path
):
    """
    Read one extracted text file, clean it,
    and save the cleaned version.
    """

    raw_text = input_path.read_text(
        encoding="utf-8"
    )

    cleaned_text = clean_text(
        raw_text
    )

    output_path.write_text(
        cleaned_text,
        encoding="utf-8"
    )


# ---------------------------------------------------------
# PROCESS ALL FILES
# ---------------------------------------------------------

def clean_all_text_files():
    """
    Clean every extracted TXT file.
    """

    cleaned_text_dir = (
        PROCESSED_DIR / "cleaned_text"
    )

    cleaned_text_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    text_files = sorted(
        EXTRACTED_TEXT_DIR.glob("*.txt")
    )

    if not text_files:
        print(
            "No extracted text files were found."
        )
        return

    print(
        f"Found {len(text_files)} text files.\n"
    )

    for input_path in text_files:

        output_path = (
            cleaned_text_dir
            / input_path.name
        )

        print(
            f"Cleaning: {input_path.name}"
        )

        try:

            clean_text_file(
                input_path,
                output_path
            )

            print(
                f"Successfully cleaned -> "
                f"{output_path.name}\n"
            )

        except Exception as error:

            print(
                f"Failed to clean "
                f"{input_path.name}: {error}\n"
            )


# ---------------------------------------------------------
# PROGRAM ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    clean_all_text_files()