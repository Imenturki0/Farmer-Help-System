from pathlib import Path
from docling.document_converter import DocumentConverter


PDF_FOLDER = Path("data/raw/pdfs")

converter = DocumentConverter()


def extract_test(pdf_path: Path):
    print(f"\n[TEST] Processing: {pdf_path.name}")

    result = converter.convert(str(pdf_path))
    document = result.document

    print("\n========== EXTRACTED ITEMS ==========\n")

    for item, level in document.iterate_items():

        item_type = type(item).__name__

        text = getattr(item, "text", None)

        print(
            f"TYPE: {item_type}"
            f" | LEVEL: {level}"
            f" | TEXT: {repr(text)}"
        )

    print("\n========== END ==========\n")


if __name__ == "__main__":

    pdfs = list(PDF_FOLDER.glob("*.pdf"))

    if not pdfs:
        print("No PDF found.")
        raise SystemExit

    extract_test(pdfs[0])