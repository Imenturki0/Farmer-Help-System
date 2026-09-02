from pathlib import Path
from typing import Any
import json

from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer

from ingestion.file_state import (
    get_file_hash,
    load_state,
    save_state,
)

from app.services.vector_db import QdrantVectorDB


# ============================================================
# CONFIG
# ============================================================

PDF_FOLDER = Path("data/raw/pdfs")

CHUNKS_FILE = Path(
    "data/processed/chunks.json"
)

EMBEDDING_MODEL = (
    "BAAI/bge-base-en-v1.5"
)

COLLECTION_NAME = (
    "farming_docs"
)

# Maximum approximate words in one chunk.
MAX_WORDS = 250

# Small overlap between chunks.
# This helps preserve context across boundaries.
CHUNK_OVERLAP_WORDS = 40

# Very small normal text elements can be ignored.
# IMPORTANT:
# List items are kept even when they are short.
MIN_TEXT_WORDS = 10


# ============================================================
# MODELS
# ============================================================

converter = DocumentConverter()

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

db = QdrantVectorDB(
    collection=COLLECTION_NAME
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Normalize extracted text.

    Removes:
        - null characters
        - excessive whitespace
        - line-break artifacts
    """

    return " ".join(
        text.replace("\x00", " ").split()
    ).strip()


# ============================================================
# ITEM TYPE HELPERS
# ============================================================

def get_item_type(item: Any) -> str:
    """
    Return normalized Docling item type.
    """

    return type(item).__name__.lower()


def is_heading(item_type: str) -> bool:
    """
    Detect section/header/title elements.
    """

    return (
        "sectionheader" in item_type
        or "heading" in item_type
        or item_type == "titleitem"
        or item_type == "title"
    )


def is_list_item(item_type: str) -> bool:
    """
    Detect Docling list items.
    """

    return "listitem" in item_type


def is_table(item_type: str) -> bool:
    """
    Detect table elements.

    Tables are currently not converted into normal text
    because their structure needs special handling.
    """

    return "table" in item_type


def is_picture(item_type: str) -> bool:
    """
    Detect images/pictures.
    """

    return "picture" in item_type


# ============================================================
# PAGE PROVENANCE
# ============================================================

def get_pages(item: Any) -> list[int]:
    """
    Extract page numbers from Docling provenance.
    """

    pages = []

    for provenance in getattr(
        item,
        "prov",
        [],
    ):

        page = getattr(
            provenance,
            "page_no",
            None,
        )

        if page is not None:

            try:
                pages.append(
                    int(page)
                )

            except (
                TypeError,
                ValueError,
            ):
                pass

    return sorted(
        set(pages)
    )


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_document(
    pdf_path: str | Path,
) -> dict[str, Any]:
    """
    Extract a PDF using Docling.

    The extractor preserves document structure:

        PDF
         ↓
        headings
        paragraphs
        list items
        page provenance

    We intentionally do NOT perform semantic embeddings here.
    """

    pdf_path = Path(pdf_path)

    print(
        f"\n[EXTRACTION] Reading: "
        f"{pdf_path.name}"
    )

    result = converter.convert(
        str(pdf_path)
    )

    document = result.document

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    title = getattr(
        document,
        "name",
        None,
    )

    if not title:
        title = pdf_path.stem

    # --------------------------------------------------------
    # ELEMENTS
    # --------------------------------------------------------

    elements = []

    current_section = None

    current_section_level = None

    for item, level in document.iterate_items():

        item_type = get_item_type(
            item
        )

        text = getattr(
            item,
            "text",
            None,
        )

        # ====================================================
        # HEADING
        # ====================================================

        if is_heading(item_type):

            if text:

                text = clean_text(
                    str(text)
                )

                if text:

                    current_section = text

                    current_section_level = level

                    print(
                        f"[EXTRACTION] "
                        f"SECTION → {current_section}"
                    )

            continue

        # ====================================================
        # PICTURE
        # ====================================================

        if is_picture(item_type):

            print(
                "[EXTRACTION] "
                f"PictureItem detected "
                f"(no text): {pdf_path.name}"
            )

            continue

        # ====================================================
        # TABLE
        # ====================================================

        if is_table(item_type):

            print(
                "[EXTRACTION] "
                f"Table detected: "
                f"{pdf_path.name}"
            )

            # We skip raw table objects for now.
            #
            # Later we can add a dedicated table-to-text
            # conversion step.
            continue

        # ====================================================
        # IGNORE ITEMS WITHOUT TEXT
        # ====================================================

        if not text:
            continue

        text = clean_text(
            str(text)
        )

        if not text:
            continue

        # ====================================================
        # LIST ITEMS
        # ====================================================

        if is_list_item(item_type):

            # IMPORTANT:
            # Keep list items even if short.
            #
            # Example:
            # "A layer of 1 kg of straw"
            #
            # This can be valuable retrieval information.

            elements.append(
                {
                    "text": text,
                    "type": "list",
                    "level": level,
                    "pages": get_pages(item),
                    "section": current_section,
                    "section_level": current_section_level,
                }
            )

            continue

        # ====================================================
        # NORMAL TEXT
        # ====================================================

        if len(text.split()) < MIN_TEXT_WORDS:

            print(
                f"[EXTRACTION] "
                f"Skipping short text: "
                f"{text[:80]!r}"
            )

            continue

        elements.append(
            {
                "text": text,
                "type": "text",
                "level": level,
                "pages": get_pages(item),
                "section": current_section,
                "section_level": current_section_level,
            }
        )

    print(
        f"[EXTRACTION] "
        f"Extracted {len(elements)} elements"
    )

    return {
        "title": title,
        "elements": elements,
    }


# ============================================================
# CHUNK TEXT HELPERS
# ============================================================

def word_count(text: str) -> int:
    """
    Approximate word count.
    """

    return len(
        text.split()
    )


def split_large_text(
    text: str,
    max_words: int,
) -> list[str]:
    """
    Split an individual oversized text element.

    Normally a paragraph should already be smaller than
    MAX_WORDS, but PDFs sometimes contain huge text blocks.
    """

    words = text.split()

    if len(words) <= max_words:
        return [text]

    parts = []

    for start in range(
        0,
        len(words),
        max_words,
    ):

        part = " ".join(
            words[
                start:start + max_words
            ]
        )

        if part:
            parts.append(part)

    return parts


# ============================================================
# CHUNK BUILDER
# ============================================================

def create_chunk(
    texts: list[str],
    pages: set[int],
    section: str | None,
    source: str,
    title: str,
    chunk_index: int,
) -> dict[str, Any]:
    """
    Create one canonical chunk.
    """

    text = " ".join(
        texts
    ).strip()

    return {
        "chunk_id": (
            f"{source}:{chunk_index}"
        ),
        "text": text,
        "pages": sorted(
            pages
        ),
        "source": source,
        "title": title,
        "section": section,
    }


# ============================================================
# STRUCTURE-AWARE CHUNKING
# ============================================================

def chunk_elements(
    elements: list[dict[str, Any]],
    source: str,
    title: str,
    max_words: int = MAX_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[dict[str, Any]]:
    """
    Production-oriented structure-aware chunking.

    Strategy:

        1. Respect document sections.
        2. Keep paragraphs together.
        3. Keep related list items together.
        4. Never exceed MAX_WORDS when possible.
        5. Split oversized paragraphs safely.
        6. Add small overlap between chunks.

    No embeddings are used here.

    This is much faster than semantic chunking and is
    deterministic/reproducible.
    """

    if not elements:
        return []

    chunks = []

    current_texts: list[str] = []

    current_pages: set[int] = set()

    current_section: str | None = None

    current_words = 0

    chunk_index = 0

    # --------------------------------------------------------
    # FINALIZE CURRENT CHUNK
    # --------------------------------------------------------

    def finalize_chunk():

        nonlocal chunk_index
        nonlocal current_texts
        nonlocal current_pages
        nonlocal current_words

        if not current_texts:
            return

        chunks.append(
            create_chunk(
                texts=current_texts,
                pages=current_pages,
                section=current_section,
                source=source,
                title=title,
                chunk_index=chunk_index,
            )
        )

        chunk_index += 1

    # --------------------------------------------------------
    # OVERLAP
    # --------------------------------------------------------

    def get_overlap_text() -> str | None:

        if overlap_words <= 0:
            return None

        if not current_texts:
            return None

        combined = " ".join(
            current_texts
        )

        words = combined.split()

        if not words:
            return None

        overlap = words[
            -overlap_words:
        ]

        return " ".join(
            overlap
        )

    # --------------------------------------------------------
    # PROCESS ELEMENTS
    # --------------------------------------------------------

    for element in elements:

        text = element["text"]

        element_type = element.get(
            "type",
            "text",
        )

        section = element.get(
            "section"
        )

        pages = set(
            element.get(
                "pages",
                []
            )
        )

        element_words = word_count(
            text
        )

        # ====================================================
        # SECTION CHANGE
        # ====================================================

        if (
            current_texts
            and section != current_section
        ):

            finalize_chunk()

            overlap_text = (
                get_overlap_text()
            )

            current_texts = []

            current_pages = set()

            current_words = 0

            # We intentionally do NOT carry overlap
            # across sections.
            #
            # Sections are strong semantic boundaries.

            current_section = section

        elif not current_texts:

            current_section = section

        # ====================================================
        # OVERSIZED ELEMENT
        # ====================================================

        if element_words > max_words:

            # Flush current chunk first.
            if current_texts:

                finalize_chunk()

                current_texts = []

                current_pages = set()

                current_words = 0

            large_parts = split_large_text(
                text,
                max_words,
            )

            for part_index, part in enumerate(
                large_parts
            ):

                part_pages = set(
                    pages
                )

                chunks.append(
                    create_chunk(
                        texts=[part],
                        pages=part_pages,
                        section=current_section,
                        source=source,
                        title=title,
                        chunk_index=chunk_index,
                    )
                )

                chunk_index += 1

            continue

        # ====================================================
        # FIRST ELEMENT
        # ====================================================

        if not current_texts:

            current_texts = [text]

            current_pages = set(
                pages
            )

            current_words = element_words

            continue

        # ====================================================
        # WOULD EXCEED MAX WORDS
        # ====================================================

        if (
            current_words + element_words
            > max_words
        ):

            # Save current chunk.
            finalize_chunk()

            # ------------------------------------------------
            # Start next chunk
            # ------------------------------------------------

            current_texts = []

            current_pages = set()

            current_words = 0

            # We only overlap normal text.
            # Avoid carrying a list item by itself.
            overlap_text = None

            if (
                element_type == "text"
                and overlap_words > 0
            ):

                # Use the tail of the previous chunk.
                previous_text = (
                    " ".join(
                        chunks[-1]["text"].split()
                    )
                )

                previous_words = (
                    previous_text.split()
                )

                if previous_words:

                    overlap_text = " ".join(
                        previous_words[
                            -overlap_words:
                        ]
                    )

            if overlap_text:

                overlap_count = word_count(
                    overlap_text
                )

                current_texts = [
                    overlap_text,
                    text,
                ]

                current_pages.update(
                    pages
                )

                current_words = (
                    overlap_count
                    + element_words
                )

            else:

                current_texts = [
                    text
                ]

                current_pages = set(
                    pages
                )

                current_words = (
                    element_words
                )

            continue

        # ====================================================
        # ADD ELEMENT
        # ====================================================

        current_texts.append(
            text
        )

        current_pages.update(
            pages
        )

        current_words += element_words

    # ========================================================
    # FINAL CHUNK
    # ========================================================

    finalize_chunk()

    return chunks


# ============================================================
# PROCESS ONE PDF
# ============================================================

def process_pdf(
    pdf_path: str | Path,
) -> list[dict[str, Any]]:
    """
    Process one PDF:

        PDF
         ↓
        Docling
         ↓
        structured extraction
         ↓
        section-aware chunking
         ↓
        final chunks

    Embeddings are NOT generated here.
    """

    pdf_path = Path(
        pdf_path
    )

    print(
        f"\n[INGESTION] Processing: "
        f"{pdf_path.name}"
    )

    document = extract_document(
        pdf_path
    )

    elements = document[
        "elements"
    ]

    if not elements:

        print(
            f"[INGESTION] No content: "
            f"{pdf_path.name}"
        )

        return []

    chunks = chunk_elements(
        elements=elements,
        source=pdf_path.name,
        title=document["title"],
    )

    print(
        f"[INGESTION] Created "
        f"{len(chunks)} chunks"
    )

    return chunks


# ============================================================
# CHUNKS.JSON
# ============================================================

def load_chunks() -> list[dict]:
    """
    Load canonical chunk dataset.
    """

    if not CHUNKS_FILE.exists():
        return []

    try:

        with CHUNKS_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

        if isinstance(
            data,
            list,
        ):
            return data

        return []

    except (
        OSError,
        json.JSONDecodeError,
    ):

        print(
            "[INGESTION] Invalid "
            "chunks.json. Starting fresh."
        )

        return []


def save_chunks(
    chunks: list[dict],
):
    """
    Atomically save chunks.json.
    """

    CHUNKS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_file = (
        CHUNKS_FILE.with_suffix(
            ".tmp"
        )
    )

    with temp_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            chunks,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temp_file.replace(
        CHUNKS_FILE
    )


# ============================================================
# SOURCE HELPERS
# ============================================================

def remove_source_chunks(
    chunks: list[dict],
    source: str,
) -> list[dict]:
    """
    Remove all chunks belonging to one source.
    """

    return [
        chunk
        for chunk in chunks
        if chunk.get(
            "source"
        ) != source
    ]


# ============================================================
# INGEST ONE PDF
# ============================================================

def ingest_pdf(
    pdf: Path,
    all_chunks: list[dict],
) -> list[dict]:
    """
    Process one PDF and synchronize:

        chunks.json
        Qdrant
    """

    source = pdf.name

    new_chunks = process_pdf(
        pdf
    )

    if not new_chunks:

        print(
            f"[INGESTION] No chunks: "
            f"{source}"
        )

        return all_chunks

    # --------------------------------------------------------
    # Remove old chunks
    # --------------------------------------------------------

    all_chunks = remove_source_chunks(
        all_chunks,
        source,
    )

    # --------------------------------------------------------
    # Add new chunks
    # --------------------------------------------------------

    all_chunks.extend(
        new_chunks
    )

    # --------------------------------------------------------
    # Update Qdrant
    # --------------------------------------------------------

    print(
        f"[INGESTION] Indexing "
        f"{len(new_chunks)} chunks..."
    )

    db.upsert_chunks(
        embedding_model,
        new_chunks,
        source,
    )

    print(
        f"[INGESTION] Indexed: "
        f"{source} "
        f"({len(new_chunks)} chunks)"
    )

    return all_chunks


# ============================================================
# FULL INGESTION PIPELINE
# ============================================================

def run_pipeline():
    """
    Synchronize PDFs with:

        PDFs
          ↓
        Docling
          ↓
        structure-aware chunks
          ↓
        chunks.json
          ↓
        Qdrant

    Handles:

        - new PDFs
        - modified PDFs
        - unchanged PDFs
        - deleted PDFs
    """

    print(
        "\n=================================================="
    )

    print(
        "[INGESTION] Starting synchronization"
    )

    print(
        "=================================================="
    )

    PDF_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    old_state = load_state()

    new_state = {}

    all_chunks = load_chunks()

    current_files = set()

    # ========================================================
    # NEW / MODIFIED / UNCHANGED
    # ========================================================

    for pdf in sorted(
        PDF_FOLDER.glob("*.pdf")
    ):

        source = pdf.name

        current_files.add(
            source
        )

        file_hash = get_file_hash(
            pdf
        )

        # ----------------------------------------------------
        # UNCHANGED
        # ----------------------------------------------------

        if (
            source in old_state
            and old_state[source]
            == file_hash
        ):

            print(
                f"[INGESTION] Unchanged: "
                f"{source}"
            )

            new_state[
                source
            ] = file_hash

            continue

        # ----------------------------------------------------
        # NEW / MODIFIED
        # ----------------------------------------------------

        print(
            f"\n[INGESTION] New/modified: "
            f"{source}"
        )

        try:

            all_chunks = ingest_pdf(
                pdf,
                all_chunks,
            )

            # Mark as processed only after
            # successful Qdrant indexing.
            new_state[
                source
            ] = file_hash

        except Exception as exc:

            print(
                f"[INGESTION] ERROR "
                f"{source}: {exc}"
            )

            # Keep old state so the PDF
            # will be retried next time.
            if source in old_state:

                new_state[
                    source
                ] = old_state[
                    source
                ]

    # ========================================================
    # DELETED FILES
    # ========================================================

    deleted = (
        set(old_state.keys())
        - current_files
    )

    for source in deleted:

        print(
            f"\n[INGESTION] Deleted: "
            f"{source}"
        )

        try:

            # ------------------------------------------------
            # Remove from Qdrant
            # ------------------------------------------------

            db.delete_by_source(
                source
            )

            # ------------------------------------------------
            # Remove from chunks.json
            # ------------------------------------------------

            all_chunks = (
                remove_source_chunks(
                    all_chunks,
                    source,
                )
            )

            print(
                f"[INGESTION] Removed: "
                f"{source}"
            )

        except Exception as exc:

            print(
                f"[INGESTION] ERROR deleting "
                f"{source}: {exc}"
            )

            # Keep state so deletion
            # can be retried.
            new_state[
                source
            ] = old_state[
                source
            ]

    # ========================================================
    # SAVE
    # ========================================================

    save_chunks(
        all_chunks
    )

    save_state(
        new_state
    )

    print(
        "\n=================================================="
    )

    print(
        "[INGESTION] Synchronization complete"
    )

    print(
        f"[INGESTION] Total chunks: "
        f"{len(all_chunks)}"
    )

    print(
        "=================================================="
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_pipeline()